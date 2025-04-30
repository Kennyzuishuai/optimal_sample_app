#!/usr/bin/env python3
"""
optimal_samples_selection.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
后端核心算法 + API 服务
==========================
- **select_optimal_samples()**: 纯函数，可被任何前端（CLI、Electron、Qt、React、Android…）直接调用。
- **/select**: FastAPI JSON 接口，前端（Axios／fetch）POST 参数即可获得结果。
- **CLI**: `python optimal_samples_selection.py --help` 保留命令行单测/运维能力。
- **DB**: 结果自动写入 `results.sqlite3`；通过 `/results/{id}` 查询或 DELETE。

依赖安装
---------
```bash
pip install fastapi uvicorn[standard] ortools numpy sqlalchemy
```
"""
from __future__ import annotations
import itertools, random, json, sys, warnings, argparse, time, logging, os, math # Removed sqlite3, pathlib
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from ortools.sat.python import cp_model  # 高性能 0‑1 MIP
try:
    import psutil # For CPU core count detection
except ImportError:
    psutil = None
    print("Warning: psutil not installed. Cannot automatically determine optimal workers.", file=sys.stderr)

from fastapi import FastAPI, HTTPException, Request # Add Request
from pydantic import BaseModel

# Import pruning utility
try:
    from utils.combo_prune import unique_k_combos
except ImportError:
    # Fallback if utils is not in the path during direct execution
    print("Warning: Could not import unique_k_combos from utils. Pruning disabled.", file=sys.stderr)
    unique_k_combos = None

########################
#  核心算法（最小阈值覆盖） #
########################

def _threshold_set_cover(
    combos: List[Tuple[int, ...]],
    j_subsets: List[Tuple[int, ...]],
    t: int,
    workers: int,
    time_limit: Optional[int] = None,
    progress_callback=None,
    start_time: Optional[float] = None,
    warm_start_hints: Optional[List[int]] = None
) -> Tuple[List[Tuple[int, ...]], float, float]:
    """OR‑Tools CP‑SAT exactly minimise combinations under threshold t. Supports warm start with hints."""
    num_combos = len(combos)
    num_j_subsets = len(j_subsets)
    print(f"Running _threshold_set_cover with {num_combos} k-combinations, {num_j_subsets} j-subsets, t={t}, workers={workers}, warm_start={'Yes' if warm_start_hints and any(warm_start_hints) else 'No'}", file=sys.stderr) # Refined warm start print

    report_progress(0, "初始化求解模型 (CP-SAT)...", start_time, progress_callback)

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}") for i in range(num_combos)]

    # === Optimized Constraint Building for s == j ===
    # Check if k == j (required for the s=j=k fast path).
    # The calling context (select_optimal_samples) ensures s == j when this function is called via the s==j branch.
    # s_size_check gets size j from j_subsets, len(combos[0]) gets size k.
    s_size_check = len(j_subsets[0]) if num_j_subsets > 0 else 0 # Infer j size
    k_size_check = len(combos[0]) if num_combos > 0 else 0 # Infer k size
    is_k_equal_j = (k_size_check == s_size_check) and k_size_check > 0 # Check if k == j and valid sizes

    j_to_cols: List[List[int]] = [[] for _ in range(num_j_subsets)] # Pre-allocate list

    use_symmetry_breaking = True # 默认启用

    # Fast path only applies if k=j=s and t=1
    if is_k_equal_j and t == 1:
        # --- Fast path for k=j=s, t=1 ---
        # A k-combo 'kc' covers a j-subset 'js' iff kc == js.
        # Constraint: For each j_subset 'js', sum(x_i for k_combos[i] == js) >= 1
        print("Using optimized constraint building for s=j, t=1", file=sys.stderr)
        report_progress(1, "优化构建约束 (s=j, t=1)...", start_time, progress_callback)
        # Map each j_subset to its index
        j_subset_to_index: Dict[Tuple[int, ...], int] = {js: idx for idx, js in enumerate(j_subsets)}
        # Map k_combos that *are* j_subsets to the corresponding j_subset index
        for k_idx, kc in enumerate(combos):
            j_idx = j_subset_to_index.get(kc)
            if j_idx is not None:
                # Add the variable index 'k_idx' to the constraint list for j_subset 'j_idx'
                j_to_cols[j_idx].append(k_idx)

        # Add constraints: each j_subset must be covered by at least one selected k_combo (which must equal it)
        constraints_added = 0
        for j_idx, covering_k_indices in enumerate(j_to_cols):
            # ----> 在这里添加打印 <----
            print(f"DEBUG: Checking j_idx={j_idx}, js={j_subsets[j_idx]}, covering_indices={covering_k_indices}", file=sys.stderr)
            if not covering_k_indices:
                 # This j_subset cannot be covered by any of the provided k_combos in this round
                 print(f"Warning: j_subset {j_subsets[j_idx]} (index {j_idx}) cannot be covered by any k_combo in this round's input.", file=sys.stderr)
                 # Raise error or allow solver to determine infeasibility? Let solver handle it.
                 # Add a constraint that is always false to force infeasibility if needed, but sum >= 1 is fine.
                 model.Add(sum(x[i] for i in covering_k_indices) >= t) # Will be Add(0 >= 1) -> infeasible
            else:
                model.Add(sum(x[i] for i in covering_k_indices) >= t)
                constraints_added += 1
        if constraints_added < num_j_subsets:
              print(f"Warning: Only {constraints_added}/{num_j_subsets} j_subsets have potential covering k_combos.", file=sys.stderr)
        print(f"Finished optimized constraint building (s=j, t=1). Added {constraints_added} constraints.", file=sys.stderr)
        use_symmetry_breaking = False # <--- 在 s=j, t=1 时禁用对称破除！
        print("Disabling symmetry breaking constraints for s=j, t=1 case.", file=sys.stderr)
    else:
        # --- Original generic path (s < j or t > 1) ---
        # This path will still be slow for large inputs if s is close to j.
        print("Using generic constraint building (s<j or t>1 or s!=j)", file=sys.stderr)
        report_progress(1, "构建通用约束 (s!=j or t>1)...", start_time, progress_callback)
        s_size = len(j_subsets[0]) if j_subsets else 0 # Get s size from j_subsets
        # Precompute combo_sets only once
        combo_s_sets = [set(itertools.combinations(c, s_size)) for c in combos]
        constraints_added = 0
        for j_idx, js in enumerate(j_subsets):
            needs = []
            st_js = set(itertools.combinations(js, s_size))
            for k_idx, covered_s_set in enumerate(combo_s_sets):
                if not covered_s_set.isdisjoint(st_js):
                    needs.append(k_idx)
            if not needs:
                # Allow solver to determine infeasibility
                print(f"Warning: j_subset {js} (index {j_idx}) cannot be covered by any k_combo.", file=sys.stderr)
            j_to_cols[j_idx] = needs # Store indices for this j_subset
            model.Add(sum(x[i] for i in needs) >= t)
            constraints_added += 1
        print(f"Finished generic constraint building. Added {constraints_added} constraints.", file=sys.stderr)

    # --- Rest of the function remains the same ---

    # 目标：最小化选中的k-组合数量
    model.Minimize(sum(x))

    # 对称破除约束 (Symmetry Breaking)
    if use_symmetry_breaking: # <--- 只有在需要时才添加
        for i in range(1, len(x)):
            model.Add(x[i-1] >= x[i])
        print("Added symmetry breaking constraints.", file=sys.stderr)
    # else: (不需要打印，上面已经打印了禁用信息)

    # 设置求解参数
    solver = cp_model.CpSolver()

    cp_sat_time_budget = time_limit or 25
    p = solver.parameters
    p.max_time_in_seconds  = cp_sat_time_budget
    # Ensure workers are set (passed parameter)
    p.num_search_workers = workers if workers > 0 else 0 # Use passed value, 0 for auto
    p.use_lns              = True
    p.linearization_level  = 2
    p.random_seed          = 42
    print(f"Solver: Setting time={cp_sat_time_budget}s, workers={p.num_search_workers}, use_lns={p.use_lns}, linearization={p.linearization_level}, seed={p.random_seed}", file=sys.stderr)

    # === Warm-start ===
    if warm_start_hints:
        hints_applied = 0
        if len(warm_start_hints) != len(x):
             print(f"Warning: warm_start_hints length ({len(warm_start_hints)}) != number of variables ({len(x)}). Skipping hints.", file=sys.stderr)
        else:
             num_hints = sum(1 for h in warm_start_hints if h)
             print(f"Applying {num_hints} non-zero hints from warm_start_hints list...", file=sys.stderr)
             if num_hints > 0: # Only add hints if there are any non-zero ones
                 for i, hint in enumerate(warm_start_hints):
                     if hint:
                         model.AddHint(x[i], 1)
                         hints_applied += 1
                 print(f"Applied {hints_applied} hints to the model.", file=sys.stderr)
             else:
                 print("Warm start hints list contained all zeros, no hints applied.", file=sys.stderr)

    else:
         print("No warm start hints provided.", file=sys.stderr)

    # 求解
    print(f"Solver: Starting solve...", file=sys.stderr)

    report_progress(10, "模型构建完成，开始求解（此步骤可能需要较长时间）...", start_time, progress_callback)
    status = solver.Solve(model)

    solver_time = solver.WallTime()
    print(f'Solver wall time: {solver_time:.3f} s', file=sys.stderr)

    report_progress(95, "求解完成，处理结果...", start_time, progress_callback)

    response_stats = solver.ResponseStats()
    print(f"Solver Response Stats:\n{response_stats}", file=sys.stderr)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        status_name = solver.StatusName(status)
        error_message = f"求解失败或超时。状态: {status_name} ({status})"
        print(f"Error: {error_message}", file=sys.stderr)
        # Check if infeasible due to uncovered j_subsets detected earlier
        if status == cp_model.INFEASIBLE:
             error_message += ". 可能原因：部分j-子集无法被任何提供的k-组合覆盖（见先前警告）。"
        raise RuntimeError(error_message)

    selected = []
    selected_values = [] # 存储 solver.Value 的结果
    for i, var in enumerate(x):
         val = solver.Value(var)
         selected_values.append(val) # 记录每个变量的值
         if val: # 或者 if val > 0.5 for safety with potential floating point issues
             selected.append(combos[i])

    # ----> 添加打印 <----
    print(f"DEBUG (inside _threshold_set_cover): Checking solver values...", file=sys.stderr)
    num_selected = sum(1 for v in selected_values if v)
    print(f"DEBUG: Number of variables where solver.Value(var) is True/non-zero: {num_selected}", file=sys.stderr)
    print(f"DEBUG: First 20 solver values: {selected_values[:20]}", file=sys.stderr)
    # ----> 结束打印 <----

    objective_value = solver.ObjectiveValue()
    best_bound = solver.BestObjectiveBound()
    print(f"Solver finished. Status: {solver.StatusName(status)}, Objective: {objective_value:.1f}, BestBound: {best_bound:.1f}", file=sys.stderr)
    return selected, objective_value, best_bound

# Import bitmask utility
try:
    from utils.bitmask import generate_masks
except ImportError:
    print("Warning: Could not import generate_masks from utils. Bitmask optimization disabled for greedy.", file=sys.stderr)
    generate_masks = None

# 处理s<j的情况的贪心算法
def _greedy_cover_partial(
    samples: List[int],
    k_combos: List[Tuple[int, ...]], # Note: k_combos might be pruned if s==j, but greedy is only called if s<j
    j: int,
    s: int,
    start_time: float, # 添加 start_time 参数
    progress_callback=None,
    use_bitmask: bool = True, # Add flag to enable/disable bitmask optimization
    beam_width: int = 1 # Add beam_width parameter
) -> Tuple[List[Tuple[int, ...]], List[int]]: # <- 修改返回值类型
    """贪心算法处理s<j的情况，确保每个j-子集中至少有一个s-子集被覆盖。包含 Beam Search 选项和 2-Opt 优化。""" # Updated docstring
    n_samples = len(samples)
    print(f"Running greedy_cover_partial: n={n_samples}, j={j}, s={s}, beam_width={beam_width}", file=sys.stderr) # Removed use_bitmask log

    # Using sparse cumulative count optimization (replaces bitmask/set logic)
    if report_progress:
        report_progress(0, "初始化贪心算法 (稀疏累积计数)...", start_time, progress_callback)

    all_j_subsets_list = list(itertools.combinations(samples, j))
    num_j_subsets = len(all_j_subsets_list)

    if not num_j_subsets:
        if report_progress:
            report_progress(100, "完成：没有需要处理的j-子集", start_time, progress_callback)
        return []

    # --- Pre-calculate data structures for coverage check ---
    j_subset_internals_sets = {js: set(itertools.combinations(js, s)) for js in all_j_subsets_list}
    k_combo_s_subsets_sets = {kc: set(itertools.combinations(kc, s)) for kc in k_combos}
    # Create a mapping from k-combo tuple to its original index for efficient lookup
    k_combo_to_index = {combo: i for i, combo in enumerate(k_combos)}

    # --- Precomputation for Sparse Cumulative Count ---
    if report_progress: report_progress(10, "预计算索引和倒排表...", start_time, progress_callback)
    j_subset_to_index: Dict[Tuple[int, ...], int] = {js: idx for idx, js in enumerate(all_j_subsets_list)}
    # Inverted index: map j_subset index to list of k_combo indices that cover it (via any s-subset)
    j_to_k: Dict[int, List[int]] = {i: [] for i in range(num_j_subsets)}
    # Can we optimize building this?
    # Iterate through k_combos and their s-subsets
    for k_idx, kc in enumerate(k_combos):
        # Find all j_subsets covered by this k_combo
        covered_j_indices = set()
        kc_s_subsets = k_combo_s_subsets_sets[kc] # Use precalculated s-subsets
        # This is the slow part if done naively (checking all j_subsets)
        # Instead, let's iterate through s_subsets of kc and see which j_subsets they belong to? Still hard.
        # Let's try iterating j_subsets and checking coverage by kc
        for js_idx, js in enumerate(all_j_subsets_list):
            # Check if kc covers js (share at least one s-subset)
            if not j_subset_internals_sets[js].isdisjoint(kc_s_subsets):
                j_to_k[js_idx].append(k_idx)
    if report_progress: report_progress(20, "倒排表计算完成", start_time, progress_callback)

    # --- Helper function for 2-Opt: Check full coverage (using set logic) ---
    def _check_full_coverage(current_selection_indices: List[int]) -> bool:
        """Checks if the given list of k-combo indices covers all j-subsets using set logic."""
        if not current_selection_indices: return num_j_subsets == 0

        # Use set logic as bitmask is no longer the primary method here
        selected_combos_for_check = [k_combos[i] for i in current_selection_indices]
        overall_covered_s = set().union(*(k_combo_s_subsets_sets[combo] for combo in selected_combos_for_check))
        return all(not j_subset_internals_sets[js].isdisjoint(overall_covered_s) for js in all_j_subsets_list)


    # --- Main Greedy Loop (Sparse Cumulative Count) ---
    if report_progress: report_progress(25, "开始贪心迭代 (稀疏累积计数)...", start_time, progress_callback)
    satisfied_j_mask = np.zeros(num_j_subsets, dtype=bool)
    num_satisfied = 0
    candidate_k_indices_set = set(range(len(k_combos))) # Use set for faster removal check
    selected_k_indices = [] # Store selected indices in order
    iter_count = 0

    while num_satisfied < num_j_subsets and candidate_k_indices_set:
        iter_count += 1
        unsatisfied_j_indices = np.where(~satisfied_j_mask)[0]
        if not unsatisfied_j_indices.size:
            print(f"Iteration {iter_count}: All j-subsets satisfied.", file=sys.stderr)
            break

        # Calculate scores (counts of newly covered j-subsets) for candidate k-combos
        # Using a dictionary for sparse scores might be faster than large np array + masking
        k_combo_scores: Dict[int, int] = {k_idx: 0 for k_idx in candidate_k_indices_set}
        num_candidate_k = len(candidate_k_indices_set)
        print(f"Iteration {iter_count}: Scoring {num_candidate_k} candidates against {len(unsatisfied_j_indices)} unsatisfied j-subsets...", file=sys.stderr)

        # Iterate through unsatisfied j-subsets and increment score for covering candidate k-combos
        for js_idx in unsatisfied_j_indices:
            # Get k_indices covering this j_subset from precomputed map
            covering_k_list = j_to_k.get(js_idx, [])
            for k_idx in covering_k_list:
                # Increment score only if the k_combo is still a candidate
                if k_idx in k_combo_scores:
                    k_combo_scores[k_idx] += 1

        # Find the best candidate k-combo (highest score)
        if not k_combo_scores: # Should not happen if candidate_k_indices_set is not empty and j_to_k was built correctly
            print(f"Warning: Greedy iteration {iter_count}: k_combo_scores is empty. Candidates: {len(candidate_k_indices_set)}", file=sys.stderr)
            break

        # Find the k_idx with the maximum score
        # max(dict, key=dict.get) finds the key with the max value
        try:
             # Use list comprehension to filter out scores of 0 before finding max, handles tie-breaking arbitrarily
             non_zero_scores = {k: v for k, v in k_combo_scores.items() if v > 0}
             if not non_zero_scores:
                 print(f"Warning: Greedy iteration {iter_count}: No k-combo covers any *new* j-subset.", file=sys.stderr)
                 break # No progress possible
             best_k_idx_iter = max(non_zero_scores, key=non_zero_scores.get)
             best_count_iter = non_zero_scores[best_k_idx_iter]
        except ValueError: # Handles case where k_combo_scores might be empty after filtering?
             print(f"Warning: Greedy iteration {iter_count}: Error finding max score. Scores: {k_combo_scores}", file=sys.stderr)
             break

        # Add the best k-combo to the results
        selected_k_indices.append(best_k_idx_iter)
        candidate_k_indices_set.remove(best_k_idx_iter) # Remove selected index

        # Update satisfied_j_mask efficiently
        # Find indices of unsatisfied j-subsets that are covered by the chosen k_idx
        newly_satisfied_indices = []
        for js_idx in unsatisfied_j_indices: # Iterate only through currently unsatisfied
            # Check if the chosen k_idx is in the precomputed list for this j_subset
            if best_k_idx_iter in j_to_k.get(js_idx, []):
                newly_satisfied_indices.append(js_idx)

        if newly_satisfied_indices:
            satisfied_j_mask[newly_satisfied_indices] = True
            num_satisfied = np.sum(satisfied_j_mask) # Update total count

        if report_progress:
             report_progress(30 + int(60 * (num_satisfied / num_j_subsets)),
                            f"迭代 {iter_count}: 选择 K-组合 #{best_k_idx_iter} (新增 {best_count_iter} 满足). 总满足 {num_satisfied}/{num_j_subsets}",
                            start_time, progress_callback)

    result_combos = [k_combos[i] for i in selected_k_indices] # Get combos from final indices

    # --- Post-processing & 2-Opt ---
    # Calculate final satisfied count using the final mask state
    final_num_satisfied = np.sum(satisfied_j_mask)
    if final_num_satisfied < num_j_subsets:
         print(f"警告：贪心算法主循环未能满足所有 {num_j_subsets} 个j-子集。"
               f"最终满足 {final_num_satisfied} 个。2-Opt 将不执行。", file=sys.stderr)
         # Don't run 2-opt if the initial greedy solution is incomplete
    else:
         # --- 单点贪心剔除优化 ---
         if report_progress: report_progress(90, "开始单点贪心剔除优化...", start_time, progress_callback) # Changed message
         print("Starting single-point greedy removal optimization...", file=sys.stderr)

         current_result_indices = selected_k_indices[:] # Work with indices
         changed = True
         removed_count_spgr = 0 # 统计单点移除数量
         while changed:
             changed = False
             indices_before_pass = len(current_result_indices)
             # Iterate over a copy of the list indices to allow safe removal
             for i in range(len(current_result_indices) - 1, -1, -1): # Iterate backwards for safe removal
                 k_idx_to_test = current_result_indices[i]
                 # Temporarily remove the current k_idx by creating a list without it
                 tmp_indices = current_result_indices[:i] + current_result_indices[i+1:]

                 # Check if coverage still holds without it
                 if _check_full_coverage(tmp_indices):
                     # Permanently remove by updating the list
                     current_result_indices.pop(i) # Remove the element at index i
                     changed = True
                     removed_count_spgr += 1
                     print(f"  SPGR: Removed k_idx {k_idx_to_test}. New size: {len(current_result_indices)}", file=sys.stderr)
                     # Since we removed an element, the indices shifted. Restarting the inner loop check
                     # or continuing backwards is safer than breaking and restarting the outer loop.
                     # Continuing backwards handles multiple removals in one pass.

             indices_after_pass = len(current_result_indices)
             if indices_before_pass > indices_after_pass:
                 print(f"  SPGR Pass complete. Size reduced from {indices_before_pass} to {indices_after_pass}.", file=sys.stderr)
             # The 'while changed' loop will handle restarting passes if any change was made.

         print(f"Single-point greedy removal finished. Removed {removed_count_spgr} combos. Final size: {len(current_result_indices)}", file=sys.stderr)
         result_combos = [k_combos[i] for i in current_result_indices] # Update result_combos based on final indices

    # --- 最后统计信息 ---
    # 确定优化状态
    optimization_status = "skipped" # Default
    if final_num_satisfied >= num_j_subsets:
        if removed_count_spgr > 0:
            optimization_status = "applied (SPGR)"
        elif selected_k_indices: # Check if greedy ran and produced a result
             optimization_status = "applied (no change)"
        # else: optimization_status remains "skipped" if greedy didn't complete

    # 使用固定的算法名称 "sparse" 并包含优化状态
    print(f"贪心算法 (sparse, single-point removal {optimization_status})."
          f" 结果组合数: {len(result_combos)}, 已覆盖 j-子集 {final_num_satisfied}/{num_j_subsets}。",
          file=sys.stderr)

    if report_progress:
        report_progress(95, "贪心及优化完成", start_time, progress_callback) # Keep final progress at 95, updated message

    # Recalculate final selected indices based on the optimized result_combos
    final_selected_k_indices = [k_combo_to_index[c] for c in result_combos]
    return result_combos, final_selected_k_indices # <- 返回组合和索引

# 全局进度报告函数
def report_progress(percent, message, start_time=None, progress_callback=None):
    """
    报告算法进度的全局函数

    Args:
        percent: 进度百分比 (0-100)
        message: 进度消息
        start_time: 开始时间（用于计算已运行时间）
        progress_callback: 可选的外部回调函数
    """
    # 计算已运行时间（如果提供了开始时间）
    elapsed_str = ""
    elapsed_time = 0
    if start_time is not None:
        elapsed_time = time.perf_counter() - start_time
        elapsed_str = f"({elapsed_time:.1f}s)"

    # 将进度信息格式化为JSON并打印到标准输出
    progress_data = {
        "type": "progress",
        "percent": percent,
        "message": f"{message} {elapsed_str}",
         "elapsed_time": elapsed_time
    }
    print(json.dumps(progress_data)) # Corrected indentation
    sys.stdout.flush() # Corrected indentation, Explicitly flush stdout buffer
    # 如果有外部回调函数，也调用它
    if progress_callback:
        progress_callback(percent, f"{message} {elapsed_str}")


##############################
#  理论界限计算 (新增)      #
##############################

def calculate_combinations(n, k):
    """计算组合数 C(n, k)，处理无效输入。"""
    if k < 0 or k > n:
        return 0 # 或者可以抛出 ValueError
    try:
        # math.comb 在 Python 3.8+ 中可用
        return math.comb(n, k)
    except AttributeError:
        # Python 3.8 之前的兼容实现 (可能较慢且处理大数时精度可能有问题)
        if k == 0 or k == n:
            return 1
        if k > n // 2:
            k = n - k

        res = 1
        for i in range(k):
            res = res * (n - i) // (i + 1)
        return res
    except ValueError:
         # math.comb 对负数输入会抛出 ValueError
         return 0


def calculate_theoretical_bounds(n: int, k: int, j: int, s: int, t: int) -> Dict[str, Any]:
    """
    根据提供的公式计算阈值t覆盖问题的理论最优解|OPT|的上下界。

    Args:
        n: 样本总数 (samples)
        k: k-组合的大小
        j: j-子集的大小
        s: s-子集的大小
        t: 覆盖阈值

    Returns:
        一个包含 delta, total_j_subsets, lower_bound, upper_bound 的字典。
        如果 delta 为 0 (无法覆盖), 返回 None 或部分结果。
    """
    result = {
        "delta": None,
        "total_j_subsets": None,
        "lower_bound": None,
        "upper_bound": None,
        "notes": ""
    }

    # --- 输入参数校验 (基础) ---
    if not (isinstance(n, int) and isinstance(k, int) and isinstance(j, int) and isinstance(s, int) and isinstance(t, int)):
        result["notes"] = "错误: 所有参数必须是整数。"
        return result
    if not (n > 0 and k > 0 and j > 0 and s > 0 and t >= 0):
         result["notes"] = "错误: n, k, j, s 必须为正整数, t 必须为非负整数。"
         return result
    if not (s <= j <= k <= n):
        result["notes"] = "错误: 必须满足 s <= j <= k <= n。"
        return result
    if t == 0:
        result["notes"] = "t=0 时，最优解为 0 (不需要选择任何组合)。"
        result["delta"] = 0 # 或者根据下面计算
        result["total_j_subsets"] = calculate_combinations(n, j)
        result["lower_bound"] = 0
        result["upper_bound"] = 0
        return result

    # --- 计算总 j-子集数 ---
    try:
        total_j_subsets = calculate_combinations(n, j)
        result["total_j_subsets"] = total_j_subsets
        if total_j_subsets == 0:
             result["notes"] = "没有 j-子集需要覆盖 (C(n, j) = 0)。"
             result["lower_bound"] = 0
             result["upper_bound"] = 0
             # Delta 可能仍然可以计算
    except ValueError as e:
        result["notes"] = f"计算 C(n={n}, j={j}) 出错: {e}"
        return result

    # --- 处理 s = j 的特例 ---
    if s == j:
        # 理论上上界和下界都等于 t * C(n, j) / C(k,s) * (ln(C(k,s)*C(n-s,j-s)) + 1) ?
        # 图片中的 Delta 公式是针对通用情况 s < j 的。
        # 当 s = j 时，一个 k-组合覆盖一个 j-子集当且仅当它们完全相同 (如果 k = j)。
        # 如果 k > j = s，情况会复杂。但原始代码的 s=j 优化似乎假设了 k=j=s。
        # 我们需要确认 s=j 情况下的 Delta 定义。
        # 按照原始代码的逻辑和 Set Cover 问题的标准界限，当 s=j (且常假设 k=j)，
        # 每个 j-子集只被一个 k-组合（即它自己）所覆盖。
        # 此时覆盖一个 j-子集的 "贡献" 应该是 1。
        # Set Cover 的界限公式为 ceil(N/Delta) <= |OPT| <= ceil(N/Delta) * (ln max_freq + 1)
        # 其中 N 是要覆盖的元素总数 (这里是 C(n, j))，Delta 是单个集合最多能覆盖的新元素的数量。
        # 但这里的 Delta 定义是 C(k,s)C(n-s,j-s)，这衡量了一个 k-组合能覆盖多少个 j-子集（通过共享 s-子集）。
        # 让我们严格按照你提供的公式中的逻辑（而不是原始代码的特例注释）：
        try:
            comb_k_s = calculate_combinations(k, s) # s=j -> C(k,j)
            # 检查 n-s 和 j-s 是否有效
            if (n - s) < (j - s) or (j-s) < 0: # s=j -> n-j < 0 or 0 < 0. 仅当 n=j 时 n-j=0
                 comb_n_minus_s_j_minus_s = 0 if n != j else 1 # C(n-j, 0) = 1 if n>=j, 0 otherwise.
                 if n < j: # This case is invalid by initial checks (j<=k<=n)
                    comb_n_minus_s_j_minus_s = 0
                 elif n==j:
                    comb_n_minus_s_j_minus_s = 1
                 else: # n > j
                    comb_n_minus_s_j_minus_s = calculate_combinations(n - j, 0) # = 1

                 # result["notes"] = f"无法计算 C(n-s, j-s) 因为 n-s ({n-s}) < j-s ({j-s}) 或 j-s < 0。"
            else:
                 comb_n_minus_s_j_minus_s = calculate_combinations(n - s, j - s) # s=j -> C(n-j, 0) = 1

            # Delta 计算 (通用公式，即使 s=j)
            delta = comb_k_s * comb_n_minus_s_j_minus_s # -> C(k,j) * 1 = C(k,j)
            result["delta"] = delta
            result["notes"] = f"情况 s=j={s}: Delta = C(k,s)*C(n-s,j-s) = C({k},{j})*C({n-j},0) = {delta}."

            if delta <= 0:
                result["notes"] += f" Delta = {delta} <= 0. 上下界无意义或为无穷。"
                # Keep bounds as None
            elif total_j_subsets is not None:
                 base_term = (t * total_j_subsets) / delta
                 result["lower_bound"] = math.ceil(base_term)
                 try:
                     log_factor = math.log(delta) + 1
                     # 遵循图片公式 ceil(t*C(n,j)/Delta) * (ln Delta + 1)
                     # 使用已计算的向上取整的 lower_bound
                     upper_bound_float = result["lower_bound"] * log_factor
                     result["upper_bound"] = upper_bound_float
                 except ValueError as e:
                     result["notes"] += f" 计算上界时出错 (可能 log({delta}) 无效?): {e}"

        except ValueError as e:
             result["notes"] += f" 计算 Delta (s=j case) 时出错: {e}"
        except Exception as e:
             result["notes"] += f" 计算界限 (s=j case) 时发生未知错误: {e}"


    # --- 处理 s < j 的通用情况 ---
    else: # s < j
        try:
            comb_k_s = calculate_combinations(k, s)
            # 检查 n-s 和 j-s 是否有效
            if (n - s) < (j - s) or (j-s) < 0:
                 comb_n_minus_s_j_minus_s = 0
                 result["notes"] = f"无法计算 C(n-s, j-s) 因为 n-s ({n-s}) < j-s ({j-s}) 或 j-s < 0。"
            else:
                 comb_n_minus_s_j_minus_s = calculate_combinations(n - s, j - s)

            delta = comb_k_s * comb_n_minus_s_j_minus_s
            result["delta"] = delta
            result["notes"] = f"通用情况 s={s} < j={j}。"

            if delta <= 0:
                result["notes"] += f" Delta = {delta} <= 0, 单个 k-组合无法触及任何 j-子集。上下界无意义或为无穷。"
                # 在这种情况下，如果 total_j_subsets > 0 且 t > 0，问题是不可行的
                # 保持 bounds 为 None
            elif total_j_subsets is not None: # 确保 C(n, j) 已成功计算
                base_term = (t * total_j_subsets) / delta
                result["lower_bound"] = math.ceil(base_term)

                # 计算上界 (ln(Delta) + 1)
                try:
                    # math.log 是自然对数 (ln)
                    log_factor = math.log(delta) + 1
                    # 遵循图片公式 ceil(t*C(n,j)/Delta) * (ln Delta + 1)
                    # 使用已计算的向上取整的 lower_bound
                    upper_bound_float = result["lower_bound"] * log_factor # 用 lower_bound 乘以因子
                    result["upper_bound"] = upper_bound_float # 报告这个理论浮点上界
                    # 或者，如果严格需要整数上界: result["upper_bound"] = math.ceil(upper_bound_float)

                except ValueError as e:
                     result["notes"] += f" 计算上界时出错 (可能 log({delta}) 无效?): {e}"

        except ValueError as e:
             result["notes"] += f" 计算 Delta 时出错: {e}"
        except Exception as e: # 捕获其他意外错误
             result["notes"] += f" 计算界限时发生未知错误: {e}"


    return result


#######################
#  外部可调函数       #
#######################

def select_optimal_samples(
    m: int,
    n: int,
    k: int,
    j: int,
    s: int,
    t: int = 1,
    *,
    samples: Optional[List[int]] = None,
    random_select: bool = False,
    seed: Optional[int] = None,
    time_limit: Optional[int] = 10, # Default time limit for the *whole* function call
    workers: Optional[int] = None, # Allow user to override, None means auto-detect
    progress_callback=None, # 添加进度回调函数
    beam_width: int = 1, # Add beam_width parameter with default
) -> Dict[str, Any]:
    """
    返回 JSON‑serialisable 结果字典。
    如果提供了progress_callback，将会定期调用它来报告进度。
    progress_callback函数签名: progress_callback(percent: int, message: str)
    """
    # 开始计时
    start_time = time.perf_counter() # Start timer for the whole function

    # 报告初始进度
    report_progress(0, "验证参数...", start_time, progress_callback)

    # 参数检查
    if not (45 <= m <= 54 and 7 <= n <= 25 and 4 <= k <= 7 and 3 <= s <= 7 and s <= j <= k):
        raise ValueError("参数范围错误，详见题目要求")
    if not (1 <= t <= j): # Add t validation
        raise ValueError(f"t ({t}) 必须满足 1 <= t <= j ({j})")
    if random_select:
        rng = random.Random(seed)
        samples = rng.sample(range(1, m + 1), n)
    if samples is None:
        raise ValueError("必须提供 samples 或使用 random_select")
    if len(samples) != n:
        raise ValueError("samples 长度与 n 不符")
    samples = sorted(samples)

    # Determine number of workers for CP-SAT
    if workers is None or workers <= 0:
        if psutil:
            # Use physical cores * 1.5 as a heuristic, min 1
            auto_workers = max(1, int(psutil.cpu_count(logical=False) * 1.5))
            print(f"Auto-detected workers: {auto_workers} (physical cores * 1.5)", file=sys.stderr)
        else:
            auto_workers = 4 # Fallback if psutil not available
            print(f"Warning: psutil not found. Defaulting workers to {auto_workers}.", file=sys.stderr)
        effective_workers = auto_workers
    else:
        effective_workers = workers
        print(f"Using user-specified workers: {effective_workers}", file=sys.stderr)

    # 生成组合
    report_progress(5, "生成组合...", start_time, progress_callback)

    k_combos = list(itertools.combinations(samples, k))
    j_subsets = list(itertools.combinations(samples, j))

    report_progress(10, f"生成了 {len(k_combos)} 个k-组合和 {len(j_subsets)} 个j-子集", start_time, progress_callback)

    # Initialize final result variables before branching
    combos_selected = []
    final_accuracy = 0.0
    final_objective = 0.0
    final_bound = 0.0
    greedy_indices_output = [] # For s < j case specifically

    # 根据s和j的关系选择不同的算法
    # 仅当 k = j = s 时才走 CP-SAT 特化路径
    if s == j and k == j:
        # 当 k=j=s 时，进行变量裁剪并使用CP-SAT求解器
        if unique_k_combos:
            report_progress(11, f"s=j: Pruning k-combinations using s={s} signature...", start_time, progress_callback)
            original_k_count = len(k_combos)
            k_combos = unique_k_combos(samples, k, s) # Prune k_combos based on s-subset signature
            report_progress(12, f"Pruned k-combinations from {original_k_count} to {len(k_combos)}", start_time, progress_callback)
        else:
            report_progress(11, "s=j: Skipping k-combination pruning (utility not loaded).", start_time, progress_callback)

        # ★ Run Greedy first to get warm start indices, even when s==j
        # Define time budgets
        overall_time_budget = time_limit or 30 # Total time for the function
        warmup_greedy_time = 3 # Time for greedy + 2-Opt phase
        cp_sat_time_budget = overall_time_budget - warmup_greedy_time # Remaining time for CP-SAT
        if cp_sat_time_budget <= 0:
             print("Warning: Not enough time budget allocated for CP-SAT phase. Setting to minimum 1s.", file=sys.stderr)
             cp_sat_time_budget = 1
        
        # Skip greedy warm-up for s==j, use zero hints
        print("s == j: Skipping greedy warmup, using zero hints for CP-SAT", file=sys.stderr)
        report_progress(13, "s=j: 跳过预热贪心，使用零提示", start_time, progress_callback) # Update progress message
        warm_start_hints_greedy = [0] * len(k_combos)

        all_k_combos = k_combos[:]          # 备份全集
        k_combos_map = {combo: i for i, combo in enumerate(all_k_combos)} # Map combo to original index for warm2 mapping

        # ---------- ① 首轮 CP-SAT ----------
        MAX_INIT_COLS = 100_000
        MAX_SUBSETS = 50_000 # Keep subset sampling limit
        TIME_ROUND_1 = 22 # Seconds
        rng = random.Random(seed if seed is not None else 42)

        # 1. Sample k_combos first
        k_combos_round1 = k_combos # Start with potentially pruned list
        if len(k_combos_round1) > MAX_INIT_COLS:
            print(f"Sampling {MAX_INIT_COLS} from {len(k_combos_round1)} k-combinations for Round 1...", file=sys.stderr)
            indices_round1 = rng.sample(range(len(k_combos_round1)), MAX_INIT_COLS)
            k_combos_round1 = [k_combos[i] for i in indices_round1] # Use original k_combos for indexing if pruned before
            # Adjust warm start hints to match the sampled subset
            # If using zero hints, this adjustment isn't needed
            # warm1 = [warm_start_hints_greedy[i] for i in indices_round1] # If warm start was used
            warm1 = [0] * len(k_combos_round1) # If using zero hints
            print(f"Sampled {len(k_combos_round1)} combos for Round 1.", file=sys.stderr)
        else:
            # warm1 = warm_start_hints_greedy # Use original hints if no sampling and warm start used
            warm1 = [0] * len(k_combos_round1) # If using zero hints
            print(f"Using all {len(k_combos_round1)} k-combinations for Round 1.", file=sys.stderr)

        # 2. Get j_subsets for Round 1 (No filtering needed based on k_combos when k=j=s,
        #    because _threshold_set_cover handles the kc == js logic internally for this case)
        #    Start with all original j_subsets.
        j_subsets_r1 = j_subsets
        original_j_count = len(j_subsets_r1)
        print(f"Starting with {original_j_count} total j-subsets for Round 1 (k=j=s case).", file=sys.stderr)

        # 3. Sample the j_subsets if needed
        if original_j_count > MAX_SUBSETS:
            print(f"Sampling j_subsets down to {MAX_SUBSETS}...", file=sys.stderr)
            rng_js = random.Random(seed if seed is not None else 42) # Use separate RNG? Fine for now.
            j_subsets_r1 = rng_js.sample(j_subsets_potential, MAX_SUBSETS)
            print(f"Sampled j_subsets for Round 1: kept {len(j_subsets_r1)}", file=sys.stderr)
        else:
             print(f"Using all {len(j_subsets_r1)} potentially coverable j_subsets for Round 1.", file=sys.stderr)

        # Ensure we don't proceed with an empty list if filtering/sampling removed everything
        # 抽样或过滤导致 j_subsets_r1 为空就 raise
        if not j_subsets_r1:
             # raise RuntimeError("无法求解：过滤/抽样后没有剩余的 j-子集可供覆盖")
             raise RuntimeError(
                 "过滤后 j_subsets 为空 —— 很可能是覆盖判定写错，"
                 "或 k 与 j 的大小关系和特化假设不符。"
             )

        # 4. Pass consistent k_combos and j_subsets to the solver
        print(f"s == j: Starting CP-SAT Round 1 (time_limit={TIME_ROUND_1}s, cols={len(k_combos_round1)}, j_subsets={len(j_subsets_r1)})...", file=sys.stderr)
        report_progress(25, f"运行CP-SAT 第1轮 (最多 {TIME_ROUND_1}s, j_subsets={len(j_subsets_r1)})...", start_time, progress_callback)

        try:
            # Pass k_combos_round1 and j_subsets_r1
            sel1, obj1, bound1 = _threshold_set_cover(
                    combos=k_combos_round1,
                    j_subsets=j_subsets_r1,   # <--- Use correctly sampled j_subsets
                    t=t,
                    workers=effective_workers,
                    time_limit=TIME_ROUND_1,
                    progress_callback=None, # Suppress nested progress
                    start_time=start_time,
                    warm_start_hints=warm1 # Use potentially adjusted hints
            )
            # ----> 添加详细打印 <----
            print(f"DEBUG: _threshold_set_cover returned:", file=sys.stderr)
            print(f"DEBUG: sel1 (first 10): {sel1[:10]}", file=sys.stderr) # 打印前10个看看
            print(f"DEBUG: len(sel1): {len(sel1)}", file=sys.stderr)
            print(f"DEBUG: obj1: {obj1}", file=sys.stderr)
            print(f"DEBUG: bound1: {bound1}", file=sys.stderr)
            # ----> 结束打印 <----

            accuracy1 = bound1 / (obj1 + 1e-9) if obj1 > 1e-9 else (1.0 if bound1 > 1e-9 else 0.0) # Avoid division by zero
            print(f"Round 1 Finished. Combos: {len(sel1)}, Obj: {obj1:.1f}, Bound: {bound1:.1f}, Accuracy: {accuracy1:.3f}", file=sys.stderr)
            report_progress(75, f"第1轮完成. 准确率: {accuracy1:.3f}", start_time, progress_callback)
            round1_successful = True
        except Exception as e: # Catch other potential errors during solve
            print(f"Error during CP-SAT Round 1 solve: {e}", file=sys.stderr)
            report_progress(75, "第1轮求解出错", start_time, progress_callback)
            sel1 = []
            obj1 = 0.0
            bound1 = 0.0
            accuracy1 = 0.0
            round1_successful = False
            # Decide whether to proceed to round 2 or exit? Let's stop here.
            # Or maybe allow Round 2 attempt? For now, we'll stop and return R1 results.


        # ---------- ② 二轮 CP-SAT（如果需要） ----------
        TIME_ROUND_2 = 10 # Seconds
        EXTRA_COLS = 50_000
        TARGET_ACCURACY = 0.80

        combos_selected = sel1 # Default to round 1 result
        final_accuracy = accuracy1
        final_objective = obj1
        final_bound = bound1

        # Only proceed to round 2 if round 1 was successful and accuracy is low
        if round1_successful and accuracy1 < TARGET_ACCURACY:
            elapsed_r1 = time.perf_counter() - start_time
            remaining_time = (time_limit or (TIME_ROUND_1 + TIME_ROUND_2 + warmup_greedy_time + 5)) - elapsed_r1 # Calculate remaining time for R2 + buffer
            actual_time_round2 = min(TIME_ROUND_2, max(1, remaining_time - 2)) # Ensure at least 1s, leave 2s buffer

            if actual_time_round2 < 1:
                print("Warning: Not enough time remaining for Round 2. Skipping.", file=sys.stderr)
            else:
                print(f"Accuracy ({accuracy1:.3f}) < {TARGET_ACCURACY}. Starting CP-SAT Round 2 (time_limit={actual_time_round2:.1f}s)...", file=sys.stderr)
                report_progress(80, f"准确率不足，运行CP-SAT 第2轮 (最多 {actual_time_round2:.1f}s)...", start_time, progress_callback)

                # Identify remaining combos from the *original* full set
                k_combos_round1_set = set(k_combos_round1)
                remaining_combos = [c for c in all_k_combos if c not in k_combos_round1_set]

                # Sample extra combos
                num_extra_to_sample = min(EXTRA_COLS, len(remaining_combos))
                if num_extra_to_sample > 0:
                    k_extra = rng.sample(remaining_combos, num_extra_to_sample)
                    print(f"Sampling {len(k_extra)} extra combos for Round 2.", file=sys.stderr)
                else:
                    k_extra = []
                    print("No remaining combos to sample for Round 2.", file=sys.stderr)

                k_combos_round2 = k_combos_round1 + k_extra
                print(f"Round 2 total combos: {len(k_combos_round2)}")

                # —— 同样过滤并抽样 j_subsets for Round 2 ——
                print(f"Filtering/Sampling j_subsets for Round 2 based on {len(k_combos_round2)} k_combos...", file=sys.stderr)
                k_set2 = set(k_combos_round2)
                j_subsets_potential_r2 = [js for js in j_subsets if js in k_set2] # Filter from original j_subsets
                original_coverable_count_r2 = len(j_subsets_potential_r2)
                print(f"Found {original_coverable_count_r2} j_subsets potentially coverable by Round 2 k_combos.", file=sys.stderr)

                j_subsets_r2 = j_subsets_potential_r2
                if original_coverable_count_r2 > MAX_SUBSETS:
                    print(f"Sampling j_subsets for Round 2 down to {MAX_SUBSETS}...", file=sys.stderr)
                    rng_js2 = random.Random(seed if seed is not None else 42)
                    j_subsets_r2 = rng_js2.sample(j_subsets_potential_r2, MAX_SUBSETS)
                    print(f"Sampled j_subsets for Round 2: kept {len(j_subsets_r2)}", file=sys.stderr)
                else:
                    print(f"Using all {len(j_subsets_r2)} potentially coverable j_subsets for Round 2.", file=sys.stderr)

                if not j_subsets_r2:
                    print("Error: No j_subsets remain after filtering/sampling for Round 2. Skipping Round 2.", file=sys.stderr)
                    # Skip the rest of Round 2 logic
                else:
                   # ... proceed with warm start hints and call _threshold_set_cover ...
                   sel1_set = set(sel1)
                   warm2 = [1 if c in sel1_set else 0 for c in k_combos_round2]
                   print(f"Round 2 warm hints: {sum(warm2)} non-zero (based on Round 1 solution).", file=sys.stderr)
                   print(f"Starting CP-SAT Round 2 (time_limit={actual_time_round2:.1f}s, cols={len(k_combos_round2)}, j_subsets={len(j_subsets_r2)})...", file=sys.stderr)

                   # Call _threshold_set_cover for Round 2
                   # Add try...except block similar to Round 1 call
                   try:
                       sel2, obj2, bound2 = _threshold_set_cover(
                               combos=k_combos_round2, # Use combined list
                               j_subsets=j_subsets_r2,   # ← 传入过滤后的子集
                               t=t,
                               workers=effective_workers,
                               time_limit=actual_time_round2,
                               progress_callback=None,
                               start_time=start_time,
                               warm_start_hints=warm2
                       )
                       accuracy2 = bound2 / (obj2 + 1e-9) if obj2 > 1e-9 else (1.0 if bound2 > 1e-9 else 0.0)
                       print(f"Round 2 Finished. Combos: {len(sel2)}, Obj: {obj2:.1f}, Bound: {bound2:.1f}, Accuracy: {accuracy2:.3f}", file=sys.stderr)
                       report_progress(95, f"第2轮完成. 最终准确率: {accuracy2:.3f}", start_time, progress_callback)

                       # Update final results if Round 2 ran successfully
                       combos_selected = sel2
                       final_accuracy = accuracy2
                       final_objective = obj2
                       final_bound = bound2
                   except Exception as e:
                        print(f"Error during CP-SAT Round 2 solve: {e}", file=sys.stderr)
                        report_progress(95, "第2轮求解出错", start_time, progress_callback)
                        # Keep Round 1 results if Round 2 fails
    else: # s < j case (Greedy is the main algorithm)
        report_progress(15, "s < j: 使用贪心算法...", start_time, progress_callback)
        # Ensure k_combos used here is the original unfiltered list
        # (unique_k_combos pruning only happens if s==j)
        original_k_combos = list(itertools.combinations(samples, k)) # Regenerate if needed? Or pass original. Let's assume k_combos is the original if s!=j path taken
        # Call greedy algorithm
        combos_selected, greedy_indices = _greedy_cover_partial( # <-- Capture greedy_indices here
            samples=samples,
            k_combos=k_combos, # Use the k_combos available in this scope (should be original if s<j)
            j=j,
            s=s,
            start_time=start_time,
            progress_callback=progress_callback,
            # use_bitmask=True, # Default is True in function def
            beam_width=beam_width
        )
        # Assign results for the s < j case
        final_accuracy = 0.0 # Greedy doesn't provide bounds/accuracy currently
        final_objective = len(combos_selected)
        final_bound = 0.0 # Greedy doesn't provide bounds/accuracy currently
        # Store greedy_indices specifically for s<j case output
        greedy_indices_output = greedy_indices # Assign result from _greedy_cover_partial

    # Correct indentation for the block after if/else
    end_time = time.perf_counter() # End timer
    execution_time = end_time - start_time

    # Removed duplicate report_progress call
    report_progress(100, "计算完成", start_time, progress_callback)

    # --- 计算理论界限 ---
    report_progress(100, "计算理论界限...", start_time, progress_callback) # Keep at 100%
    theoretical_bounds = calculate_theoretical_bounds(n, k, j, s, t)
    theoretical_lower = theoretical_bounds.get("lower_bound")
    theoretical_upper = theoretical_bounds.get("upper_bound")
    theoretical_notes = theoretical_bounds.get("notes", "")
    if theoretical_notes:
        print(f"Theoretical Bounds Notes: {theoretical_notes}", file=sys.stderr)

    # Prepare the final result dictionary
    res = {
        "m": m,
        "n": n,
        "k": k,
        "j": j,
        "s": s,
        "t": t,
        "samples": samples,
        "combos": combos_selected, # Already updated if R2 ran
        "execution_time": round(execution_time, 3),
        "workers": effective_workers,
        "greedy_indices": greedy_indices_output, # Use the dedicated output variable
        "accuracy": round(final_accuracy, 4), # ★ Add final accuracy
        "objective_value": round(final_objective, 1), # Add final objective
        "best_bound": round(final_bound, 1), # Add final bound
        "theoretical_lower_bound": theoretical_lower, # Add theoretical lower bound
        "theoretical_upper_bound": round(theoretical_upper, 2) if theoretical_upper is not None else None, # Add theoretical upper bound (rounded)
        "theoretical_notes": theoretical_notes # Add notes from bounds calculation
    }

    # Print the final result as a single JSON line to stdout
    print(json.dumps(res, ensure_ascii=False, separators=(',', ':'))) # Compact JSON

    return res # Return the dictionary as before for potential direct calls

# Database saving is now handled by the Electron main process.
# Removed _DB, _init_db, save_result functions.

################
#  FastAPI APP #
################

app = FastAPI(title="Optimal Samples Selection System")

# Middleware for timing requests
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start_api = time.perf_counter()
    response = await call_next(request)
    elapsed_api = time.perf_counter() - start_api
    response.headers["X-Process-Time"] = f"{elapsed_api:.3f}s"
    logging.info(f"API Request {request.method} {request.url.path} completed in {elapsed_api:.3f}s") # Optional logging
    return response

class RequestModel(BaseModel):
    m: int
    n: int
    k: int
    j: int
    s: int
    t: int = 1
    samples: Optional[List[int]] = None
    random_select: bool = False
    seed: Optional[int] = None
    time_limit: Optional[int] = 10
    workers: Optional[int] = 8 # Add optional workers, defaulting to 8

@app.post("/select")
async def api_select(req: RequestModel):
    # Extract workers, providing default if not present or None
    request_params = req.dict()
    workers_to_use = request_params.pop('workers', 8) # Remove workers from dict, use default 8 if missing
    if workers_to_use is None: # Handle explicit null if pydantic allows
        workers_to_use = 8

    try:
        # Pass remaining params and the processed workers value
        result = select_optimal_samples(**request_params, workers=workers_to_use)

        # Saving is now handled by the main process
        # rid = save_result(result) # REMOVED
        # result["id"] = rid # REMOVED

        return result # Return result without DB id
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))

# Reading and deleting results are now handled by the main process
# Removed /results/{rid} GET and DELETE endpoints
# @app.get("/results/{rid}")
# async def api_get_result(rid: int):
#     _init_db()
#     with sqlite3.connect(_DB) as conn:
        cur = conn.execute("SELECT params, combos FROM results WHERE id=?", (rid,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "记录不存在")
        params = json.loads(row[0])
        combos = json.loads(row[1])
        return {**params, "combos": combos, "id": rid}

# @app.delete("/results/{rid}")
# async def api_delete_result(rid: int):
#     _init_db()
#     with sqlite3.connect(_DB) as conn:
#         cur = conn.execute("DELETE FROM results WHERE id=?", (rid,))
#         if cur.rowcount == 0:
#             raise HTTPException(404, "记录不存在")
#         return {"deleted": rid}

########################
#  CLI for quick tests #
#  (Note: CLI will no longer save results to DB)
########################

def main():
    p = argparse.ArgumentParser(description="Optimal Samples Selection CLI")
    p.add_argument("-m", type=int, required=True, help='总样本数量 (45 <= m <= 54)')
    p.add_argument("-n", type=int, required=True, help='选择的样本数量 (7 <= n <= 25)')
    p.add_argument("-k", type=int, required=True, help='组合大小 (4 <= k <= 7)')
    p.add_argument("-j", type=int, required=True, help='子集大小 (s <= j <= k)')
    p.add_argument("-s", type=int, required=True, help='内部子集大小 (3 <= s <= 7)')
    p.add_argument("-t", type=int, default=1, help='覆盖阈值 (默认: 1)')
    p.add_argument("--samples", type=str, help='逗号分隔的样本列表 (例如: "1,2,3,4,5,6,7")')
    p.add_argument("--random", action="store_true", help="随机选样")
    p.add_argument("--seed", type=int, help="随机种子")
    p.add_argument("--time", type=int, default=30, help="求解时间限制(秒, 默认 30)") # Default CLI time limit to 30
    p.add_argument("--workers", type=int, help="求解器使用的 CPU 工作线程数 (默认: 自动检测)") # Changed help text
    p.add_argument("--beam", type=int, default=1, help="贪心算法 s<j 情况下的 Beam Width (默认: 1)") # Add beam argument
    args = p.parse_args()

    try:
        # 解析样本列表
        samples_list = None
        if args.samples:
            samples_list = [int(x) for x in args.samples.split(",") if x.strip()]

        # 调用核心函数并计时
        start_cli = time.perf_counter() # Start timer
        res = select_optimal_samples(
            args.m,
            args.n,
            args.k,
            args.j,
            args.s,
            args.t,
            samples=samples_list,
            random_select=args.random,
            seed=args.seed,
            time_limit=args.time,
            workers=args.workers, # Pass workers from args
            beam_width=args.beam, # Pass beam width from args
        )
        # execution_time is now part of the result 'res'

        # 输出JSON结果 (contains execution_time)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        # Print solver time (if any) and overall success message to stderr
        # The main execution time is already in the JSON output.
        # We can still print the original CLI timer for comparison/verification if needed
        # elapsed_cli = time.perf_counter() - start_cli
        # print(f'CLI Measured Runtime: {elapsed_cli:.3f} s', file=sys.stderr)
        print("算法执行成功。", file=sys.stderr) # Keep success message on stderr

    except ValueError as ve:
        print(f"输入验证错误: {ve}", file=sys.stderr)
        sys.exit(1)  # 验证错误退出码
    except RuntimeError as re:
        print(f"运行时错误: {re}", file=sys.stderr)
        sys.exit(2)  # 运行时错误退出码
    except Exception as e:
        print(f"发生意外错误: {e}", file=sys.stderr)
        sys.exit(3)  # 通用错误退出码

if __name__ == "__main__":
    main()
