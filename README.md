最优样本选择系统 用户手册

目录

1. 简介

2. 架构与技术栈

2.1 软件架构

2.2 技术栈

2.3 关键文件与目录

3. 安装与环境要求

3.1 环境要求

3.2 安装依赖

4. 运行 & 使用

4.1 开发模式

4.2 使用指南

4.2.1 参数输入页面

4.2.2 管理结果页

4.2.3 查看结果页

5. 发布 & 打包

6. 算法原理

6.1 问题定义

6.2 算法流程

7. 故障排除

8. 性能 & 基准测试

附录: 术语表

1. 简介

本系统旨在解决如下组合优化问题：

在 M 个样本中选取 N 个初始样本。

找到最小数量的、每个包含 K 个样本的 k-组合集合。

满足对所有 J 元子集的 s-层级覆盖：每个含 J 个样本的子集，须至少被 T 个 k-组合覆盖。

用户可通过 GUI 输入参数 (M, N, K, J, S, T) 及初始样本列表，一键生成最优或近似最优的组合并保存结果。

2. 架构与技术栈

2.1 软件架构

主进程 (Main)：

入口：src/main/index.ts → dist/main/main/index.js

负责创建窗口、管理生命周期、IPC 调度、数据库文件操作。

IPC handlers 位于 src/main/ipcHandlers/。

渲染进程 (Renderer)：

技术：React + TypeScript + Ant Design

入口：src/renderer/main.tsx，HTML：src/renderer/index.html

构建：Vite → dist/renderer

预加载脚本 (Preload)：

文件：src/preload.ts → dist/main/preload.js

使用 contextBridge 暴露安全 IPC 接口。

核心算法 (Python Backend)：

脚本：src/python/algorithm.py

依赖：ortools, numpy, psutil

通过 python-shell 从 Node 主进程调用。

服务层 (Services)：

src/services/validator.ts：参数校验。

db.ts（可选）：辅助数据库操作。

2.2 技术栈

框架：Electron, React, Vite

语言：TypeScript, JavaScript, Python

UI：Ant Design

数据库：SQLite (better-sqlite3)

导出：Excel (exceljs)

打包：electron-builder

IPC：Electron IPC, python-shell

2.3 关键文件与目录
![image](https://github.com/user-attachments/assets/8b0d948c-26ff-429d-917d-5d2a5212a972)

3. 安装与环境要求

3.1 环境要求

Node.js (LTS) + npm

Python ≥3.7（勾选 “Add Python to PATH”）

pip

3.2 安装依赖
![image](https://github.com/user-attachments/assets/f7df5d06-9224-4bb9-acce-c50da82ab8ab)

4. 运行 & 使用

4.1 开发模式
![image](https://github.com/user-attachments/assets/c76549aa-75df-4291-a886-baf4e8bd54e1)
启动本地 Vite 服务并打开 Electron 窗口。

4.2 使用指南

4.2.1 参数输入页面

M: 总样本数 (45–54)

N: 初始样本数 (7–25)

K: 组合大小 (4–7)

J/S/T: 覆盖参数

已选样本: 手动输入或随机生成

点击 生成最优组合，进度条 & 日志实时更新。

4.2.2 管理结果页

列出 database/ 中所有 .db 文件

刷新、查看 (👁️)、导出 Excel (📄)、删除 (🗑️)

4.2.3 查看结果页

展示运行参数、初始样本、最终 k-组合。

5. 发布 & 打包
![image](https://github.com/user-attachments/assets/40f61960-47c5-4f72-b581-53a5ff2b88f3)
在 release/ 获取安装包。

6. 算法原理

6.1 问题定义

在参数 (M,N,K,J,S,T) 和样本列表下，从所有可能的 K-组合中选取最少数量，
使得每个 J-子集在 s-层级被 ≥T 个组合覆盖。

6.2 算法流程

s == j (精确求解)：

剪枝 (可选)

构建 CP-SAT 模型：布尔变量 + 目标最小化 + 覆盖约束

对称性破除 & 并行 & 时间限制

s < j (贪心启发式)：

迭代选取覆盖增益最大的组合

位掩码加速 + Beam Search（可选）

2-Opt 局部搜索优化

7. 故障排除

空白/卡住: 检查 Vite & 打包路径

Python 错误: 验证 Python 环境 & 依赖

结果文件未列出: 确认 database/ 目录与正则匹配

Excel 导出失败: 检查权限 & exceljs 安装

8. 性能 & 基准测试

8.1 性能提示

s=j 情况下 CP-SAT 多核 & 剪枝加速

s<j 情况下 位掩码 & 2-Opt 提升效果

8.2 基准测试

脚本：bench/bench.py

参数示例：t=1, workers=4, time_limit=60s

指标：执行时间、组合数量

数据：benchmark_results.csv6. 算法原理

6.1 问题定义

在参数 (M,N,K,J,S,T) 和样本列表下，从所有可能的 K-组合中选取最少数量，
使得每个 J-子集在 s-层级被 ≥T 个组合覆盖。

6.2 算法流程

s == j (精确求解)：

剪枝 (可选)

构建 CP-SAT 模型：布尔变量 + 目标最小化 + 覆盖约束

对称性破除 & 并行 & 时间限制

s < j (贪心启发式)：

迭代选取覆盖增益最大的组合

位掩码加速 + Beam Search（可选）

2-Opt 局部搜索优化

7. 故障排除

空白/卡住: 检查 Vite & 打包路径

Python 错误: 验证 Python 环境 & 依赖

结果文件未列出: 确认 database/ 目录与正则匹配

Excel 导出失败: 检查权限 & exceljs 安装

8. 性能 & 基准测试

8.1 性能提示

s=j 情况下 CP-SAT 多核 & 剪枝加速

s<j 情况下 位掩码 & 2-Opt 提升效果

8.2 基准测试

脚本：bench/bench.py

参数示例：t=1, workers=4, time_limit=60s

指标：执行时间、组合数量

数据：benchmark_results.csv

附录: 术语表

缩写

含义

CP-SAT

Google OR-Tools 约束编程求解器

K-组合

每个组合的样本数量

J-子集

用于覆盖检测的样本子集

S-子集

s-层级覆盖的最小子集

2-Opt

局部搜索优化算法
