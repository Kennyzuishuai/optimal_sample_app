import React, { useState } from 'react';
import { HashRouter as Router, Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import { Layout, Menu, Typography, theme, ConfigProvider } from 'antd';
import { HomeOutlined, DatabaseOutlined, BarChartOutlined } from '@ant-design/icons';

// Import actual page components
import HomePage from './pages/Home';
import ResultPage from './pages/Result';
import DbManagerPage from './pages/DbManager';

const { Header, Content, Footer, Sider } = Layout;
const { Title } = Typography;

// Navigation component with Ant Design
const AppNavigation: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);

  const items = [
    {
      key: '/',
      icon: <HomeOutlined />,
      label: 'Home (Input Parameters)'
    },
    {
      key: '/db',
      icon: <DatabaseOutlined />,
      label: 'Manage Results'
    },
    {
      key: '/results',
      icon: <BarChartOutlined />,
      label: 'View Results'
    }
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={(value) => setCollapsed(value)}
        theme="light"
        style={{
          boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
        }}
      >
        <div style={{ height: 32, margin: 16, textAlign: 'center' }}>
          <Title level={5} style={{ margin: 0, color: '#1890ff' }}>Optimal Sample Selection</Title>
        </div>
        <Menu
          theme="light"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={items}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout>
        <Header style={{ padding: '0 16px', background: '#fff', boxShadow: '0 1px 4px rgba(0,0,0,0.1)' }}>
          <Title level={3} style={{ margin: '16px 0' }}>Optimal Sample Selection System</Title>
        </Header>
        <Content style={{ margin: '16px' }}>
          <div style={{ padding: 24, minHeight: 360, background: '#fff', borderRadius: 4 }}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              {/* Route for page showing details of a specific result file */}
              <Route path="/results/:filename" element={<ResultPage />} />
              {/* Route for the general results path (e.g., if navigated without filename) */}
              <Route path="/results" element={<ResultPage />} />
              <Route path="/db" element={<DbManagerPage />} />
              {/* Add other routes as needed */}
            </Routes>
          </div>
        </Content>
        <Footer style={{ textAlign: 'center', background: '#f0f2f5' }}>
        Optimal Sample Selection System ©{new Date().getFullYear()} All rights reserved
        </Footer>
      </Layout>
    </Layout>
  );
};

// Main application component
function App() {
  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1890ff',
          borderRadius: 4,
        },
      }}
    >
      <Router>
        <AppNavigation />
      </Router>
    </ConfigProvider>
  );
}

export default App;
