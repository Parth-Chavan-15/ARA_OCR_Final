import React, { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';
import { createWebSocketClient } from '../services/api';

const Layout = () => {
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleSyncComplete = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  useEffect(() => {
    // Connect WebSocket for live real-time state sync across all pages
    const cleanupWs = createWebSocketClient((data) => {
      console.log('Live Scrutiny Event received via WebSocket:', data);
      setRefreshTrigger(prev => prev + 1);
    });

    return () => {
      if (cleanupWs) cleanupWs();
    };
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-[#F4F6F9]">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header onSyncComplete={handleSyncComplete} />
        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          <Outlet context={{ refreshTrigger }} />
        </main>
      </div>
    </div>
  );
};

export default Layout;
