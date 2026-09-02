import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';

const Layout = () => {
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleSyncComplete = () => {
    setRefreshTrigger(prev => prev + 1);
  };

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
