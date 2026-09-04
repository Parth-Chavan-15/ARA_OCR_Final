import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Users, FileCheck2, ShieldAlert, Sparkles, FolderCheck, BookOpen } from 'lucide-react';

const Sidebar = () => {
  const navItems = [
    {
      name: 'Scrutiny Dashboard',
      path: '/',
      icon: <LayoutDashboard size={18} />,
      badge: 'Live',
    },
    {
      name: 'Candidate Registry',
      path: '/candidates',
      icon: <Users size={18} />,
      badge: '27 Students',
    },
    {
      name: 'Document Scrutiny',
      path: '/incoming',
      icon: <FileCheck2 size={18} />,
      badge: 'Multimodal',
    },
    {
      name: 'Review & Audit Queue',
      path: '/review',
      icon: <ShieldAlert size={18} />,
    },
  ];

  return (
    <aside className="flex flex-col w-64 bg-[#0B1B2B] text-slate-200 min-h-screen border-r border-slate-800 shadow-2xl z-20 select-none">
      {/* Sidebar Header */}
      <div className="flex items-center space-x-3 px-5 py-5 border-b border-slate-800/80 bg-[#081420]">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-amber-400 to-amber-600 flex items-center justify-center text-slate-950 font-serif font-black text-sm shadow">
          ARA
        </div>
        <div>
          <span className="text-[11px] font-bold tracking-wider text-amber-400 uppercase block">
            GOVT. OF MAHARASHTRA
          </span>
          <span className="text-xs font-semibold text-slate-100 block">
            State CET Scrutiny
          </span>
        </div>
      </div>

      {/* Navigation Section */}
      <div className="flex-1 py-5 px-3 space-y-6 overflow-y-auto">
        <div>
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-3 mb-2 block">
            Scrutiny Modules
          </span>
          <nav className="space-y-1.5">
            {navItems.map((item) => (
              <NavLink
                key={item.name}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center justify-between px-3.5 py-2.5 rounded-lg text-xs font-medium transition-all duration-150 ${
                    isActive
                      ? 'bg-gradient-to-r from-blue-900/80 to-blue-800/60 text-white font-semibold shadow-inner border-l-4 border-amber-400'
                      : 'text-slate-300 hover:bg-slate-800/60 hover:text-white'
                  }`
                }
              >
                <div className="flex items-center space-x-3">
                  <span className="text-amber-400/90">{item.icon}</span>
                  <span>{item.name}</span>
                </div>
                {item.badge && (
                  <span className="text-[9px] font-bold uppercase tracking-wider bg-slate-800 text-amber-300/90 px-1.5 py-0.5 rounded border border-slate-700/60">
                    {item.badge}
                  </span>
                )}
              </NavLink>
            ))}
          </nav>
        </div>

        {/* Model Specs Card */}
        <div className="mx-1 p-3.5 bg-gradient-to-b from-slate-900/90 to-[#0F2338] rounded-xl border border-slate-800/80 text-[11px] space-y-2">
          <div className="flex items-center space-x-1.5 text-amber-400 font-semibold">
            <Sparkles size={14} />
            <span>AI Verification Model</span>
          </div>
          <div className="space-y-1.5 text-slate-300 text-[10px]">
            <div className="flex justify-between">
              <span className="text-slate-400">Architecture:</span>
              <span className="font-mono text-amber-300">LayoutLMv3</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Test Accuracy:</span>
              <span className="font-bold text-emerald-400">91.55%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">OOD F1 Score:</span>
              <span className="font-bold text-sky-400">91.72%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Trained Dataset:</span>
              <span className="font-medium text-slate-200">741 Docs (6 Classes)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">OCR Engine:</span>
              <span className="font-medium text-slate-200">PaddleOCR (GPU)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Sidebar Footer */}
      <div className="p-4 border-t border-slate-800/80 bg-[#081420] text-center">
        <p className="text-[10px] text-slate-400 font-medium">
          Admissions Regulating Authority
        </p>
        <p className="text-[9px] text-slate-500 mt-0.5">
          Maharashtra Act No. XXVIII of 2015
        </p>
      </div>
    </aside>
  );
};

export default Sidebar;