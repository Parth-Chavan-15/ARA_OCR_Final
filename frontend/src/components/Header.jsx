import React, { useState } from 'react';
import { apiService } from '../services/api';
import { RefreshCw, Database, Cpu, CheckCircle2, RotateCcw, AlertTriangle } from 'lucide-react';

const Header = ({ onSyncComplete }) => {
  const [syncing, setSyncing] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [syncMsg, setSyncMsg] = useState(null);

  const handleSync = async () => {
    try {
      setSyncing(true);
      const res = await apiService.syncStorage();
      setSyncMsg(`Disk Storage Synced: ${res.candidates_added} new candidates, ${res.documents_added} new documents (${res.total_candidates} total candidates in vault)`);
      setTimeout(() => setSyncMsg(null), 5000);
      if (onSyncComplete) onSyncComplete();
    } catch (err) {
      console.error(err);
      setSyncMsg('Sync failed. Please check backend connection.');
      setTimeout(() => setSyncMsg(null), 5000);
    } finally {
      setSyncing(false);
    }
  };

  const handleReset = async () => {
    const confirmReset = window.confirm(
      'Are you sure you want to reset all document scrutiny? This will reset all document statuses to PENDING and clear previous OCR results so you can re-scan and classify all files fresh.'
    );
    if (!confirmReset) return;

    try {
      setResetting(true);
      const res = await apiService.resetScrutiny();
      setSyncMsg(`Reset Completed: All ${res.total_documents} documents reset to PENDING. Ready for fresh scrutiny.`);
      setTimeout(() => setSyncMsg(null), 6000);
      if (onSyncComplete) onSyncComplete();
    } catch (err) {
      console.error(err);
      setSyncMsg('Reset failed. Please check backend connection.');
      setTimeout(() => setSyncMsg(null), 5000);
    } finally {
      setResetting(false);
    }
  };

  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-30 shadow-sm">
      {/* Indian National Tricolor Ribbon */}
      <div className="h-1.5 w-full flex">
        <div className="h-full w-1/3 bg-[#FF9933]" />
        <div className="h-full w-1/3 bg-white border-y border-gray-100" />
        <div className="h-full w-1/3 bg-[#138808]" />
      </div>

      <div className="px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
        {/* Government Identity */}
        <div className="flex items-center space-x-3.5">
          <div className="w-11 h-11 rounded-lg bg-[#0F2942] text-amber-400 flex flex-col items-center justify-center font-serif font-black shadow border border-amber-500/40">
            <span className="text-xs leading-none font-bold text-amber-300">ARA</span>
            <span className="text-[8px] leading-none text-amber-400/80 mt-0.5">MAHA</span>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-200/80">
                GOVERNMENT OF MAHARASHTRA • महाराष्ट्र शासन
              </span>
              <span className="text-[10px] text-gray-500 font-medium hidden sm:inline">
                Academic Year 2026–2027
              </span>
            </div>
            <h1 className="text-base font-bold text-gray-900 tracking-tight flex items-center gap-2 mt-0.5">
              Admissions Regulating Authority (ARA)
              <span className="text-[10px] font-semibold text-blue-800 bg-blue-50 px-2 py-0.5 rounded-full border border-blue-200">
                State CET Cell
              </span>
            </h1>
          </div>
        </div>

        {/* System Indicators & Sync/Reset Actions */}
        <div className="flex items-center space-x-2.5">
          <div className="hidden lg:flex items-center space-x-2.5 bg-gray-50 px-3 py-1.5 rounded-lg border border-gray-200 text-xs text-gray-700">
            <div className="flex items-center space-x-1.5">
              <Database size={13} className="text-emerald-600" />
              <span className="font-medium">PostgreSQL Active</span>
            </div>
            <span className="text-gray-300">|</span>
            <div className="flex items-center space-x-1.5" title="Champion Run 3: 91.55% Test Accuracy, 90.37% Weighted F1, 91.72% OOD F1">
              <Cpu size={13} className="text-blue-600" />
              <span className="font-medium">LayoutLMv3 (91.6% Acc, 91.7% OOD)</span>
            </div>
          </div>

          {/* Sync Button */}
          <button
            onClick={handleSync}
            disabled={syncing || resetting}
            className="flex items-center space-x-1.5 bg-[#0F2942] hover:bg-[#1A3D60] text-amber-300 font-semibold px-3.5 py-2 rounded-lg text-xs shadow-sm transition border border-amber-500/30 active:scale-95 disabled:opacity-50 cursor-pointer"
          >
            <RefreshCw size={13} className={syncing ? "animate-spin text-amber-400" : "text-amber-400"} />
            <span>{syncing ? "Scanning Storage..." : "Sync Storage"}</span>
          </button>

          {/* Reset Scrutiny Button */}
          <button
            onClick={handleReset}
            disabled={syncing || resetting}
            className="flex items-center space-x-1.5 bg-slate-100 hover:bg-rose-50 text-slate-700 hover:text-rose-700 font-semibold px-3.5 py-2 rounded-lg text-xs transition border border-slate-300 hover:border-rose-300 active:scale-95 disabled:opacity-50 cursor-pointer"
            title="Reset all document classifications to PENDING and clear OCR cache"
          >
            <RotateCcw size={13} className={resetting ? "animate-spin text-rose-600" : "text-slate-500 hover:text-rose-600"} />
            <span>{resetting ? "Resetting..." : "Reset All Scrutiny"}</span>
          </button>
        </div>
      </div>

      {syncMsg && (
        <div className="bg-emerald-50 border-t border-b border-emerald-200 px-6 py-2 flex items-center justify-between text-xs text-emerald-800 font-medium">
          <div className="flex items-center space-x-2">
            <CheckCircle2 size={15} className="text-emerald-600" />
            <span>{syncMsg}</span>
          </div>
        </div>
      )}
    </header>
  );
};

export default Header;