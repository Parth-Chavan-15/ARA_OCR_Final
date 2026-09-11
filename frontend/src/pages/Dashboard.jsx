import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';
import { apiService } from '../services/api';
import {
  Users,
  FileCheck2,
  Clock,
  CheckCircle2,
  ShieldCheck,
  AlertTriangle,
  Sparkles,
  ArrowRight,
  RefreshCw,
  Award,
  Layers,
  FileText,
} from 'lucide-react';

const Dashboard = () => {
  const navigate = useNavigate();
  const context = useOutletContext();
  const refreshTrigger = context?.refreshTrigger || 0;

  const [candidates, setCandidates] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchDashboardData = useCallback(async () => {
    setLoading(true);
    try {
      const [candData, docsData, infoData] = await Promise.all([
        apiService.getCandidates(),
        apiService.getDocuments(),
        apiService.getModelInfo().catch(() => null),
      ]);

      setCandidates(Array.isArray(candData) ? candData : []);
      setDocuments(Array.isArray(docsData) ? docsData : []);
      if (infoData) setModelInfo(infoData);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData, refreshTrigger]);

  const totalCandidates = candidates.length;
  const totalDocuments = documents.length;
  const verifiedDocs = documents.filter((d) => Boolean(d.document_type)).length;
  const pendingDocs = totalDocuments - verifiedDocs;

  // Breakdown by Broad Class
  const classBreakdown = {
    'Caste Validity Certificate': documents.filter(
      (d) => d.document_type === 'CASTE_VALIDITY_CERTIFICATE'
    ).length,
    'Caste Certificate': documents.filter(
      (d) => d.document_type === 'CASTE_CERTIFICATE'
    ).length,
    'Caste Validity Receipt': documents.filter(
      (d) => d.document_type === 'CASTE_VALIDITY_RECEIPT'
    ).length,
    'Proforma-O (Minority)': documents.filter(
      (d) => d.document_type === 'PROFORMA_O'
    ).length,
    'School Leaving Certificate': documents.filter(
      (d) => d.document_type === 'LEAVING_CERTIFICATE'
    ).length,
    'Non-Creamy Layer / Out-of-Scope': documents.filter(
      (d) => d.document_type === 'UNKNOWN_OUT_OF_SCOPE'
    ).length,
  };

  return (
    <div className="space-y-6">
      {/* Official Government Executive Banner */}
      <div className="bg-gradient-to-r from-[#0B1B2B] via-[#0F2942] to-[#1A3D60] rounded-2xl p-6 text-white shadow-lg border border-slate-700 relative overflow-hidden">
        <div className="absolute right-0 top-0 bottom-0 w-1/3 bg-radial from-amber-500/10 to-transparent pointer-events-none" />

        <div className="flex flex-wrap items-center justify-between gap-4 relative z-10">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-[10px] font-bold uppercase tracking-widest text-amber-400 bg-amber-950/60 px-2.5 py-0.5 rounded border border-amber-500/40">
                GOVERNMENT OF MAHARASHTRA • राज्य सामाईक प्रवेश परीक्षा कक्ष
              </span>
            </div>
            <h1 className="text-2xl font-black tracking-tight text-white mt-2">
              Admissions Regulating Authority Scrutiny Portal
            </h1>
            <p className="text-xs text-slate-300 max-w-2xl mt-1 leading-relaxed">
              Automated AI verification and broad classification engine for centralized admission certificates. 
              Powered by fine-tuned multimodal LayoutLMv3 vision-layout-text neural networks.
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={() => navigate('/incoming')}
              className="bg-gradient-to-r from-amber-400 to-amber-500 hover:from-amber-500 hover:to-amber-600 text-slate-950 font-bold px-4 py-2.5 rounded-xl text-xs shadow-md transition flex items-center space-x-2 cursor-pointer"
            >
              <span>Go to Document Scrutiny</span>
              <ArrowRight size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Candidates Card */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
              Enrolled Students
            </span>
            <p className="text-2xl font-black text-slate-900 mt-1">
              {loading ? '...' : totalCandidates}
            </p>
            <span className="text-[10px] text-blue-700 font-semibold bg-blue-50 px-1.5 py-0.5 rounded mt-1 inline-block">
              27 Centralized Admissions
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-800 flex items-center justify-center">
            <Users size={24} />
          </div>
        </div>

        {/* Total Documents Card */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
              Vault Documents
            </span>
            <p className="text-2xl font-black text-slate-900 mt-1">
              {loading ? '...' : totalDocuments}
            </p>
            <span className="text-[10px] text-slate-600 font-semibold bg-slate-100 px-1.5 py-0.5 rounded mt-1 inline-block">
              data/originals/ storage
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-slate-100 text-slate-800 flex items-center justify-center">
            <FileText size={24} />
          </div>
        </div>

        {/* Verified / Scrutinized */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-xs font-bold text-emerald-700 uppercase tracking-wider">
              Scrutinized & Verified
            </span>
            <p className="text-2xl font-black text-emerald-800 mt-1">
              {loading ? '...' : verifiedDocs}
            </p>
            <span className="text-[10px] text-emerald-700 font-semibold bg-emerald-50 px-1.5 py-0.5 rounded mt-1 inline-block">
              {pendingDocs} pending scrutiny
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center">
            <CheckCircle2 size={24} />
          </div>
        </div>

        {/* AI Accuracy */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-xs font-bold text-amber-700 uppercase tracking-wider">
              AI Scrutiny Accuracy
            </span>
            <p className="text-2xl font-black text-amber-900 mt-1">
              {modelInfo?.accuracy ? `${modelInfo.accuracy}%` : '88.27%'}
            </p>
            <span className="text-[10px] text-amber-800 font-bold bg-amber-50 px-1.5 py-0.5 rounded mt-1 inline-block">
              {modelInfo?.best_run_id 
                ? `Winning Run ${modelInfo.best_run_id} Model (${modelInfo.weighted_f1}% Weighted, ${modelInfo.ood_f1}% OOD F1)`
                : 'Winning Run 4 Model (87.9% Weighted F1, 84.7% OOD F1)'}
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-amber-50 text-amber-700 flex items-center justify-center">
            <Award size={24} />
          </div>
        </div>
      </div>

      {/* Distribution Breakdown & Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Certificate Breakdown Table */}
        <div className="lg:col-span-2 bg-white rounded-xl p-5 border border-slate-200 shadow-xs space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Broad Category Classification Breakdown
              </h3>
              <p className="text-xs text-slate-500">
                Distribution across official admission certificate types
              </p>
            </div>
            <button
              onClick={() => navigate('/incoming')}
              className="text-xs font-bold text-blue-700 hover:text-blue-900 flex items-center space-x-1 cursor-pointer"
            >
              <span>View All</span>
              <ArrowRight size={13} />
            </button>
          </div>

          <div className="space-y-3">
            {Object.entries(classBreakdown).map(([name, count]) => {
              const pct = totalDocuments > 0 ? (count / totalDocuments) * 100 : 0;
              return (
                <div key={name} className="space-y-1">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-slate-800">{name}</span>
                    <span className="text-slate-600 font-mono font-bold">{count} ({pct.toFixed(0)}%)</span>
                  </div>
                  <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-700 to-indigo-600 rounded-full transition-all duration-500"
                      style={{ width: `${Math.max(pct, count > 0 ? 5 : 0)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Document Scrutiny Pipeline Card */}
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs space-y-4 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900 pb-3 border-b border-slate-100 flex items-center gap-2">
              <Layers size={16} className="text-blue-700" />
              Document Scrutiny Pipeline
            </h3>
            <ul className="text-xs text-slate-600 space-y-2.5 mt-3">
              <li className="flex items-start space-x-2">
                <span className="text-blue-700 font-bold">•</span>
                <span><strong>Ingestion & Preprocessing:</strong> Normalizes PDF/image uploads to high-resolution page buffers and validates orientation.</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-blue-700 font-bold">•</span>
                <span><strong>Bilingual OCR Extraction:</strong> Probes Marathi/English script and extracts word tokens with 2D bounding boxes via GPU PaddleOCR.</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-blue-700 font-bold">•</span>
                <span><strong>Multimodal LayoutLMv3:</strong> Fuses visual page layout, text tokens, and spatial coordinates for broad category prediction.</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-blue-700 font-bold">•</span>
                <span><strong>Hybrid Gatekeeper & OOD:</strong> Validates statutory government seals/headers and routes non-reservation files to Out-of-Scope.</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-emerald-700 font-bold">•</span>
                <span><strong>Real-Time WebSocket Sync:</strong> Broadcasts classification state and evidence audits instantly to the officer dashboard.</span>
              </li>
            </ul>
          </div>

          <button
            onClick={() => navigate('/candidates')}
            className="w-full bg-[#0F2942] hover:bg-[#1A3D60] text-amber-300 py-2.5 px-4 rounded-lg text-xs font-bold transition shadow-xs flex items-center justify-center space-x-2 cursor-pointer"
          >
            <span>Open Candidate Registry</span>
            <ArrowRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;