import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';
import { apiService } from '../services/api';
import StatusBadge from '../components/StatusBadge';
import {
  Upload,
  Play,
  FileText,
  CheckCircle2,
  Loader2,
  XCircle,
  RefreshCw,
  Search,
  Filter,
  Eye,
  ShieldCheck,
  AlertTriangle,
  FolderSync,
  Layers,
  Sparkles,
  RotateCcw,
} from 'lucide-react';

const PIPELINE_STAGES = [
  { key: 'fetch', label: '1. Ingestion & File Check' },
  { key: 'file_validation', label: '2. Format & Integrity Validation' },
  { key: 'pdf_rendering', label: '3. Multi-page Rendering' },
  { key: 'preprocessing', label: '4. Deskew & Image Normalization' },
  { key: 'ocr', label: '5. Multilingual PaddleOCR' },
  { key: 'ocr_cache', label: '6. OCR Token Spatial Cache' },
  { key: 'language_detection', label: '7. Script & Language Identification' },
  { key: 'layoutxlm_classification', label: '8. LayoutLMv3 Multimodal Classification' },
  { key: 'persist_result', label: '9. Evidence Match & Audit Ledger' },
];

const ProcessingModal = ({ documentId, onClose, onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isCompleted, setIsCompleted] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    // Step-by-step progress ticker through all 9 stages
    timer = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < PIPELINE_STAGES.length - 1) {
          return prev + 1;
        }
        return prev;
      });
    }, 280);

    const runClassification = async () => {
      try {
        const data = await apiService.classifyDocument(documentId);
        if (cancelled) return;

        clearInterval(timer);
        setCurrentStep(PIPELINE_STAGES.length);
        setIsCompleted(true);
        setResult(data);

        if (data.status === 'SUCCESS') {
          setTimeout(() => {
            if (!cancelled) onComplete(documentId);
          }, 1200);
        } else if (data.status === 'ERROR') {
          setError(data.error || 'Classification failed');
        }
      } catch (err) {
        if (!cancelled) {
          clearInterval(timer);
          setError(err.response?.data?.detail || err.message || 'Scrutiny pipeline failed');
        }
      }
    };

    runClassification();

    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [documentId, onComplete]);

  const renderStage = (stage, index) => {
    let state = 'pending';
    if (isCompleted || index < currentStep) {
      state = 'done';
    } else if (index === currentStep && !error) {
      state = 'processing';
    } else if (error && index === currentStep) {
      state = 'error';
    }

    return (
      <div key={stage.key} className="flex items-center space-x-3 my-2 text-xs">
        {state === 'pending' && (
          <div className="w-4 h-4 rounded-full border border-slate-300 flex items-center justify-center text-[9px] text-slate-400 font-mono shrink-0">
            {index + 1}
          </div>
        )}
        {state === 'processing' && <Loader2 size={16} className="text-blue-600 animate-spin shrink-0" />}
        {state === 'done' && <CheckCircle2 size={16} className="text-emerald-600 shrink-0" />}
        {state === 'error' && <XCircle size={16} className="text-rose-600 shrink-0" />}
        <span
          className={`${
            state === 'done'
              ? 'text-slate-900 font-semibold'
              : state === 'processing'
              ? 'text-blue-700 font-bold'
              : 'text-slate-400 font-medium'
          }`}
        >
          {stage.label}
        </span>
      </div>
    );
  };

  return (
    <div className="fixed inset-0 bg-slate-950/60 backdrop-blur-xs flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl p-6 w-[460px] max-w-full border border-slate-200 animate-fadeIn">
        <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100">
          <div className="flex items-center space-x-2">
            <div className="w-7 h-7 rounded-md bg-[#0F2942] text-amber-300 flex items-center justify-center">
              <Sparkles size={15} />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">AI Document Scrutiny Engine</h3>
              <p className="text-[11px] text-slate-500">Government CET Scrutiny Verification Pipeline</p>
            </div>
          </div>
        </div>

        <div className="space-y-1 my-4 bg-slate-50 p-3 rounded-lg border border-slate-200">
          {PIPELINE_STAGES.map((s, idx) => renderStage(s, idx))}
        </div>

        {result && result.status === 'SUCCESS' && (
          <div className="bg-emerald-50 border border-emerald-300 rounded-lg p-3 my-3">
            <p className="text-xs text-emerald-800 font-bold flex items-center gap-1.5">
              <CheckCircle2 size={14} className="text-emerald-600" />
              Document Scrutinized Successfully
            </p>
            <div className="mt-1.5 text-xs text-emerald-900 flex justify-between font-medium">
              <span>Category: <strong>{result.predicted_class}</strong></span>
              <span>Confidence: <strong>{(result.overall_confidence * 100).toFixed(1)}%</strong></span>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-rose-50 border border-rose-300 rounded-lg p-3 my-3 text-xs text-rose-800">
            <p className="font-bold flex items-center gap-1.5">
              <XCircle size={14} className="text-rose-600" />
              Scrutiny Error
            </p>
            <p className="mt-1 text-slate-700">{error}</p>
          </div>
        )}

        <div className="mt-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-200 hover:bg-slate-300 text-slate-800 rounded-md text-xs font-semibold transition cursor-pointer"
          >
            Close Window
          </button>
        </div>
      </div>
    </div>
  );
};

const IncomingDocuments = () => {
  const navigate = useNavigate();
  const context = useOutletContext();
  const refreshTrigger = context?.refreshTrigger || 0;

  const [documents, setDocuments] = useState([]);
  const [candidates, setCandidates] = useState({});
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [typeFilter, setTypeFilter] = useState('ALL');
  const [processingDocId, setProcessingDocId] = useState(null);
  const [batchProcessing, setBatchProcessing] = useState(false);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });
  const [resetting, setResetting] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [docsData, candData] = await Promise.all([
        apiService.getDocuments(),
        apiService.getCandidates(),
      ]);

      const candMap = {};
      if (Array.isArray(candData)) {
        candData.forEach((c) => {
          candMap[c.candidate_id] = c;
        });
      }

      setCandidates(candMap);
      setDocuments(Array.isArray(docsData) ? docsData : []);
    } catch (err) {
      console.error('Failed to load documents:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData, refreshTrigger]);

  const handleStartClassification = (docId) => {
    setProcessingDocId(docId);
  };

  const handleModalComplete = (docId) => {
    setProcessingDocId(null);
    fetchData();
    navigate(`/result/${docId}`);
  };

  // Batch classification of all pending documents
  const handleBatchClassify = async () => {
    const pendingDocs = filteredDocuments.filter(
      (d) => d.status === 'PENDING' || !d.document_type
    );
    if (pendingDocs.length === 0) return;

    setBatchProcessing(true);
    setBatchProgress({ current: 0, total: pendingDocs.length });

    for (let i = 0; i < pendingDocs.length; i++) {
      setBatchProgress({ current: i + 1, total: pendingDocs.length });
      try {
        await apiService.classifyDocument(pendingDocs[i].document_id);
      } catch (err) {
        console.error(`Failed to classify document ${pendingDocs[i].document_id}:`, err);
      }
    }

    setBatchProcessing(false);
    fetchData();
  };

  const handleResetScrutiny = async () => {
    const confirmReset = window.confirm(
      'Reset all document scrutiny? This will return all documents to PENDING and clear previous OCR cache so you can scrutinize every certificate fresh in real time.'
    );
    if (!confirmReset) return;

    try {
      setResetting(true);
      await apiService.resetScrutiny();
      await fetchData();
    } catch (err) {
      console.error('Failed to reset scrutiny:', err);
    } finally {
      setResetting(false);
    }
  };

  // Filtered documents calculation
  const filteredDocuments = useMemo(() => {
    return documents.filter((doc) => {
      const candidate = candidates[doc.candidate_id];
      const candName = candidate ? candidate.name.toLowerCase() : '';
      const candEnroll = candidate ? candidate.enrollment_number.toLowerCase() : '';
      const docName = (doc.filename || '').toLowerCase();
      const q = searchQuery.toLowerCase().trim();

      const matchesSearch =
        !q || candName.includes(q) || candEnroll.includes(q) || docName.includes(q);

      const matchesStatus =
        statusFilter === 'ALL' ||
        (statusFilter === 'PENDING' && (doc.status === 'PENDING' || !doc.document_type)) ||
        (statusFilter === 'CLASSIFIED' && doc.status === 'CLASSIFIED');

      const matchesType =
        typeFilter === 'ALL' ||
        (doc.document_type && doc.document_type.toUpperCase() === typeFilter);

      return matchesSearch && matchesStatus && matchesType;
    });
  }, [documents, candidates, searchQuery, statusFilter, typeFilter]);

  // Summary Metrics
  const stats = useMemo(() => {
    const total = documents.length;
    const classified = documents.filter((d) => d.document_type).length;
    const pending = total - classified;
    const validity = documents.filter(
      (d) => d.document_type === 'CASTE_VALIDITY_CERTIFICATE'
    ).length;
    const outOfScope = documents.filter(
      (d) => d.document_type === 'UNKNOWN_OUT_OF_SCOPE'
    ).length;
    return { total, classified, pending, validity, outOfScope };
  }, [documents]);

  return (
    <div className="space-y-6">
      {/* Top Header & Metrics Banner */}
      <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-100">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-blue-900 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                Document Scrutiny Repository
              </span>
              <span className="text-xs text-slate-500">
                State CET Admissions 2026-27
              </span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight mt-1">
              Candidate Document Scrutiny & Ingestion
            </h2>
            <p className="text-xs text-slate-500">
              Scrutinize, verify, and classify broad certificate categories using fine-tuned Multimodal AI.
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={fetchData}
              disabled={loading || batchProcessing || resetting}
              className="flex items-center space-x-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-2 rounded-lg text-xs font-semibold transition border border-slate-300 cursor-pointer"
            >
              <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
              <span>Refresh Vault</span>
            </button>

            <button
              onClick={handleResetScrutiny}
              disabled={resetting || batchProcessing}
              className="flex items-center space-x-1.5 bg-slate-100 hover:bg-rose-50 text-slate-700 hover:text-rose-700 px-3 py-2 rounded-lg text-xs font-semibold transition border border-slate-300 hover:border-rose-300 cursor-pointer disabled:opacity-50"
              title="Reset all documents to PENDING and clear OCR cache"
            >
              <RotateCcw size={13} className={resetting ? 'animate-spin text-rose-600' : 'text-slate-500'} />
              <span>{resetting ? 'Resetting...' : 'Reset Scrutiny'}</span>
            </button>

            <button
              onClick={handleBatchClassify}
              disabled={batchProcessing || resetting || stats.pending === 0}
              className="flex items-center space-x-1.5 bg-[#0F2942] hover:bg-[#1A3D60] text-amber-300 px-4 py-2 rounded-lg text-xs font-bold shadow-sm transition border border-amber-500/30 disabled:opacity-50 cursor-pointer"
            >
              {batchProcessing ? (
                <>
                  <Loader2 size={14} className="animate-spin text-amber-400" />
                  <span>Processing ({batchProgress.current}/{batchProgress.total})...</span>
                </>
              ) : (
                <>
                  <Sparkles size={14} className="text-amber-400" />
                  <span>Scrutinize All Pending ({stats.pending})</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Quick Stats Chips */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mt-4">
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-center">
            <span className="text-[10px] font-bold text-slate-500 uppercase">Total Vault Files</span>
            <p className="text-lg font-black text-slate-900 mt-0.5">{stats.total}</p>
          </div>
          <div className="bg-emerald-50/70 p-3 rounded-lg border border-emerald-200 text-center">
            <span className="text-[10px] font-bold text-emerald-700 uppercase">Verified / Classified</span>
            <p className="text-lg font-black text-emerald-800 mt-0.5">{stats.classified}</p>
          </div>
          <div className="bg-amber-50/70 p-3 rounded-lg border border-amber-200 text-center">
            <span className="text-[10px] font-bold text-amber-700 uppercase">Pending Scrutiny</span>
            <p className="text-lg font-black text-amber-800 mt-0.5">{stats.pending}</p>
          </div>
          <div className="bg-blue-50/70 p-3 rounded-lg border border-blue-200 text-center">
            <span className="text-[10px] font-bold text-blue-700 uppercase">Validity Certificates</span>
            <p className="text-lg font-black text-blue-800 mt-0.5">{stats.validity}</p>
          </div>
          <div className="bg-rose-50/70 p-3 rounded-lg border border-rose-200 text-center col-span-2 sm:col-span-1">
            <span className="text-[10px] font-bold text-rose-700 uppercase">Out-of-Scope (NCL)</span>
            <p className="text-lg font-black text-rose-800 mt-0.5">{stats.outOfScope}</p>
          </div>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs flex flex-wrap items-center justify-between gap-3">
        {/* Search */}
        <div className="relative flex-1 min-w-[260px]">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search by student name (e.g. TAWARE), enrollment (e.g. EN20260001), or file..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-hidden focus:ring-2 focus:ring-blue-500 focus:bg-white"
          />
        </div>

        {/* Filter Chips */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-slate-50 border border-slate-300 text-slate-700 text-xs rounded-lg px-2.5 py-2 font-medium focus:ring-2 focus:ring-blue-500 cursor-pointer"
          >
            <option value="ALL">Status: All Records</option>
            <option value="PENDING">Status: Pending Scrutiny</option>
            <option value="CLASSIFIED">Status: Verified & Classified</option>
          </select>

          {/* Type Filter */}
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="bg-slate-50 border border-slate-300 text-slate-700 text-xs rounded-lg px-2.5 py-2 font-medium focus:ring-2 focus:ring-blue-500 cursor-pointer"
          >
            <option value="ALL">Category: All Types</option>
            <option value="CASTE_VALIDITY_CERTIFICATE">Caste Validity Certificate</option>
            <option value="CASTE_CERTIFICATE">Caste Certificate</option>
            <option value="CASTE_VALIDITY_RECEIPT">Caste Validity Receipt</option>
            <option value="PROFORMA_O">Proforma-O</option>
            <option value="LEAVING_CERTIFICATE">Leaving Certificate</option>
            <option value="UNKNOWN_OUT_OF_SCOPE">Non-Creamy Layer / Out-of-Scope</option>
          </select>
        </div>
      </div>

      {/* Document Records Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-100/80 border-b border-slate-200 text-[11px] font-bold text-slate-700 uppercase tracking-wider">
                <th className="py-3.5 px-4">Candidate Information</th>
                <th className="py-3.5 px-4">Original File (Vault)</th>
                <th className="py-3.5 px-4">Classified Document Type</th>
                <th className="py-3.5 px-4">Verification Status</th>
                <th className="py-3.5 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-xs">
              {loading ? (
                <tr>
                  <td colSpan="5" className="py-12 text-center text-slate-500">
                    <Loader2 size={24} className="animate-spin mx-auto text-blue-600 mb-2" />
                    Loading document repository from database...
                  </td>
                </tr>
              ) : filteredDocuments.length === 0 ? (
                <tr>
                  <td colSpan="5" className="py-12 text-center text-slate-500">
                    <Layers size={28} className="mx-auto text-slate-400 mb-2" />
                    No documents matched the specified search criteria.
                  </td>
                </tr>
              ) : (
                filteredDocuments.map((doc) => {
                  const candidate = candidates[doc.candidate_id];
                  const hasClassification = Boolean(doc.document_type);

                  return (
                    <tr
                      key={doc.document_id}
                      className="hover:bg-blue-50/40 transition-colors duration-100"
                    >
                      {/* Candidate Column */}
                      <td className="py-3.5 px-4">
                        <div className="flex items-center space-x-3">
                          <div className="w-8 h-8 rounded-full bg-[#0F2942] text-amber-300 flex items-center justify-center font-bold text-xs shrink-0">
                            {candidate ? candidate.name.charAt(0) : 'C'}
                          </div>
                          <div>
                            <span className="font-bold text-slate-900 block">
                              {candidate ? candidate.name : 'Unknown Candidate'}
                            </span>
                            <div className="flex items-center space-x-1.5 mt-0.5">
                              <span className="font-mono text-[10px] font-semibold text-blue-800 bg-blue-50 px-1.5 py-0.2 rounded border border-blue-200">
                                {candidate ? candidate.enrollment_number : '—'}
                              </span>
                              <span className="text-[10px] font-medium text-slate-500">
                                {candidate?.reserve_category || 'OBC'}
                              </span>
                            </div>
                          </div>
                        </div>
                      </td>

                      {/* File Column */}
                      <td className="py-3.5 px-4">
                        <div className="flex items-center space-x-2">
                          <div className="p-1.5 rounded-md bg-slate-100 text-slate-600 shrink-0">
                            <FileText size={16} />
                          </div>
                          <div className="max-w-[200px] truncate" title={doc.filename}>
                            <span className="font-medium text-slate-800 block truncate">
                              {doc.filename}
                            </span>
                            <span className="text-[10px] text-slate-400">
                              {doc.filename.endsWith('.pdf') ? 'PDF Document' : 'Image File'}
                            </span>
                          </div>
                        </div>
                      </td>

                      {/* Type Column */}
                      <td className="py-3.5 px-4">
                        {doc.document_type ? (
                          <StatusBadge status={doc.document_type} />
                        ) : (
                          <span className="text-[11px] text-slate-400 italic">
                            Unclassified (Pending)
                          </span>
                        )}
                      </td>

                      {/* Status Column */}
                      <td className="py-3.5 px-4">
                        <StatusBadge status={doc.status} />
                      </td>

                      {/* Actions Column */}
                      <td className="py-3.5 px-4 text-right">
                        <div className="flex items-center justify-end space-x-2">
                          {hasClassification ? (
                            <button
                              onClick={() => navigate(`/result/${doc.document_id}`)}
                              className="inline-flex items-center space-x-1 text-xs font-bold text-blue-700 hover:text-blue-900 bg-blue-50 hover:bg-blue-100 px-3 py-1.5 rounded-lg border border-blue-200 transition cursor-pointer"
                            >
                              <Eye size={13} />
                              <span>View Scrutiny</span>
                            </button>
                          ) : (
                            <button
                              onClick={() => handleStartClassification(doc.document_id)}
                              className="inline-flex items-center space-x-1 text-xs font-bold text-amber-900 hover:text-amber-950 bg-amber-300 hover:bg-amber-400 px-3 py-1.5 rounded-lg shadow-xs transition cursor-pointer"
                            >
                              <Play size={13} fill="currentColor" />
                              <span>Scrutinize</span>
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Processing Modal */}
      {processingDocId && (
        <ProcessingModal
          documentId={processingDocId}
          onClose={() => setProcessingDocId(null)}
          onComplete={handleModalComplete}
        />
      )}
    </div>
  );
};

export default IncomingDocuments;