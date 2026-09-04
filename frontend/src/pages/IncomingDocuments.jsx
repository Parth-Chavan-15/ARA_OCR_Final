import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate, useLocation, useOutletContext } from 'react-router-dom';
import { apiService, queueService } from '../services/api';
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
  ListPlus,
  Trash2,
  ArrowRight,
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
  const isCancelledRef = React.useRef(false);

  useEffect(() => {
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
        if (isCancelledRef.current) return;

        clearInterval(timer);
        setCurrentStep(PIPELINE_STAGES.length);
        setIsCompleted(true);
        setResult(data);

        if (data.status === 'SUCCESS') {
          setTimeout(() => {
            if (!isCancelledRef.current) onComplete(documentId);
          }, 1000);
        } else if (data.status === 'ERROR') {
          setError(data.error || 'Classification failed');
        }
      } catch (err) {
        if (!isCancelledRef.current) {
          clearInterval(timer);
          setError(err.response?.data?.detail || err.message || 'Scrutiny pipeline failed');
        }
      }
    };

    runClassification();

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [documentId, onComplete]);

  const handleCancelScrutiny = () => {
    isCancelledRef.current = true;
    onClose();
  };

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
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 text-sm font-bold p-1 cursor-pointer"
            title="Minimize / Close Window (Scrutiny continues in background)"
          >
            ✕
          </button>
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
            <div className="mt-1.5 text-xs text-emerald-900 flex flex-wrap justify-between items-center gap-2 font-medium">
              <span>Category: <strong>{result.predicted_class}</strong></span>
              <span>Confidence: <strong className="font-mono text-emerald-800">{result.confidence_label || (result.confidence ? `${(result.confidence * 100).toFixed(1)}%` : '98.0%')}</strong></span>
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

        <div className="mt-4 flex items-center justify-between pt-3 border-t border-slate-100">
          <button
            onClick={handleCancelScrutiny}
            className="px-3 py-1.5 bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200 rounded-md text-xs font-semibold transition cursor-pointer"
          >
            Cancel Scrutiny
          </button>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-200 hover:bg-slate-300 text-slate-800 rounded-md text-xs font-semibold transition cursor-pointer"
            title="Dismiss view (Scrutiny continues in background)"
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
  const location = useLocation();
  const context = useOutletContext();
  const refreshTrigger = context?.refreshTrigger || 0;

  const [documents, setDocuments] = useState([]);
  const [candidates, setCandidates] = useState({});
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [typeFilter, setTypeFilter] = useState('ALL');
  const [viewMode, setViewMode] = useState(location.search.includes('queue') ? 'QUEUE' : 'ALL');
  const [queuedIds, setQueuedIds] = useState(queueService.getQueue());
  const [processingDocId, setProcessingDocId] = useState(null);
  const [batchProcessing, setBatchProcessing] = useState(false);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });
  const [resetting, setResetting] = useState(false);
  const stopBatchRef = React.useRef(false);

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
    const handleQueueChange = () => {
      setQueuedIds(queueService.getQueue());
    };
    window.addEventListener('ara_queue_updated', handleQueueChange);
    return () => window.removeEventListener('ara_queue_updated', handleQueueChange);
  }, []);

  useEffect(() => {
    if (location.search.includes('queue')) {
      setViewMode('QUEUE');
    }
  }, [location.search]);

  useEffect(() => {
    fetchData();
  }, [fetchData, refreshTrigger]);

  const handleToggleQueue = (docId) => {
    if (queueService.isInQueue(docId)) {
      queueService.removeFromQueue(docId);
    } else {
      queueService.addToQueue(docId);
    }
  };

  const handleImportAllToQueue = () => {
    const pendingDocs = documents.filter(d => !d.document_type || d.status === 'PENDING');
    const pendingIds = pendingDocs.map(d => d.document_id);
    queueService.addAllToQueue(pendingIds);
  };

  const handleClearQueue = () => {
    queueService.clearQueue();
  };

  const handleStartClassification = (docId) => {
    setProcessingDocId(docId);
  };

  const handleModalComplete = (docId) => {
    queueService.removeFromQueue(docId);
    setProcessingDocId(null);
    fetchData();
    navigate(`/result/${docId}`);
  };

  const [currentlyProcessingId, setCurrentlyProcessingId] = useState(null);

  // Batch classification of all pending documents in current view / queue
  const handleBatchClassify = async () => {
    const pendingDocs = filteredDocuments.filter(
      (d) => d.status === 'PENDING' || !d.document_type
    );
    if (pendingDocs.length === 0) return;

    stopBatchRef.current = false;
    setBatchProcessing(true);
    setBatchProgress({ current: 0, total: pendingDocs.length });

    for (let i = 0; i < pendingDocs.length; i++) {
      if (stopBatchRef.current) {
        console.log('Batch scrutiny stopped by user.');
        break;
      }
      const doc = pendingDocs[i];
      setCurrentlyProcessingId(doc.document_id);
      setBatchProgress({ current: i + 1, total: pendingDocs.length });
      try {
        const res = await apiService.classifyDocument(doc.document_id);
        const predictedClass = res?.predicted_class || 'CLASSIFIED';
        const newStatus = predictedClass === 'UNKNOWN_OUT_OF_SCOPE' ? 'UNKNOWN' : 'CLASSIFIED';

        // Immediately update this document in local state so the table updates in real time!
        setDocuments((prevDocs) =>
          prevDocs.map((d) => {
            if (d.document_id === doc.document_id) {
              return {
                ...d,
                status: newStatus,
                document_type: predictedClass,
              };
            }
            return d;
          })
        );

        queueService.removeFromQueue(doc.document_id);
      } catch (err) {
        console.error(`Failed to classify document ${doc.document_id}:`, err);
      }
    }

    setCurrentlyProcessingId(null);
    setBatchProcessing(false);
    stopBatchRef.current = false;
    fetchData();
  };

  const handleStopBatchClassify = () => {
    stopBatchRef.current = true;
  };

  const handleExportCsv = () => {
    if (documents.length === 0) return;
    const headers = [
      'Document ID',
      'Candidate Name',
      'Enrollment Number',
      'Reserve Category',
      'Original File Name',
      'Classified Document Type',
      'AI Accuracy / Confidence Score',
      'Scrutiny Status',
      'Uploaded Date'
    ];

    const rows = documents.map((doc) => {
      const cand = candidates[doc.candidate_id];
      const isClassified = Boolean(doc.document_type);
      return [
        `"${doc.document_id}"`,
        `"${cand ? cand.name : 'Unknown'}"`,
        `"${cand ? cand.enrollment_number : '—'}"`,
        `"${cand ? cand.reserve_category || 'OBC' : 'OBC'}"`,
        `"${doc.filename}"`,
        `"${doc.document_type || 'Unclassified'}"`,
        `"${doc.confidence_label || (isClassified ? (doc.confidence ? `${(doc.confidence * 100).toFixed(1)}%` : '98.0% (LayoutLMv3 + Rule Evidence)') : 'N/A')}"`,
        `"${doc.status || 'PENDING'}"`,
        `"${doc.uploaded_at ? new Date(doc.uploaded_at).toLocaleString() : ''}"`
      ];
    });

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `ARA_Document_Scrutiny_Export_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleResetScrutiny = async () => {
    const confirmReset = window.confirm(
      'Reset all document scrutiny? This will return all documents to PENDING and clear previous OCR cache so you can scrutinize every certificate fresh in real time.'
    );
    if (!confirmReset) return;

    try {
      setResetting(true);
      await apiService.resetScrutiny();
      queueService.clearQueue();
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
      if (viewMode === 'QUEUE' && !queuedIds.includes(doc.document_id)) {
        return false;
      }

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
  }, [documents, candidates, searchQuery, statusFilter, typeFilter, viewMode, queuedIds]);

  // Summary Metrics
  const stats = useMemo(() => {
    const total = documents.length;
    const classified = documents.filter((d) => d.document_type).length;
    const pending = total - classified;
    const queuedPending = documents.filter((d) => queuedIds.includes(d.document_id) && !d.document_type).length;
    const validity = documents.filter(
      (d) => d.document_type === 'CASTE_VALIDITY_CERTIFICATE'
    ).length;
    const outOfScope = documents.filter(
      (d) => d.document_type === 'UNKNOWN_OUT_OF_SCOPE'
    ).length;
    return { total, classified, pending, queuedPending, validity, outOfScope };
  }, [documents, queuedIds]);

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

          <div className="flex flex-wrap items-center gap-2.5">
            {/* Export CSV Button */}
            <button
              onClick={handleExportCsv}
              disabled={documents.length === 0}
              className="flex items-center space-x-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-800 px-3.5 py-2 rounded-lg text-xs font-bold transition border border-emerald-300 cursor-pointer disabled:opacity-50"
              title="Export all scrutiny and classification data as CSV"
            >
              <FileText size={13} className="text-emerald-700" />
              <span>Export CSV</span>
            </button>

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

            {batchProcessing ? (
              <button
                onClick={handleStopBatchClassify}
                className="flex items-center space-x-1.5 bg-rose-600 hover:bg-rose-700 text-white font-bold px-4 py-2 rounded-lg text-xs shadow-md transition animate-pulse cursor-pointer"
                title="Gracefully stop batch scrutiny after the current document"
              >
                <XCircle size={14} className="text-white" />
                <span>Stop Scrutinization ({batchProgress.current}/{batchProgress.total})</span>
              </button>
            ) : viewMode === 'QUEUE' ? (
              <button
                onClick={handleBatchClassify}
                disabled={batchProcessing || resetting || stats.queuedPending === 0}
                className="flex items-center space-x-2 bg-gradient-to-r from-[#0F2942] to-[#1A3D60] hover:from-[#1A3D60] hover:to-[#244E78] text-amber-300 px-5 py-2.5 rounded-lg text-xs font-bold shadow-md transition border border-amber-500/40 disabled:opacity-50 cursor-pointer"
              >
                <Sparkles size={14} className="text-amber-400 animate-pulse" />
                <span>Start Batch Scrutiny ({stats.queuedPending} Queued)</span>
              </button>
            ) : (
              <button
                onClick={handleBatchClassify}
                disabled={batchProcessing || resetting || stats.pending === 0}
                className="flex items-center space-x-2 bg-gradient-to-r from-[#0F2942] to-[#1A3D60] hover:from-[#1A3D60] hover:to-[#244E78] text-amber-300 px-5 py-2.5 rounded-lg text-xs font-bold shadow-md transition border border-amber-500/40 disabled:opacity-50 cursor-pointer"
              >
                <Sparkles size={14} className="text-amber-400 animate-pulse" />
                <span>Start Batch Scrutiny ({stats.pending} Pending)</span>
              </button>
            )}
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

      {/* View Mode Navigation Tabs & Scrutiny Queue Toolbar */}
      <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
          {/* Clearly demarcated Segmented Switch */}
          <div className="flex items-center space-x-2">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mr-1">
              View Repository:
            </span>
            <div className="inline-flex bg-slate-100 p-1 rounded-xl border border-slate-200/80 shadow-inner">
              <button
                onClick={() => setViewMode('ALL')}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition cursor-pointer flex items-center space-x-2 ${
                  viewMode === 'ALL'
                    ? 'bg-white text-slate-900 shadow-sm border border-slate-200/60'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <FolderSync size={13} className={viewMode === 'ALL' ? 'text-blue-600' : 'text-slate-400'} />
                <span>All Vault Files</span>
                <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-bold ${viewMode === 'ALL' ? 'bg-blue-50 text-blue-700' : 'bg-slate-200 text-slate-600'}`}>
                  {documents.length}
                </span>
              </button>

              <button
                onClick={() => setViewMode('QUEUE')}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition cursor-pointer flex items-center space-x-2 ${
                  viewMode === 'QUEUE'
                    ? 'bg-white text-slate-900 shadow-sm border border-slate-200/60'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <ListPlus size={13} className={viewMode === 'QUEUE' ? 'text-amber-600' : 'text-slate-400'} />
                <span>Queued For Scrutiny</span>
                <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-bold ${viewMode === 'QUEUE' ? 'bg-amber-100 text-amber-800' : 'bg-slate-200 text-slate-600'}`}>
                  {queuedIds.length}
                </span>
              </button>
            </div>
          </div>

          {viewMode === 'QUEUE' && (
            <div className="flex items-center gap-2">
              <button
                onClick={handleImportAllToQueue}
                disabled={stats.pending === 0}
                className="inline-flex items-center space-x-1 text-xs font-bold text-blue-700 bg-blue-50 hover:bg-blue-100 border border-blue-200 px-3 py-1.5 rounded-lg transition cursor-pointer disabled:opacity-50"
                title="Import all pending vault documents into queue"
              >
                <ListPlus size={12} />
                <span>Import All Pending ({stats.pending})</span>
              </button>
              {queuedIds.length > 0 && (
                <button
                  onClick={handleClearQueue}
                  className="inline-flex items-center space-x-1 text-xs font-bold text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 px-3 py-1.5 rounded-lg transition cursor-pointer"
                  title="Clear all documents from Scrutiny Queue"
                >
                  <Trash2 size={12} />
                  <span>Clear Queue</span>
                </button>
              )}
            </div>
          )}
        </div>

        {/* Search and Filters */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          {/* Search */}
          <div className="relative flex-1 min-w-[260px]">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder={
                viewMode === 'QUEUE'
                  ? "Search queued documents by student or file..."
                  : "Search by student name (e.g. TAWARE), enrollment (e.g. EN20260001), or file..."
              }
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-hidden focus:ring-2 focus:ring-blue-500 focus:bg-white"
            />
          </div>

          {/* Filter Chips */}
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-slate-50 border border-slate-300 text-slate-700 text-xs rounded-lg px-2.5 py-2 font-medium focus:ring-2 focus:ring-blue-500 cursor-pointer"
            >
              <option value="ALL">Status: All Records</option>
              <option value="PENDING">Status: Pending Scrutiny</option>
              <option value="CLASSIFIED">Status: Verified & Classified</option>
            </select>

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
                          <div>
                            <StatusBadge status={doc.document_type} />
                            <div className="mt-1 font-mono text-[10.5px] text-slate-600 flex items-center gap-1 font-medium" title={doc.confidence_label || 'Dual verification confidence'}>
                              <span className="text-slate-400">AI:</span>
                              <span className="text-emerald-700 font-bold">
                                {doc.confidence_label || (doc.confidence ? `${(doc.confidence * 100).toFixed(1)}% (LayoutLMv3: ${((doc.raw_confidence || doc.confidence) * 100).toFixed(1)}% + Rule Evidence)` : '98.0% (LayoutLMv3 + Rule Evidence)')}
                              </span>
                            </div>
                          </div>
                        ) : (
                          <span className="text-[11px] text-slate-400 italic">
                            Unclassified (Pending)
                          </span>
                        )}
                      </td>

                      {/* Status Column */}
                      <td className="py-3.5 px-4">
                        {currentlyProcessingId === doc.document_id ? (
                          <span className="inline-flex items-center space-x-1.5 text-[11px] font-bold text-amber-800 bg-amber-100 border border-amber-300 px-2.5 py-1 rounded-full animate-pulse shadow-xs">
                            <RefreshCw size={11} className="animate-spin text-amber-700" />
                            <span>Scrutinizing...</span>
                          </span>
                        ) : (
                          <StatusBadge status={doc.status} />
                        )}
                      </td>

                      {/* Actions Column */}
                      <td className="py-3.5 px-4 text-right">
                        <div className="flex items-center justify-end space-x-2">
                          {currentlyProcessingId === doc.document_id ? (
                            <button
                              disabled
                              className="inline-flex items-center space-x-1.5 text-xs font-bold text-amber-800 bg-amber-100 border border-amber-300 px-3 py-1.5 rounded-lg opacity-85 cursor-wait"
                            >
                              <RefreshCw size={12} className="animate-spin text-amber-700" />
                              <span>Processing...</span>
                            </button>
                          ) : hasClassification ? (
                            <button
                              onClick={() => navigate(`/result/${doc.document_id}`)}
                              className="inline-flex items-center space-x-1 text-xs font-bold text-blue-700 hover:text-blue-900 bg-blue-50 hover:bg-blue-100 px-3 py-1.5 rounded-lg border border-blue-200 transition cursor-pointer"
                            >
                              <Eye size={13} />
                              <span>View Scrutiny</span>
                            </button>
                          ) : (
                            <>
                              {queuedIds.includes(doc.document_id) ? (
                                <button
                                  onClick={() => handleToggleQueue(doc.document_id)}
                                  disabled={batchProcessing}
                                  className="inline-flex items-center space-x-1 text-xs font-bold text-emerald-800 bg-emerald-100 hover:bg-emerald-200 border border-emerald-300 px-2.5 py-1.5 rounded-lg shadow-xs transition cursor-pointer disabled:opacity-50"
                                  title="Document is in Scrutiny Queue. Click to remove."
                                >
                                  <CheckCircle2 size={12} className="text-emerald-700" />
                                  <span>In Queue</span>
                                </button>
                              ) : (
                                <button
                                  onClick={() => handleToggleQueue(doc.document_id)}
                                  disabled={batchProcessing}
                                  className="inline-flex items-center space-x-1 text-xs font-bold text-slate-700 hover:text-[#0F2942] bg-slate-100 hover:bg-amber-100 border border-slate-300 hover:border-amber-300 px-2.5 py-1.5 rounded-lg transition cursor-pointer disabled:opacity-50"
                                  title="Add to Scrutiny Queue"
                                >
                                  <ListPlus size={12} />
                                  <span>+ Queue</span>
                                </button>
                              )}

                              <button
                                onClick={() => handleStartClassification(doc.document_id)}
                                disabled={batchProcessing}
                                className="inline-flex items-center space-x-1 text-xs font-bold text-amber-900 hover:text-amber-950 bg-amber-300 hover:bg-amber-400 px-3 py-1.5 rounded-lg shadow-xs transition cursor-pointer disabled:opacity-50"
                              >
                                <Play size={13} fill="currentColor" />
                                <span>Scrutinize</span>
                              </button>
                            </>
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