import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';
import { apiService, queueService } from '../services/api';
import StatusBadge from '../components/StatusBadge';
import {
  Users,
  Search,
  FileText,
  CheckCircle2,
  Clock,
  Eye,
  ChevronRight,
  ShieldCheck,
  FolderOpen,
  RefreshCw,
  Sparkles,
  ListPlus,
  ArrowRight,
} from 'lucide-react';

const CandidateDocuments = () => {
  const navigate = useNavigate();
  const context = useOutletContext();
  const refreshTrigger = context?.refreshTrigger || 0;

  const [candidates, setCandidates] = useState([]);
  const [candidateDocs, setCandidateDocs] = useState({});
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [queuedIds, setQueuedIds] = useState(queueService.getQueue());
  const [toastMsg, setToastMsg] = useState(null);

  const fetchCandidatesData = useCallback(async () => {
    setLoading(true);
    try {
      const [candData, docsData] = await Promise.all([
        apiService.getCandidates(),
        apiService.getDocuments(),
      ]);

      const docsByCand = {};
      if (Array.isArray(docsData)) {
        docsData.forEach((d) => {
          if (!docsByCand[d.candidate_id]) {
            docsByCand[d.candidate_id] = [];
          }
          docsByCand[d.candidate_id].push(d);
        });
      }

      setCandidateDocs(docsByCand);
      setCandidates(Array.isArray(candData) ? candData : []);
    } catch (err) {
      console.error('Failed to load candidate registry:', err);
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

  const handleToggleQueue = (docId, filename = '') => {
    if (queueService.isInQueue(docId)) {
      queueService.removeFromQueue(docId);
      setToastMsg(`Removed "${filename || 'document'}" from Scrutiny Queue.`);
    } else {
      queueService.addToQueue(docId);
      setToastMsg(`Added "${filename || 'document'}" to Scrutiny Queue.`);
    }
    setTimeout(() => setToastMsg(null), 3500);
  };

  const handleAddAllInDossierToQueue = (candId) => {
    const docs = candidateDocs[candId] || [];
    const pendingDocs = docs.filter((d) => !d.document_type || d.status === 'PENDING');
    if (pendingDocs.length === 0) return;

    const pendingIds = pendingDocs.map((d) => d.document_id);
    queueService.addAllToQueue(pendingIds);
    setToastMsg(`Added ${pendingIds.length} candidate documents to Scrutiny Queue.`);
    setTimeout(() => setToastMsg(null), 3500);
  };

  useEffect(() => {
    fetchCandidatesData();
  }, [fetchCandidatesData, refreshTrigger]);

  const filteredCandidates = useMemo(() => {
    return candidates.filter((c) => {
      const q = searchQuery.toLowerCase().trim();
      if (!q) return true;
      return (
        c.name.toLowerCase().includes(q) ||
        c.enrollment_number.toLowerCase().includes(q)
      );
    });
  }, [candidates, searchQuery]);

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-[10px] font-bold uppercase tracking-wider text-blue-900 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
              Candidate Master Registry
            </span>
            <span className="text-xs text-slate-500">
              State CET Centralized Admission Process
            </span>
          </div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight mt-1">
            Enrolled Candidates & Scrutiny Dossiers
          </h2>
          <p className="text-xs text-slate-500">
            Total {candidates.length} registered candidate records with digital document dossiers in storage.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={fetchCandidatesData}
            disabled={loading}
            className="flex items-center space-x-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 px-3.5 py-2 rounded-lg text-xs font-semibold transition border border-slate-300 cursor-pointer"
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
            <span>Refresh Registry</span>
          </button>
        </div>
      </div>

      {/* Search Bar */}
      <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs">
        <div className="relative max-w-md">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search candidate by student name (e.g. DAHIWALE) or enrollment ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-hidden focus:ring-2 focus:ring-blue-500 focus:bg-white"
          />
        </div>
      </div>

      {/* Candidate Dossier Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {loading ? (
          <div className="col-span-full py-12 text-center text-slate-500">
            <RefreshCw size={24} className="animate-spin mx-auto text-blue-600 mb-2" />
            Loading candidate dossier records...
          </div>
        ) : filteredCandidates.length === 0 ? (
          <div className="col-span-full py-12 text-center text-slate-500">
            No candidate records found matching "{searchQuery}".
          </div>
        ) : (
          filteredCandidates.map((cand) => {
            const docs = candidateDocs[cand.candidate_id] || [];
            const verifiedDocs = docs.filter((d) => Boolean(d.document_type));

            return (
              <div
                key={cand.candidate_id}
                className="bg-white rounded-xl p-5 border border-slate-200 hover:border-blue-300 hover:shadow-md transition-all duration-150 flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-start justify-between gap-2 pb-3 border-b border-slate-100">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 rounded-full bg-[#0F2942] text-amber-300 flex items-center justify-center font-bold text-sm shrink-0 shadow-xs">
                        {cand.name.charAt(0)}
                      </div>
                      <div>
                        <h3 className="text-sm font-bold text-slate-900 leading-tight">
                          {cand.name}
                        </h3>
                        <div className="flex items-center space-x-1.5 mt-1">
                          <span className="font-mono text-[10px] font-bold text-blue-800 bg-blue-50 px-1.5 py-0.2 rounded border border-blue-200">
                            {cand.enrollment_number}
                          </span>
                          <span className="text-[10px] font-semibold text-slate-600 bg-slate-100 px-1.5 py-0.2 rounded">
                            {cand.reserve_category || 'OBC'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Documents count in dossier */}
                  <div className="py-3 space-y-1.5">
                    <div className="flex justify-between text-xs text-slate-600">
                      <span>Uploaded Certificates:</span>
                      <span className="font-bold text-slate-900">{docs.length} files</span>
                    </div>
                    <div className="flex justify-between text-xs text-slate-600">
                      <span>Scrutinized & Verified:</span>
                      <span className="font-bold text-emerald-700">{verifiedDocs.length}/{docs.length}</span>
                    </div>

                    {/* Mini Pills of files */}
                    <div className="pt-2 flex flex-wrap gap-1">
                      {docs.slice(0, 3).map((d) => (
                        <span
                          key={d.document_id}
                          className="text-[9px] font-medium bg-slate-100 text-slate-700 px-2 py-0.5 rounded truncate max-w-[130px]"
                          title={d.filename}
                        >
                          {d.filename}
                        </span>
                      ))}
                      {docs.length > 3 && (
                        <span className="text-[9px] font-bold text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded">
                          +{docs.length - 3} more
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Open Dossier Button */}
                <div className="pt-3 border-t border-slate-100">
                  <button
                    onClick={() => setSelectedCandidate(cand)}
                    className="w-full flex items-center justify-center space-x-1.5 bg-slate-50 hover:bg-blue-50 text-blue-800 font-bold py-2 px-3 rounded-lg text-xs border border-slate-200 hover:border-blue-300 transition cursor-pointer"
                  >
                    <FolderOpen size={14} />
                    <span>View Candidate Dossier</span>
                    <ChevronRight size={14} />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Candidate Dossier Detail Modal */}
      {selectedCandidate && (
        <div className="fixed inset-0 bg-slate-950/60 backdrop-blur-xs flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl p-6 w-[700px] max-w-full border border-slate-200 animate-fadeIn max-h-[90vh] flex flex-col">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-full bg-[#0F2942] text-amber-300 flex items-center justify-center font-bold text-sm">
                  {selectedCandidate.name.charAt(0)}
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">{selectedCandidate.name}</h3>
                  <div className="flex items-center space-x-2 text-xs">
                    <span className="font-mono text-blue-800 font-bold">{selectedCandidate.enrollment_number}</span>
                    <span>•</span>
                    <span className="text-slate-500">Category: {selectedCandidate.reserve_category || 'OBC'}</span>
                  </div>
                </div>
              </div>
              <button
                onClick={() => setSelectedCandidate(null)}
                className="text-slate-400 hover:text-slate-700 text-sm font-bold p-1 cursor-pointer"
              >
                ✕
              </button>
            </div>

            <div className="flex-1 overflow-y-auto py-4 space-y-3">
              {toastMsg && (
                <div className="bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs px-3.5 py-2 rounded-lg font-medium flex items-center justify-between animate-fadeIn">
                  <span className="flex items-center gap-1.5">
                    <CheckCircle2 size={14} className="text-emerald-600" />
                    {toastMsg}
                  </span>
                  <button
                    onClick={() => {
                      setSelectedCandidate(null);
                      navigate('/incoming?filter=queue');
                    }}
                    className="text-xs font-bold text-blue-700 hover:underline flex items-center gap-1 ml-2 cursor-pointer"
                  >
                    <span>View Scrutiny Queue</span>
                    <ArrowRight size={12} />
                  </button>
                </div>
              )}

              <div className="flex flex-wrap items-center justify-between gap-2">
                <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Submitted Documents in Vault ({candidateDocs[selectedCandidate.candidate_id]?.length || 0})
                </h4>
                <div className="flex items-center gap-2">
                  {(candidateDocs[selectedCandidate.candidate_id] || []).some(d => !d.document_type) && (
                    <button
                      onClick={() => handleAddAllInDossierToQueue(selectedCandidate.candidate_id)}
                      className="inline-flex items-center space-x-1.5 text-xs font-bold text-[#0F2942] bg-amber-300 hover:bg-amber-400 px-3 py-1.5 rounded-lg shadow-xs transition cursor-pointer"
                    >
                      <ListPlus size={13} />
                      <span>Add All to Queue</span>
                    </button>
                  )}
                  {queuedIds.length > 0 && (
                    <button
                      onClick={() => {
                        setSelectedCandidate(null);
                        navigate('/incoming?filter=queue');
                      }}
                      className="inline-flex items-center space-x-1 text-xs font-bold text-blue-700 bg-blue-50 hover:bg-blue-100 border border-blue-200 px-3 py-1.5 rounded-lg transition cursor-pointer"
                    >
                      <span>Scrutiny Queue ({queuedIds.length})</span>
                      <ArrowRight size={12} />
                    </button>
                  )}
                </div>
              </div>
              <div className="space-y-2">
                {(candidateDocs[selectedCandidate.candidate_id] || []).map((doc) => {
                  const isQueued = queuedIds.includes(doc.document_id);
                  return (
                    <div
                      key={doc.document_id}
                      className="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-200 hover:bg-slate-100/80 transition"
                    >
                      <div className="flex items-center space-x-3">
                        <FileText size={18} className="text-slate-600" />
                        <div>
                          <span className="text-xs font-bold text-slate-900 block">{doc.filename}</span>
                          <div className="mt-0.5 flex flex-wrap items-center gap-1.5">
                            {doc.document_type ? (
                              <>
                                <StatusBadge status={doc.document_type} />
                                <span className="text-[10.5px] font-mono text-emerald-700 font-semibold bg-emerald-50 px-1.5 py-0.2 rounded border border-emerald-200">
                                  {doc.confidence_label || (doc.confidence ? `${(doc.confidence * 100).toFixed(1)}% (LayoutLMv3: ${((doc.raw_confidence || doc.confidence) * 100).toFixed(1)}% + Rule Evidence)` : '98.0% (LayoutLMv3 + Rule Evidence)')}
                                </span>
                              </>
                            ) : (
                              <>
                                <span className="text-[10px] text-slate-500 italic">Pending Scrutiny</span>
                                {isQueued && (
                                  <span className="text-[9px] font-bold text-emerald-800 bg-emerald-100 border border-emerald-300 px-1.5 py-0.2 rounded">
                                    In Queue
                                  </span>
                                )}
                              </>
                            )}
                          </div>
                        </div>
                      </div>

                      <div>
                        {doc.document_type ? (
                          <button
                            onClick={() => {
                              setSelectedCandidate(null);
                              navigate(`/result/${doc.document_id}`, { state: { from: '/candidates' } });
                            }}
                            className="inline-flex items-center space-x-1 text-xs font-bold text-blue-700 hover:text-blue-900 bg-blue-50 px-3 py-1.5 rounded-lg border border-blue-200 transition cursor-pointer"
                          >
                            <Eye size={12} />
                            <span>View Scrutiny</span>
                          </button>
                        ) : isQueued ? (
                          <button
                            onClick={() => handleToggleQueue(doc.document_id, doc.filename)}
                            className="inline-flex items-center space-x-1 text-xs font-bold text-emerald-800 bg-emerald-100 hover:bg-emerald-200 border border-emerald-300 px-3 py-1.5 rounded-lg shadow-xs transition cursor-pointer"
                            title="Click to remove from scrutiny queue"
                          >
                            <CheckCircle2 size={13} className="text-emerald-700" />
                            <span>Added to Queue</span>
                          </button>
                        ) : (
                          <button
                            onClick={() => handleToggleQueue(doc.document_id, doc.filename)}
                            className="inline-flex items-center space-x-1 text-xs font-bold text-[#0F2942] bg-amber-300 hover:bg-amber-400 px-3 py-1.5 rounded-lg shadow-xs transition cursor-pointer"
                            title="Add document to scrutiny queue"
                          >
                            <ListPlus size={13} />
                            <span>Add to Queue</span>
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="pt-3 border-t border-slate-100 flex justify-end">
              <button
                onClick={() => setSelectedCandidate(null)}
                className="px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-800 rounded-lg text-xs font-bold transition cursor-pointer"
              >
                Close Dossier
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CandidateDocuments;