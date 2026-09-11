import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';
import { apiService, queueService } from '../services/api';
import StatusBadge from '../components/StatusBadge';
import {
  Building2,
  GraduationCap,
  Users,
  FileText,
  CheckCircle2,
  Clock,
  Eye,
  ChevronRight,
  ChevronLeft,
  ArrowRight,
  Download,
  RefreshCw,
  Search,
  Filter,
  Layers,
  Sparkles,
  ListPlus,
  ShieldCheck,
  AlertTriangle,
  FileCheck2,
  Calendar,
  Hash,
  Award,
} from 'lucide-react';

const InstituteHierarchy = () => {
  const navigate = useNavigate();
  const context = useOutletContext();
  const refreshTrigger = context?.refreshTrigger || 0;

  const [summaryData, setSummaryData] = useState(null);
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [selectedStream, setSelectedStream] = useState('ALL');
  const [selectedInstitute, setSelectedInstitute] = useState(null);

  // Candidates for selected institute
  const [candidates, setCandidates] = useState([]);
  const [loadingCandidates, setLoadingCandidates] = useState(false);
  const [candidateSearch, setCandidateSearch] = useState('');
  const [expandedCandidateId, setExpandedCandidateId] = useState(null);

  // Queue state
  const [queuedIds, setQueuedIds] = useState(queueService.getQueue());
  const [toastMsg, setToastMsg] = useState(null);

  useEffect(() => {
    const handleQueueChange = () => {
      setQueuedIds(queueService.getQueue());
    };
    window.addEventListener('ara_queue_updated', handleQueueChange);
    return () => window.removeEventListener('ara_queue_updated', handleQueueChange);
  }, []);

  const fetchSummary = useCallback(async () => {
    setLoadingSummary(true);
    try {
      const data = await apiService.getInstitutesSummary();
      setSummaryData(data);
    } catch (err) {
      console.error('Failed to load institutes summary:', err);
    } finally {
      setLoadingSummary(false);
    }
  }, []);

  useEffect(() => {
    fetchSummary();
  }, [fetchSummary, refreshTrigger]);

  // Load candidates when an institute is selected
  const fetchInstituteCandidates = useCallback(async (instCode, stream) => {
    setLoadingCandidates(true);
    try {
      const data = await apiService.getInstituteCandidates(instCode, stream === 'ALL' ? '' : stream);
      setCandidates(Array.isArray(data) ? data : []);
      if (data && data.length > 0) {
        setExpandedCandidateId(data[0].candidate_id);
      }
    } catch (err) {
      console.error('Failed to load institute candidates:', err);
    } finally {
      setLoadingCandidates(false);
    }
  }, []);

  const handleSelectInstitute = (inst) => {
    setSelectedInstitute(inst);
    fetchInstituteCandidates(inst.institute_code, inst.stream);
  };

  const handleBackToInstitutes = () => {
    setSelectedInstitute(null);
    setCandidates([]);
  };

  const handleToggleQueue = (docId) => {
    if (queueService.isInQueue(docId)) {
      queueService.removeFromQueue(docId);
    } else {
      queueService.addToQueue(docId);
    }
  };

  const handleAddAllInstituteToQueue = () => {
    if (!candidates || candidates.length === 0) return;
    const docIdsToAdd = [];
    candidates.forEach((cand) => {
      cand.documents.forEach((d) => {
        if (!d.document_type || d.status === 'PENDING') {
          docIdsToAdd.push(d.document_id);
        }
      });
    });

    if (docIdsToAdd.length > 0) {
      queueService.addAllToQueue(docIdsToAdd);
      setToastMsg(`Added ${docIdsToAdd.length} pending certificates to Scrutiny Queue.`);
      setTimeout(() => setToastMsg(null), 3500);
    }
  };

  // Filtered institutes based on stream
  const filteredInstitutes = useMemo(() => {
    if (!summaryData?.institutes) return [];
    if (selectedStream === 'ALL') return summaryData.institutes;
    return summaryData.institutes.filter((i) => i.stream === selectedStream);
  }, [summaryData, selectedStream]);

  // Filtered candidates in selected institute
  const filteredCandidates = useMemo(() => {
    if (!candidates) return [];
    const q = candidateSearch.toLowerCase().trim();
    if (!q) return candidates;
    return candidates.filter(
      (c) =>
        c.name.toLowerCase().includes(q) ||
        c.enrollment_number.toLowerCase().includes(q) ||
        (c.reserve_category && c.reserve_category.toLowerCase().includes(q))
    );
  }, [candidates, candidateSearch]);

  // CSV download trigger
  const handleDownloadCsv = (stream = '', instCode = '') => {
    const url = apiService.getInstituteExportCsvUrl(stream, instCode);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `ARA_Scrutiny_Export_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-100">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-blue-900 bg-blue-50 px-2 py-0.5 rounded border border-blue-200 flex items-center gap-1">
                <Building2 size={12} />
                Institute Scrutiny Hierarchy
              </span>
              <span className="text-xs text-slate-500">
                Post-Admission Centralized Verification
              </span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight mt-1">
              Stream & Institute-Wise Document Repositories
            </h2>
            <p className="text-xs text-slate-500">
              Audit admissions across medical, nursing, engineering, agriculture, and fine arts institutes as submitted to ARA.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <button
              onClick={() => handleDownloadCsv(selectedStream === 'ALL' ? '' : selectedStream, selectedInstitute?.institute_code || '')}
              className="flex items-center space-x-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-800 px-3.5 py-2 rounded-lg text-xs font-bold transition border border-emerald-300 cursor-pointer shadow-xs"
              title="Download structured CSV export including all extracted fields"
            >
              <Download size={13} className="text-emerald-700" />
              <span>Export Scrutiny CSV</span>
            </button>

            <button
              onClick={() => navigate('/incoming?filter=queue')}
              className="flex items-center space-x-1.5 bg-[#0F2942] hover:bg-[#1A3D60] text-amber-300 font-bold px-3.5 py-2 rounded-lg text-xs shadow-xs transition border border-amber-500/30 cursor-pointer"
              title="Navigate directly to Scrutiny Queue"
            >
              <ListPlus size={13} className="text-amber-400" />
              <span>Scrutiny Queue</span>
              {queuedIds.length > 0 && (
                <span className="bg-amber-400 text-slate-950 text-[10px] font-black px-1.5 py-0.2 rounded-full ml-1">
                  {queuedIds.length}
                </span>
              )}
            </button>

            <button
              onClick={fetchSummary}
              disabled={loadingSummary}
              className="flex items-center space-x-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 px-3.5 py-2 rounded-lg text-xs font-semibold transition border border-slate-300 cursor-pointer"
            >
              <RefreshCw size={13} className={loadingSummary ? 'animate-spin' : ''} />
              <span>Refresh Hierarchy</span>
            </button>
          </div>
        </div>

        {/* Global Summary Metric Chips */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
            <span className="text-[10px] font-bold text-slate-500 uppercase flex items-center gap-1">
              <GraduationCap size={12} className="text-blue-600" />
              Total Streams
            </span>
            <p className="text-lg font-black text-slate-900 mt-0.5">
              {summaryData?.total_streams || 0} Professional Streams
            </p>
          </div>

          <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
            <span className="text-[10px] font-bold text-slate-500 uppercase flex items-center gap-1">
              <Building2 size={12} className="text-amber-600" />
              Participating Institutes
            </span>
            <p className="text-lg font-black text-slate-900 mt-0.5">
              {summaryData?.total_institutes || 0} Registered Colleges
            </p>
          </div>

          <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
            <span className="text-[10px] font-bold text-slate-500 uppercase flex items-center gap-1">
              <Users size={12} className="text-indigo-600" />
              Enrolled Candidates
            </span>
            <p className="text-lg font-black text-slate-900 mt-0.5">
              {summaryData?.total_candidates || 0} Students
            </p>
          </div>

          <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
            <span className="text-[10px] font-bold text-slate-500 uppercase flex items-center gap-1">
              <FileText size={12} className="text-emerald-600" />
              Submitted Certificates
            </span>
            <p className="text-lg font-black text-slate-900 mt-0.5">
              {summaryData?.total_documents || 0} Vault Documents
            </p>
          </div>
        </div>
      </div>

      {/* Toast Alert */}
      {toastMsg && (
        <div className="bg-emerald-50 border border-emerald-300 text-emerald-900 px-4 py-3 rounded-xl text-xs font-semibold flex items-center justify-between shadow-sm animate-fadeIn">
          <span className="flex items-center gap-2">
            <CheckCircle2 size={16} className="text-emerald-600" />
            {toastMsg}
          </span>
          <button
            onClick={() => navigate('/incoming?filter=queue')}
            className="text-xs font-bold text-blue-700 hover:underline flex items-center gap-1 cursor-pointer"
          >
            <span>View Scrutiny Queue</span>
            <ArrowRight size={12} />
          </button>
        </div>
      )}

      {/* Main Content: Either Institute Grid or Candidate Drilldown */}
      {!selectedInstitute ? (
        <div className="space-y-4">
          {/* Stream Selector Tabs */}
          <div className="bg-white rounded-xl p-3 border border-slate-200 shadow-xs flex items-center space-x-2 overflow-x-auto">
            <span className="text-xs font-bold text-slate-500 uppercase px-2 shrink-0">
              Filter By Stream:
            </span>
            <button
              onClick={() => setSelectedStream('ALL')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition whitespace-nowrap cursor-pointer ${
                selectedStream === 'ALL'
                  ? 'bg-[#0F2942] text-amber-300 shadow-xs'
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
            >
              All Streams ({summaryData?.total_institutes || 0})
            </button>
            {(summaryData?.streams || []).map((stream) => (
              <button
                key={stream}
                onClick={() => setSelectedStream(stream)}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition whitespace-nowrap cursor-pointer ${
                  selectedStream === stream
                    ? 'bg-[#0F2942] text-amber-300 shadow-xs'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                {stream}
              </button>
            ))}
          </div>

          {/* Institutes Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {loadingSummary ? (
              <div className="col-span-full py-16 text-center text-slate-500">
                <RefreshCw size={24} className="animate-spin mx-auto text-blue-600 mb-2" />
                Loading institute summary from database...
              </div>
            ) : filteredInstitutes.length === 0 ? (
              <div className="col-span-full py-16 text-center text-slate-500">
                No institutes found under the selected filter.
              </div>
            ) : (
              filteredInstitutes.map((inst) => (
                <div
                  key={`${inst.stream}-${inst.institute_code}`}
                  className="bg-white rounded-xl p-5 border border-slate-200 hover:border-blue-300 hover:shadow-md transition-all duration-150 flex flex-col justify-between"
                >
                  <div>
                    {/* Top Row: Stream Badge & Institute Code */}
                    <div className="flex items-start justify-between gap-2 pb-2">
                      <span className="text-[10px] font-bold text-blue-800 bg-blue-50 px-2 py-0.5 rounded border border-blue-200 truncate max-w-[200px]" title={inst.stream}>
                        {inst.stream}
                      </span>
                      <span className="font-mono text-xs font-bold text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                        Code: {inst.institute_code}
                      </span>
                    </div>

                    {/* Institute Name */}
                    <h3 className="text-sm font-bold text-slate-900 mt-1 leading-snug">
                      {inst.institute_name}
                    </h3>

                    {/* Scrutiny Progress Bar */}
                    <div className="mt-4 space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-500">Verification Progress:</span>
                        <span className="font-bold text-slate-900">{inst.completion_rate}%</span>
                      </div>
                      <div className="w-full h-2.5 bg-slate-200/80 rounded-full overflow-hidden flex border border-slate-200/80">
                        <div
                          className="bg-emerald-500 h-full transition-all duration-300"
                          style={{ width: `${inst.completion_rate}%` }}
                        />
                        <div
                          className="bg-white h-full transition-all duration-300 border-l border-slate-200/60"
                          style={{ width: `${100 - inst.completion_rate}%` }}
                        />
                      </div>
                    </div>

                    {/* Metrics Grid */}
                    <div className="grid grid-cols-3 gap-2 mt-4 pt-3 border-t border-slate-100 text-center">
                      <div className="bg-slate-50 p-2 rounded">
                        <span className="text-[9px] font-semibold text-slate-500 block">Candidates</span>
                        <span className="text-xs font-bold text-slate-900">{inst.total_candidates}</span>
                      </div>
                      <div className="bg-slate-50 p-2 rounded">
                        <span className="text-[9px] font-semibold text-slate-500 block">Documents</span>
                        <span className="text-xs font-bold text-slate-900">{inst.total_documents}</span>
                      </div>
                      <div className="bg-slate-50 p-2 rounded">
                        <span className="text-[9px] font-semibold text-slate-500 block">Pending</span>
                        <span className="text-xs font-bold text-amber-700">{inst.pending_documents}</span>
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="mt-4 pt-3 border-t border-slate-100 flex items-center space-x-2">
                    <button
                      onClick={() => handleSelectInstitute(inst)}
                      className="flex-1 flex items-center justify-center space-x-1.5 bg-[#0F2942] hover:bg-[#1A3D60] text-amber-300 py-2 px-3 rounded-lg text-xs font-bold transition shadow-xs cursor-pointer"
                    >
                      <Users size={14} />
                      <span>View Candidates</span>
                      <ChevronRight size={14} />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      ) : (
        /* Candidate Drilldown View */
        <div className="space-y-4 animate-fadeIn">
          {/* Back Button & Institute Header Bar */}
          <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center space-x-3">
              <button
                onClick={handleBackToInstitutes}
                className="flex items-center space-x-1 bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-2 rounded-lg text-xs font-bold transition border border-slate-300 cursor-pointer"
              >
                <ChevronLeft size={14} />
                <span>All Institutes</span>
              </button>

              <div>
                <div className="flex items-center space-x-2">
                  <span className="text-[10px] font-bold text-blue-800 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                    {selectedInstitute.stream}
                  </span>
                  <span className="font-mono text-[10px] font-bold text-amber-800 bg-amber-50 px-1.5 py-0.5 rounded border border-amber-200">
                    Code: {selectedInstitute.institute_code}
                  </span>
                </div>
                <h3 className="text-base font-bold text-slate-900 mt-0.5">
                  {selectedInstitute.institute_name}
                </h3>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleAddAllInstituteToQueue}
                className="flex items-center space-x-1.5 bg-amber-300 hover:bg-amber-400 text-[#0F2942] font-bold px-3.5 py-2 rounded-lg text-xs shadow-xs transition cursor-pointer"
                title="Add all pending certificates from this institute to Scrutiny Queue"
              >
                <ListPlus size={14} />
                <span>Queue All Pending Docs</span>
              </button>

              <button
                onClick={() => navigate('/incoming?filter=queue')}
                className="flex items-center space-x-1.5 bg-[#0F2942] hover:bg-[#1A3D60] text-amber-300 font-bold px-3.5 py-2 rounded-lg text-xs shadow-xs transition border border-amber-500/30 cursor-pointer"
                title="Go directly to Scrutiny Queue"
              >
                <ArrowRight size={13} className="text-amber-400" />
                <span>Scrutiny Queue</span>
                {queuedIds.length > 0 && (
                  <span className="bg-amber-400 text-slate-950 text-[10px] font-black px-1.5 py-0.2 rounded-full ml-0.5">
                    {queuedIds.length}
                  </span>
                )}
              </button>

              <button
                onClick={() => handleDownloadCsv(selectedInstitute.stream, selectedInstitute.institute_code)}
                className="flex items-center space-x-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-800 font-bold px-3.5 py-2 rounded-lg text-xs border border-emerald-300 transition cursor-pointer"
                title="Download this institute's scrutiny roster CSV"
              >
                <Download size={13} />
                <span>Institute CSV</span>
              </button>
            </div>
          </div>

          {/* Search bar inside Institute */}
          <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs flex items-center justify-between gap-4">
            <div className="relative flex-1 max-w-md">
              <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                placeholder="Search candidates by student name or enrollment number..."
                value={candidateSearch}
                onChange={(e) => setCandidateSearch(e.target.value)}
                className="w-full pl-9 pr-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-hidden focus:ring-2 focus:ring-blue-500 focus:bg-white"
              />
            </div>
            <span className="text-xs text-slate-500 font-medium">
              Showing {filteredCandidates.length} of {candidates.length} enrolled candidates
            </span>
          </div>

          {/* Candidates List with Document Dossier Expander */}
          <div className="space-y-3">
            {loadingCandidates ? (
              <div className="bg-white rounded-xl p-12 text-center text-slate-500 border border-slate-200">
                <RefreshCw size={24} className="animate-spin mx-auto text-blue-600 mb-2" />
                Loading candidate dossiers for {selectedInstitute.institute_name}...
              </div>
            ) : filteredCandidates.length === 0 ? (
              <div className="bg-white rounded-xl p-12 text-center text-slate-500 border border-slate-200">
                No candidate records found matching "{candidateSearch}".
              </div>
            ) : (
              filteredCandidates.map((cand) => {
                const isExpanded = expandedCandidateId === cand.candidate_id;
                const hasPending = cand.documents.some((d) => !d.document_type || d.status === 'PENDING');

                return (
                  <div
                    key={cand.candidate_id}
                    className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden transition-all duration-150"
                  >
                    {/* Candidate Row Header */}
                    <div
                      onClick={() => setExpandedCandidateId(isExpanded ? null : cand.candidate_id)}
                      className="p-4 flex flex-wrap items-center justify-between gap-3 cursor-pointer hover:bg-slate-50/70 transition select-none"
                    >
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-full bg-[#0F2942] text-amber-300 flex items-center justify-center font-bold text-sm shrink-0 shadow-xs">
                          {cand.name.charAt(0)}
                        </div>
                        <div>
                          <h4 className="text-sm font-bold text-slate-900 leading-tight">
                            {cand.name}
                          </h4>
                          <div className="flex items-center space-x-2 mt-0.5">
                            <span className="font-mono text-[10px] font-bold text-blue-800 bg-blue-50 px-1.5 py-0.2 rounded border border-blue-200">
                              {cand.enrollment_number}
                            </span>
                            <span className="text-[10px] font-semibold text-slate-600 bg-slate-100 px-1.5 py-0.2 rounded">
                              Category: {cand.reserve_category}
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center space-x-3">
                        {/* Status chip */}
                        {cand.is_fully_verified ? (
                          <span className="inline-flex items-center space-x-1 text-xs font-bold text-emerald-800 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-full">
                            <CheckCircle2 size={13} className="text-emerald-600" />
                            <span>Fully Scrutinized ({cand.verified_documents}/{cand.total_documents})</span>
                          </span>
                        ) : (
                          <span className="inline-flex items-center space-x-1 text-xs font-bold text-amber-800 bg-amber-50 border border-amber-200 px-2.5 py-1 rounded-full">
                            <Clock size={13} className="text-amber-600" />
                            <span>Pending Scrutiny ({cand.verified_documents}/{cand.total_documents} verified)</span>
                          </span>
                        )}

                        <span className="text-slate-400 font-bold text-sm">
                          {isExpanded ? '▲' : '▼'}
                        </span>
                      </div>
                    </div>

                    {/* Expanded Document Dossier */}
                    {isExpanded && (
                      <div className="p-4 bg-slate-50/80 border-t border-slate-100 space-y-3 animate-fadeIn">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-bold text-slate-600 uppercase tracking-wider">
                            Submitted Documents & Certificate Findings ({cand.documents.length})
                          </span>
                          <div className="flex items-center space-x-2">
                            {hasPending && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const pendingIds = cand.documents
                                    .filter((d) => !d.document_type || d.status === 'PENDING')
                                    .map((d) => d.document_id);
                                  queueService.addAllToQueue(pendingIds);
                                  setToastMsg(`Added candidate's pending documents to queue.`);
                                  setTimeout(() => setToastMsg(null), 3500);
                                }}
                                className="inline-flex items-center space-x-1 text-[11px] font-bold text-[#0F2942] bg-amber-300 hover:bg-amber-400 px-2.5 py-1 rounded-md transition cursor-pointer"
                              >
                                <ListPlus size={12} />
                                <span>Queue Candidate Docs</span>
                              </button>
                            )}
                            {queuedIds.length > 0 && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  navigate('/incoming?filter=queue');
                                }}
                                className="inline-flex items-center space-x-1 text-[11px] font-bold text-blue-700 bg-blue-50 hover:bg-blue-100 border border-blue-200 px-2.5 py-1 rounded-md transition cursor-pointer shadow-2xs"
                                title="Navigate directly to Scrutiny Queue"
                              >
                                <span>Scrutiny Queue ({queuedIds.length})</span>
                                <ArrowRight size={11} />
                              </button>
                            )}
                          </div>
                        </div>

                        <div className="space-y-2">
                          {cand.documents.map((doc) => {
                            const isQueued = queuedIds.includes(doc.document_id);
                            const ext = doc.extracted_fields || {};
                            const hasExtracted = Object.keys(ext).length > 0;

                            return (
                              <div
                                key={doc.document_id}
                                className="bg-white p-3.5 rounded-lg border border-slate-200 shadow-2xs space-y-2"
                              >
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                  <div className="flex items-center space-x-2.5">
                                    <div className="p-1.5 rounded bg-blue-50 text-blue-700">
                                      <FileText size={16} />
                                    </div>
                                    <div>
                                      <span className="text-xs font-bold text-slate-900 block truncate max-w-[320px]" title={doc.filename}>
                                        {doc.filename}
                                      </span>
                                      <div className="flex items-center space-x-1.5 mt-0.5">
                                        {doc.document_type ? (
                                          <StatusBadge status={doc.document_type} />
                                        ) : (
                                          <span className="text-[10px] text-slate-400 italic">Unclassified (Pending)</span>
                                        )}
                                        {doc.confidence && (
                                          <span className="font-mono text-[10px] font-bold text-emerald-800 bg-emerald-50 px-1.5 py-0.2 rounded border border-emerald-200">
                                            {doc.confidence_label || `${(doc.confidence * 100).toFixed(1)}%`}
                                          </span>
                                        )}
                                      </div>
                                    </div>
                                  </div>

                                  {/* Document Action Button */}
                                  <div>
                                    {doc.document_type ? (
                                      <button
                                        onClick={() => navigate(`/result/${doc.document_id}`, { state: { from: '/institutes' } })}
                                        className="inline-flex items-center space-x-1 text-xs font-bold text-blue-700 hover:text-blue-900 bg-blue-50 px-3 py-1.5 rounded-lg border border-blue-200 transition cursor-pointer"
                                      >
                                        <Eye size={12} />
                                        <span>View Scrutiny Details</span>
                                      </button>
                                    ) : isQueued ? (
                                      <button
                                        onClick={() => handleToggleQueue(doc.document_id)}
                                        className="inline-flex items-center space-x-1 text-xs font-bold text-emerald-800 bg-emerald-100 hover:bg-emerald-200 border border-emerald-300 px-3 py-1.5 rounded-lg transition cursor-pointer"
                                      >
                                        <CheckCircle2 size={13} className="text-emerald-700" />
                                        <span>Added to Queue</span>
                                      </button>
                                    ) : (
                                      <button
                                        onClick={() => handleToggleQueue(doc.document_id, doc.filename)}
                                        className="inline-flex items-center space-x-1 text-xs font-bold text-[#0F2942] bg-amber-300 hover:bg-amber-400 px-3 py-1.5 rounded-lg transition cursor-pointer"
                                      >
                                        <ListPlus size={13} />
                                        <span>Add to Queue</span>
                                      </button>
                                    )}
                                  </div>
                                </div>

                                {/* Extracted Fields Ribbon */}
                                {hasExtracted && (
                                  <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100 flex flex-wrap items-center gap-2 text-[10.5px]">
                                    {(ext.certificate_no || ext.vc_no) && (
                                      <span className="bg-white px-2 py-0.5 rounded border border-slate-200 font-mono text-slate-800">
                                        <strong>VC / Cert No:</strong> {ext.certificate_no || ext.vc_no}
                                      </span>
                                    )}
                                    {(ext.decision_no || ext.bearing_no || ext.application_no || ext.gr_no) && (
                                      <span className="bg-white px-2 py-0.5 rounded border border-slate-200 font-mono text-slate-800">
                                        <strong>Decision / Bearing:</strong> {ext.decision_no || ext.bearing_no || ext.application_no || ext.gr_no}
                                      </span>
                                    )}
                                    {(ext.issued_date || ext.dated || ext.received_date || ext.dob) && (
                                      <span className="bg-white px-2 py-0.5 rounded border border-slate-200 font-medium text-slate-800">
                                        <strong>Dated:</strong> {ext.issued_date || ext.dated || ext.received_date || ext.dob}
                                      </span>
                                    )}
                                    {ext.validity_decision && (
                                      <span className={`px-2 py-0.5 rounded font-bold ${
                                        ext.validity_decision === 'VALID'
                                          ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                                          : 'bg-amber-100 text-amber-800 border border-amber-300'
                                      }`}>
                                        <strong>Status:</strong> {ext.validity_decision}
                                      </span>
                                    )}
                                    {(ext.caste_claim || ext.community_mother_tongue || ext.mother_tongue) && (
                                      <span className="bg-white px-2 py-0.5 rounded border border-slate-200 font-medium text-slate-800">
                                        <strong>Claim:</strong> {ext.caste_claim || ext.community_mother_tongue || ext.mother_tongue}
                                      </span>
                                    )}
                                    {(ext.committee || ext.district) && (
                                      <span className="bg-white px-2 py-0.5 rounded border border-slate-200 text-slate-600">
                                        <strong>District:</strong> {ext.district || ext.committee}
                                      </span>
                                    )}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default InstituteHierarchy;
