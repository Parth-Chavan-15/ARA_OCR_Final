import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';
import StatusBadge from '../components/StatusBadge';
import { AlertTriangle, FileSearch, ArrowRight, ShieldAlert, RefreshCw, Eye } from 'lucide-react';

const ReviewQueue = () => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchReviewDocs();
  }, []);

  const fetchReviewDocs = async () => {
    setLoading(true);
    try {
      const data = await apiService.getReviewDocuments();
      setDocuments(Array.isArray(data) ? data : []);
    } catch (error) {
      console.error('Failed to load review documents', error);
    } finally {
      setLoading(false);
    }
  };

  const getReason = (status) => {
    switch (status) {
      case 'LOW_CONFIDENCE':
        return 'Classification confidence fell below the acceptable threshold (0.60). Manual verification required.';
      case 'UNKNOWN':
      case 'OUT_OF_SCOPE':
      case 'UNKNOWN_OUT_OF_SCOPE':
        return 'Document did not match known ARA broad classes. Non-Creamy Layer (NCL) or invalid admission certificate.';
      case 'ERROR':
        return 'An error occurred during the OCR or classification pipeline.';
      default:
        return 'Flagged for scrutiny officer audit.';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-[10px] font-bold uppercase tracking-wider text-rose-900 bg-rose-50 px-2 py-0.5 rounded border border-rose-200">
              Scrutiny Officer Review Queue
            </span>
            <span className="text-xs text-slate-500">
              Manual Verification & Audit Desk
            </span>
          </div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight mt-1 flex items-center gap-2">
            <ShieldAlert className="text-amber-600" size={22} />
            Documents Requiring Officer Audit
          </h2>
          <p className="text-xs text-slate-500">
            Cases with low confidence scores, out-of-scope declarations, or anomalous certificate formats.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <span className="bg-amber-100 text-amber-900 px-3.5 py-1.5 rounded-lg text-xs font-bold border border-amber-300">
            {documents.length} Requiring Review
          </span>
          <button
            onClick={fetchReviewDocs}
            disabled={loading}
            className="flex items-center space-x-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-2 rounded-lg text-xs font-semibold transition border border-slate-300 cursor-pointer"
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
            <span>Refresh Queue</span>
          </button>
        </div>
      </div>

      {loading ? (
        <div className="py-16 text-center text-slate-500 text-xs">
          <RefreshCw size={24} className="animate-spin mx-auto text-blue-600 mb-2" />
          Checking scrutiny audit desk...
        </div>
      ) : documents.length === 0 ? (
        <div className="bg-white rounded-xl shadow-xs border border-slate-200 p-12 text-center space-y-2">
          <FileSearch size={40} className="mx-auto text-slate-300" />
          <h3 className="text-base font-bold text-slate-900">Scrutiny Audit Queue is Clear</h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            All submitted candidate documents in the vault have been categorized with high confidence and zero flagged exceptions.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {documents.map((doc) => {
            const docId = doc.document_id || doc.id;
            return (
              <div
                key={docId}
                className="bg-white rounded-xl shadow-xs border border-amber-200 hover:border-amber-300 transition flex flex-col justify-between"
              >
                <div className="p-4 border-b border-slate-100 bg-slate-50 flex justify-between items-start gap-2">
                  <div className="max-w-[200px] truncate">
                    <h4 className="font-bold text-slate-900 text-xs truncate" title={doc.filename}>
                      {doc.filename}
                    </h4>
                    <p className="text-[10px] text-slate-500 mt-0.5">{doc.candidate_name || 'Candidate Document'}</p>
                  </div>
                  <StatusBadge status={doc.status} />
                </div>

                <div className="p-4 flex-1 space-y-3">
                  <div className="text-xs text-slate-700 bg-amber-50/70 p-3 rounded-lg border border-amber-200">
                    <span className="font-bold block mb-0.5 text-amber-900 text-[11px]">Audit Reason:</span>
                    <p className="text-[11px] text-slate-700 leading-snug">{getReason(doc.status)}</p>
                  </div>

                  {doc.document_type && (
                    <div className="text-xs flex justify-between items-center">
                      <span className="text-slate-500">Detected Class:</span>
                      <StatusBadge status={doc.document_type} />
                    </div>
                  )}
                </div>

                <div className="p-3 border-t border-slate-100 bg-slate-50">
                  <button
                    onClick={() => navigate(/result/)}
                    className="w-full flex items-center justify-center space-x-1.5 py-2 bg-white border border-slate-300 rounded-lg text-xs font-bold text-slate-800 hover:bg-slate-100 transition cursor-pointer"
                  >
                    <Eye size={13} />
                    <span>Audit & Verify</span>
                    <ArrowRight size={13} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default ReviewQueue;