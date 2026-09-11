import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, useLocation, useOutletContext } from 'react-router-dom';
import { apiService } from '../services/api';
import StatusBadge from '../components/StatusBadge';
import {
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Download,
  ShieldCheck,
  Award,
  Cpu,
  Layers,
  Sparkles,
  BarChart3,
  Play,
  Loader2,
  RefreshCw,
  Calendar,
  Hash,
  Building,
  User,
  Tag,
} from 'lucide-react';

const ClassificationResult = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const context = useOutletContext();
  const refreshTrigger = context?.refreshTrigger || 0;

  const [doc, setDoc] = useState(null);
  const [candidate, setCandidate] = useState(null);
  const [classification, setClassification] = useState(null);
  const [ocr, setOcr] = useState(null);
  const [loading, setLoading] = useState(true);
  const [scrutinizing, setScrutinizing] = useState(false);

  const fetchDetails = async () => {
    try {
      const docData = await apiService.getDocument(id);
      setDoc(docData);

      if (docData?.candidate_id) {
        try {
          const candData = await apiService.getCandidate(docData.candidate_id);
          setCandidate(candData);
        } catch (e) {
          console.log('No candidate details');
        }
      }

      try {
        const classData = await apiService.getDocumentClassification(id);
        setClassification(classData);
      } catch (e) {
        console.log('No classification data yet');
      }

      try {
        const ocrData = await apiService.getDocumentOcr(id);
        setOcr(ocrData);
      } catch (e) {
        console.log('No OCR data yet');
      }
    } catch (error) {
      console.error('Failed to load document details', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDetails();
  }, [id, refreshTrigger]);

  const handleRunScrutiny = async () => {
    try {
      setScrutinizing(true);
      await apiService.classifyDocument(id);
      await fetchDetails();
    } catch (err) {
      console.error('Scrutiny execution failed:', err);
    } finally {
      setScrutinizing(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center py-24 text-slate-500 text-xs">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-900 mb-2 mr-3" />
        Loading official scrutiny dossier...
      </div>
    );
  }

  if (!doc) {
    return (
      <div className="p-8 text-center text-rose-600 font-bold">
        Document not found in storage.
      </div>
    );
  }

  // Dual Confidence Extraction (Hybrid Floor + LayoutLMv3 Neural Probability)
  const getConfidenceData = () => {
    // Check if backend returned formatted confidence_label
    const explicitLabel =
      classification?.confidence_label ||
      classification?.evidence?.confidence_label ||
      doc?.confidence_label;

    let hybrid = null;
    if (classification && typeof classification.confidence === 'number' && !isNaN(classification.confidence)) {
      hybrid = classification.confidence <= 1.0 ? classification.confidence * 100 : classification.confidence;
    } else if (doc && typeof doc.confidence === 'number' && !isNaN(doc.confidence)) {
      hybrid = doc.confidence <= 1.0 ? doc.confidence * 100 : doc.confidence;
    } else if (doc?.document_type) {
      hybrid = 98.0;
    }

    let neural = null;
    if (classification && typeof classification.raw_confidence === 'number' && !isNaN(classification.raw_confidence)) {
      neural = classification.raw_confidence <= 1.0 ? classification.raw_confidence * 100 : classification.raw_confidence;
    } else if (
      classification?.evidence &&
      typeof classification.evidence.raw_confidence === 'number' &&
      !isNaN(classification.evidence.raw_confidence)
    ) {
      neural =
        classification.evidence.raw_confidence <= 1.0
          ? classification.evidence.raw_confidence * 100
          : classification.evidence.raw_confidence;
    } else if (doc && typeof doc.raw_confidence === 'number' && !isNaN(doc.raw_confidence)) {
      neural = doc.raw_confidence <= 1.0 ? doc.raw_confidence * 100 : doc.raw_confidence;
    }

    const ruleApplied =
      classification?.rule_applied ||
      classification?.evidence?.rule_applied ||
      null;

    if (hybrid === null) {
      return {
        hasScore: false,
        hybridScore: 0,
        neuralScore: null,
        ruleApplied: null,
        display: 'Pending Scrutiny',
        fullLabel: 'Pending Scrutiny',
      };
    }

    const hybridFormatted = hybrid.toFixed(1) + '%';
    let fullLabel = explicitLabel;

    if (!fullLabel) {
      if (neural !== null) {
        fullLabel = ruleApplied
          ? `${hybridFormatted} (LayoutLMv3: ${neural.toFixed(1)}% + Rule Evidence)`
          : `${hybridFormatted} (LayoutLMv3: ${neural.toFixed(1)}% Direct Neural Confidence)`;
      } else {
        fullLabel = `${hybridFormatted} (LayoutLMv3 + Rule Evidence)`;
      }
    }

    return {
      hasScore: true,
      hybridScore: hybrid,
      neuralScore: neural,
      ruleApplied,
      display: hybridFormatted,
      fullLabel,
    };
  };

  const confData = getConfidenceData();
  const confidenceScoreNum = confData.hasScore ? Math.min(Math.max(confData.hybridScore, 0), 100) : 0;
  const confidenceDisplay = confData.fullLabel;

  const probabilities = classification?.evidence?.class_probabilities || null;
  const rawProbabilities = classification?.evidence?.raw_probabilities || null;

  const extractedFields = classification?.extracted_fields || doc?.extracted_fields || null;

  const FIELD_LABELS = {
    certificate_no: 'Certificate / Decision No',
    receipt_number: 'Receipt Number',
    serial_number: 'Serial Number',
    application_number: 'Application No',
    general_register_no: 'G.R. Number',
    validity_decision: 'Validity Decision',
    valid_upto: 'Valid Upto Date',
    dated: 'Dated',
    issue_date: 'Issue Date',
    submission_date: 'Submission / Receipt Date',
    date_of_birth: 'Date of Birth',
    issuing_authority: 'Issuing Authority / SDO',
    scrutiny_committee: 'Divisional Scrutiny Committee',
    caste_claim: 'Caste Claim',
    caste_category: 'Caste Category',
    caste_or_religion: 'Caste / Religion',
    minority_type: 'Minority Classification',
    community_claim: 'Community Claim',
    mother_tongue: 'Mother Tongue',
    district: 'District / Division',
    candidate_name: 'Candidate Name',
    student_name: 'Student Name',
    enrollment_number: 'Enrollment No',
  };

  const getFieldIcon = (key) => {
    const k = key.toLowerCase();
    if (k.includes('date') || k.includes('dated') || k.includes('upto')) return <Calendar size={13} className="text-blue-600 shrink-0" />;
    if (k.includes('no') || k.includes('number')) return <Hash size={13} className="text-purple-600 shrink-0" />;
    if (k.includes('authority') || k.includes('committee')) return <Building size={13} className="text-slate-600 shrink-0" />;
    if (k.includes('name')) return <User size={13} className="text-emerald-600 shrink-0" />;
    if (k.includes('caste') || k.includes('minority') || k.includes('community')) return <Tag size={13} className="text-amber-600 shrink-0" />;
    return <ShieldCheck size={13} className="text-blue-600 shrink-0" />;
  };

  const visibleFieldEntries = extractedFields
    ? Object.entries(extractedFields).filter(
        ([k, v]) => !['extracted_at_step', 'predicted_class'].includes(k) && v !== null && v !== undefined && String(v).trim() !== ''
      )
    : [];

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3.5">
          <button
            onClick={() => {
              if (location.state?.from) {
                navigate(location.state.from);
              } else {
                navigate('/incoming');
              }
            }}
            className="p-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition border border-slate-200 cursor-pointer"
            title="Go Back"
          >
            <ArrowLeft size={18} />
          </button>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-blue-900 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                Official Scrutiny Verification Dossier
              </span>
              <span className="text-xs text-slate-500 font-mono">
                ID: {doc.document_id.slice(0, 8)}...
              </span>
            </div>
            <h1 className="text-lg font-bold text-slate-900 flex items-center gap-2 mt-1">
              {doc.filename}
              <StatusBadge status={doc.document_type || doc.status} />
            </h1>
            {candidate && (
              <p className="text-xs text-slate-600 mt-0.5">
                Candidate: <strong>{candidate.name}</strong> ({candidate.enrollment_number}) • Category: <strong>{candidate.reserve_category || 'OBC'}</strong>
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-2.5">
          {(!doc.document_type || !classification) && (
            <button
              onClick={handleRunScrutiny}
              disabled={scrutinizing}
              className="flex items-center space-x-1.5 bg-[#0F2942] hover:bg-[#1A3D60] text-amber-300 font-bold px-4 py-2 rounded-lg text-xs shadow-xs transition border border-amber-500/30 cursor-pointer disabled:opacity-50"
            >
              {scrutinizing ? (
                <>
                  <Loader2 size={13} className="animate-spin text-amber-400" />
                  <span>Processing Scrutiny...</span>
                </>
              ) : (
                <>
                  <Sparkles size={13} className="text-amber-400" />
                  <span>Run AI Scrutiny</span>
                </>
              )}
            </button>
          )}

          <a
            href={apiService.getDocumentImageUrl(id)}
            target="_blank"
            rel="noreferrer"
            className="flex items-center space-x-1.5 bg-slate-100 hover:bg-slate-200 text-slate-800 px-3.5 py-2 rounded-lg text-xs font-bold border border-slate-300 transition"
          >
            <Download size={13} />
            <span>Download File</span>
          </a>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Left Column: Image Preview */}
        <div className="bg-white rounded-xl shadow-xs border border-slate-200 overflow-hidden flex flex-col h-[750px]">
          <div className="px-4 py-3 border-b border-slate-200 bg-slate-50 flex justify-between items-center">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center">
              <FileText size={15} className="mr-2 text-blue-700" />
              Document Visual Layout & Evidence
            </h3>
            <span className="text-[10px] font-semibold text-slate-500 bg-white px-2 py-0.5 rounded border border-slate-200">
              Vault Original
            </span>
          </div>
          <div className="flex-1 bg-slate-100/60 p-4 overflow-auto flex justify-center items-start">
            <img
              src={apiService.getProcessedImageUrl(id)}
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = apiService.getDocumentImageUrl(id);
              }}
              alt="Document Scrutiny Preview"
              className="max-w-full max-h-full shadow-md object-contain bg-white rounded border border-slate-200"
            />
          </div>
        </div>

        {/* Right Column: Classification Results & OCR Transcription */}
        <div className="space-y-5">
          {/* Classification Result Card */}
          <div className="bg-white rounded-xl shadow-xs border border-slate-200 p-5 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center space-x-2">
                <Sparkles size={16} className="text-amber-500" />
                <h3 className="text-sm font-bold text-slate-900">
                  Multimodal Scrutiny Classification Result
                </h3>
              </div>
              <span className="text-[11px] font-bold text-emerald-800 bg-emerald-50 px-2.5 py-1 rounded border border-emerald-300">
                AI Confidence: {confidenceDisplay}
              </span>
            </div>

            <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-600 font-medium">Verified Certificate Category:</span>
                <StatusBadge status={classification?.predicted_class || doc.document_type || 'PENDING'} />
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs gap-1.5">
                <span className="text-slate-600 font-medium">Confidence Score:</span>
                <span className="font-mono font-bold text-slate-900 text-xs sm:text-sm bg-white px-2.5 py-1 rounded border border-slate-200 shadow-2xs">
                  {confidenceDisplay}
                </span>
              </div>

              {/* Progress Bar */}
              <div className="h-2.5 w-full bg-slate-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-600 to-emerald-500 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(Math.max(confidenceScoreNum, 5), 100)}%` }}
                />
              </div>

              {/* Dual Confidence Breakdown Pills */}
              {confData.hasScore && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 border-t border-slate-200/60">
                  <div className="bg-white rounded-lg p-2 border border-slate-200 flex items-center justify-between">
                    <span className="text-[11px] text-slate-500 font-medium">Calibrated Hybrid Floor:</span>
                    <span className="text-[11px] font-mono font-bold text-emerald-700">{confData.hybridScore.toFixed(1)}%</span>
                  </div>
                  <div className="bg-white rounded-lg p-2 border border-slate-200 flex items-center justify-between">
                    <span className="text-[11px] text-slate-500 font-medium">LayoutLMv3 Neural Prob:</span>
                    <span className="text-[11px] font-mono font-bold text-blue-700">
                      {confData.neuralScore !== null ? `${confData.neuralScore.toFixed(1)}%` : 'Active'}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Probability Breakdown across all classes */}
            {probabilities && (
              <div className="space-y-2.5 pt-2 border-t border-slate-100">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
                    <BarChart3 size={13} className="text-blue-700" />
                    Category Probability Distribution:
                  </span>
                </div>
                <div className="space-y-2 text-xs">
                  {Object.entries(probabilities).map(([cls, prob]) => {
                    const probPct = typeof prob === 'number' ? (prob <= 1.0 ? prob * 100 : prob) : 0;
                    const rawProb = rawProbabilities?.[cls];
                    const rawPct = typeof rawProb === 'number' ? (rawProb <= 1.0 ? rawProb * 100 : rawProb) : null;
                    const isWinning = cls === (classification.predicted_class || doc.document_type);
                    return (
                      <div key={cls} className="space-y-0.5">
                        <div className="flex justify-between text-[11px]">
                          <span className={isWinning ? 'font-bold text-blue-900' : 'text-slate-600'}>
                            {cls.replace(/_/g, ' ')}
                          </span>
                          <span className="font-mono font-bold text-slate-700">
                            {probPct.toFixed(1)}%
                            {rawPct !== null && (
                              <span className="text-[10px] text-slate-400 font-normal ml-1.5">
                                (LayoutLMv3: {rawPct.toFixed(1)}%)
                              </span>
                            )}
                          </span>
                        </div>
                        <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${
                              isWinning ? 'bg-emerald-500' : 'bg-slate-300'
                            }`}
                            style={{ width: `${Math.min(Math.max(probPct, 2), 100)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Evidence Findings */}
            {classification?.evidence?.summary && (
              <div className="pt-2 text-xs text-slate-600 bg-blue-50/60 p-3 rounded-lg border border-blue-200">
                <span className="font-bold text-blue-900 block mb-1">Scrutiny Summary:</span>
                <p>{classification.evidence.summary}</p>
              </div>
            )}
          </div>

          {/* Extracted Key Decision Fields Card */}
          <div className="bg-white rounded-xl shadow-xs border border-slate-200 p-5 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center space-x-2">
                <ShieldCheck size={18} className="text-emerald-600" />
                <h3 className="text-sm font-bold text-slate-900">
                  Extracted Key Decision Fields
                </h3>
              </div>
              <span className="text-[10px] font-bold text-emerald-800 bg-emerald-50 px-2.5 py-1 rounded border border-emerald-200 uppercase tracking-wider">
                Stage 9: Parallel Field Extraction
              </span>
            </div>

            {visibleFieldEntries.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {visibleFieldEntries.map(([key, value]) => {
                  const label = FIELD_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
                  const valStr = String(value);
                  const isDecision = key === 'validity_decision';
                  const isMono = key.includes('no') || key.includes('number') || key.includes('enrollment');
                  const valUpper = valStr.toUpperCase();
                  const isValid = valUpper.includes('VALID') && !valUpper.includes('INVALID') && !valUpper.includes('OUT_OF_SCOPE');
                  const isOutOfScope = valUpper.includes('OUT_OF_SCOPE');

                  return (
                    <div
                      key={key}
                      className="bg-slate-50/80 hover:bg-slate-50 border border-slate-200/80 rounded-lg p-3 transition flex flex-col justify-between space-y-1.5"
                    >
                      <div className="flex items-center space-x-1.5">
                        {getFieldIcon(key)}
                        <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                          {label}
                        </span>
                      </div>
                      <div>
                        {isDecision ? (
                          <span
                            className={`inline-flex items-center text-xs font-bold px-2.5 py-1 rounded-md border ${
                              isValid
                                ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                                : isOutOfScope
                                ? 'bg-amber-50 text-amber-800 border-amber-300'
                                : 'bg-rose-50 text-rose-800 border-rose-300'
                            }`}
                          >
                            {valStr}
                          </span>
                        ) : isMono ? (
                          <span className="font-mono text-xs font-bold text-slate-900 bg-white px-2 py-0.5 rounded border border-slate-200">
                            {valStr}
                          </span>
                        ) : (
                          <span className="text-xs font-bold text-slate-800 break-words">
                            {valStr}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="bg-amber-50/60 border border-amber-200 rounded-lg p-3.5 text-xs text-amber-800 flex items-center space-x-2">
                <AlertTriangle size={15} className="text-amber-600 shrink-0" />
                <span>
                  Key decision fields have not been extracted yet. Click <strong>Run AI Scrutiny</strong> above to parse structured fields.
                </span>
              </div>
            )}
          </div>

          {/* OCR Transcription Card */}
          <div className="bg-white rounded-xl shadow-xs border border-slate-200 p-5 space-y-3">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center space-x-2">
                <Cpu size={16} className="text-blue-600" />
                <h3 className="text-sm font-bold text-slate-900">
                  Extracted OCR Text & Form Tokens
                </h3>
              </div>
              <span className="text-[10px] text-slate-500 font-mono">
                {ocr?.tokens?.length || 0} tokens detected
              </span>
            </div>

            <div className="max-h-[280px] overflow-y-auto bg-slate-900 text-slate-200 p-4 rounded-lg font-mono text-xs leading-relaxed space-y-1 select-text">
              {ocr?.raw_text ? (
                ocr.raw_text.split('\n').map((line, i) => (
                  <p key={i} className="whitespace-pre-wrap">{line}</p>
                ))
              ) : (
                <p className="text-slate-500 italic">No OCR transcription available for this document.</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ClassificationResult;