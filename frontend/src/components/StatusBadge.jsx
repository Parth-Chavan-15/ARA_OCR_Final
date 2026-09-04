import React from 'react';
import { CheckCircle2, Clock, AlertTriangle, HelpCircle, XCircle } from 'lucide-react';

const StatusBadge = ({ status }) => {
  const normalized = (status || '').toUpperCase();

  switch (normalized) {
    case 'CASTE_CERTIFICATE':
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold bg-emerald-50 text-emerald-800 border border-emerald-300">
          <CheckCircle2 size={13} className="text-emerald-600" />
          Caste Certificate (जातीचे दाखला)
        </span>
      );

    case 'CASTE_VALIDITY_CERTIFICATE':
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold bg-blue-50 text-blue-800 border border-blue-300">
          <CheckCircle2 size={13} className="text-blue-600" />
          Caste Validity Certificate (जात वैधता)
        </span>
      );

    case 'CASTE_VALIDITY_RECEIPT':
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold bg-amber-50 text-amber-900 border border-amber-300">
          <Clock size={13} className="text-amber-600" />
          Caste Validity Receipt (पावती)
        </span>
      );

    case 'PROFORMA_O':
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold bg-purple-50 text-purple-800 border border-purple-300">
          <CheckCircle2 size={13} className="text-purple-600" />
          Proforma-O (प्रपत्र-ओ)
        </span>
      );

    case 'LEAVING_CERTIFICATE':
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold bg-teal-50 text-teal-800 border border-teal-300">
          <CheckCircle2 size={13} className="text-teal-600" />
          Leaving Certificate (शाळा सोडल्याचा दाखला)
        </span>
      );

    case 'UNKNOWN_OUT_OF_SCOPE':
    case 'OUT_OF_SCOPE':
      return (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold bg-rose-50 text-rose-800 border border-rose-300">
          <XCircle size={13} className="text-rose-600" />
          Non-Creamy Layer / Out-of-Scope
        </span>
      );

    case 'COMPLETED':
    case 'CLASSIFIED':
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-100 text-emerald-800">
          <CheckCircle2 size={12} className="text-emerald-700" />
          Classified
        </span>
      );

    case 'PENDING':
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-100 text-slate-700 border border-slate-300">
          <Clock size={12} className="text-slate-500" />
          Pending Scrutiny
        </span>
      );

    case 'PROCESSING':
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-semibold bg-blue-100 text-blue-800 animate-pulse">
          <Clock size={12} className="text-blue-700" />
          Processing
        </span>
      );

    case 'NEEDS_REVIEW':
    case 'FLAGGED':
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-100 text-amber-900 border border-amber-300">
          <AlertTriangle size={12} className="text-amber-700" />
          Officer Review
        </span>
      );

    default:
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-medium bg-gray-100 text-gray-700">
          <HelpCircle size={12} className="text-gray-500" />
          {status || 'Unknown'}
        </span>
      );
  }
};

export default StatusBadge;