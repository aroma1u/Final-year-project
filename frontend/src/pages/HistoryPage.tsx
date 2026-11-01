import React from 'react';
import { 
  History as HistoryIcon, 
  Trash2, 
  ShieldAlert, 
  ShieldCheck, 
  Clock, 
  Layers, 
  Info,
  ArrowRight
} from 'lucide-react';
import { PredictionResult, TabType } from '../types';

interface HistoryPageProps {
  history: PredictionResult[];
  clearHistory: () => void;
  removeItem: (id: string) => void;
  setActiveTab: (tab: TabType) => void;
}

export const HistoryPage: React.FC<HistoryPageProps> = ({
  history,
  clearHistory,
  removeItem,
  setActiveTab,
}) => {
  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white p-6 rounded-3xl border border-slate-200/80 shadow-xs">
        <div>
          <h2 className="text-xl font-bold text-slate-900 flex items-center space-x-2">
            <HistoryIcon className="w-5 h-5 text-indigo-600" />
            <span>Analysis Session History</span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Recorded client-side predictions from your current browser session ({history.length} {history.length === 1 ? 'sample' : 'samples'})
          </p>
        </div>

        {history.length > 0 && (
          <button
            onClick={() => {
              if (window.confirm('Are you sure you want to clear your local session history?')) {
                clearHistory();
              }
            }}
            className="inline-flex items-center space-x-1.5 px-4 py-2 rounded-xl text-xs font-semibold text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 transition-colors cursor-pointer self-start sm:self-auto"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear History</span>
          </button>
        )}
      </div>

      {/* Local Storage Privacy Notice */}
      <div className="flex items-start space-x-3 p-4 rounded-2xl bg-slate-100/70 border border-slate-200/60 text-slate-600 text-xs">
        <Info className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
        <p>
          History records are stored exclusively in your browser's local storage. No user images or meme captions are stored permanently on the inference backend server.
        </p>
      </div>

      {/* History Records List */}
      {history.length === 0 ? (
        <div className="bg-white rounded-3xl p-12 text-center border border-slate-200/80 shadow-xs space-y-3">
          <div className="w-12 h-12 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto">
            <HistoryIcon className="w-6 h-6" />
          </div>
          <h3 className="font-bold text-slate-900 text-base">No Analysis Records Yet</h3>
          <p className="text-xs text-slate-500 max-w-sm mx-auto leading-relaxed">
            Submit memes and text in the Moderation Studio to evaluate predictions and review their cross-modal attention weights here.
          </p>
          <button
            onClick={() => setActiveTab('moderation')}
            className="inline-flex items-center space-x-1.5 px-5 py-2.5 rounded-xl bg-indigo-600 text-white font-semibold text-xs hover:bg-indigo-500 transition-all shadow-sm cursor-pointer mt-2"
          >
            <span>Open Moderation Studio</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {history.map((item, idx) => (
            <div
              key={item.id || idx}
              className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs hover:border-slate-300 transition-all flex flex-col md:flex-row md:items-center justify-between gap-4"
            >
              {/* Left: Thumbnail & Text snippet */}
              <div className="flex items-start space-x-4 flex-1">
                {item.image_preview_url ? (
                  <img
                    src={item.image_preview_url}
                    alt="Sample thumbnail"
                    className="w-16 h-16 rounded-xl object-cover bg-slate-100 shrink-0 border border-slate-200"
                  />
                ) : (
                  <div className="w-16 h-16 rounded-xl bg-slate-100 text-slate-400 flex items-center justify-center shrink-0">
                    <Layers className="w-6 h-6" />
                  </div>
                )}
                <div className="space-y-1">
                  <div className="flex items-center space-x-2">
                    {item.predicted_class === 1 ? (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-rose-100 text-rose-800">
                        <ShieldAlert className="w-3 h-3 mr-1" /> Hateful
                      </span>
                    ) : (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-100 text-emerald-800">
                        <ShieldCheck className="w-3 h-3 mr-1" /> Non-Hateful
                      </span>
                    )}
                    <span className="text-xs font-semibold text-slate-700">
                      {item.confidence.toFixed(1)}% Confidence
                    </span>
                  </div>
                  {item.input_text ? (
                    <p className="text-xs text-slate-600 line-clamp-2 italic">
                      "{item.input_text}"
                    </p>
                  ) : (
                    <p className="text-xs text-slate-400 italic">
                      Image-only analysis (no text caption)
                    </p>
                  )}
                  <div className="flex items-center space-x-3 text-[10px] text-slate-400">
                    <span className="flex items-center space-x-1">
                      <Clock className="w-3 h-3" />
                      <span>{new Date(item.timestamp).toLocaleString()}</span>
                    </span>
                    <span>Threshold: τ* = {item.calibrated_threshold}</span>
                  </div>
                </div>
              </div>

              {/* Right: Modality Attention Breakdown & Controls */}
              <div className="flex items-center space-x-4 md:border-l md:border-slate-100 md:pl-6 shrink-0">
                <div className="space-y-1.5 w-36 text-[11px]">
                  <div>
                    <div className="flex justify-between text-slate-500 mb-0.5">
                      <span>Visual Attn</span>
                      <span className="font-mono font-semibold text-indigo-600">{item.attention.visual.toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                      <div className="h-full bg-indigo-600 rounded-full" style={{ width: `${item.attention.visual}%` }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-slate-500 mb-0.5">
                      <span>Text Attn</span>
                      <span className="font-mono font-semibold text-teal-600">{item.attention.textual.toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                      <div className="h-full bg-teal-600 rounded-full" style={{ width: `${item.attention.textual}%` }} />
                    </div>
                  </div>
                </div>

                {item.id && (
                  <button
                    onClick={() => removeItem(item.id!)}
                    className="p-2 text-slate-300 hover:text-rose-600 hover:bg-rose-50 rounded-lg transition-colors"
                    title="Delete record"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
