import React from 'react';
import { 
  ArrowRight, 
  Upload, 
  Cpu, 
  ShieldAlert, 
  CheckCircle, 
  Layers, 
  Eye, 
  AlertTriangle,
  Zap,
  ScanText,
  Info
} from 'lucide-react';
import type { TabType, HealthStatus } from '../types';
import { projectConfig } from '../config/projectConfig';

interface OverviewPageProps {
  setActiveTab: (tab: TabType) => void;
  health: HealthStatus;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({ setActiveTab, health }) => {
  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-indigo-950 via-slate-900 to-indigo-900 text-white p-8 sm:p-12 shadow-xl border border-indigo-900/40">
        {/* Subtle background glow */}
        <div className="absolute top-0 right-0 -mt-12 -mr-12 w-96 h-96 rounded-full bg-indigo-500/10 blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 -mb-16 w-80 h-80 rounded-full bg-teal-500/10 blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-2xl">
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight leading-tight mb-4">
            Understand the Image. <br className="hidden sm:block" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-300 via-cyan-200 to-indigo-200">
              Understand the Context.
            </span>
          </h1>

          <p className="text-slate-300 text-base sm:text-lg leading-relaxed mb-8 font-light">
            Analyze complex internet memes and social media posts where neither text nor imagery is toxic in isolation, but their multimodal juxtaposition conveys hate speech.
          </p>

          <div className="flex flex-wrap items-center gap-4">
            <button
              onClick={() => setActiveTab('moderation')}
              className="inline-flex items-center space-x-2 px-6 py-3.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition-all shadow-lg shadow-indigo-600/30 hover:shadow-indigo-600/50 cursor-pointer"
            >
              <span>Analyze Content</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => setActiveTab('about')}
              className="inline-flex items-center space-x-2 px-6 py-3.5 rounded-xl bg-white/10 hover:bg-white/15 text-white font-medium text-sm transition-all border border-white/10 cursor-pointer"
            >
              <Info className="w-4 h-4" />
              <span>About This Website</span>
            </button>
          </div>
        </div>

        {/* Visual Pipeline Abstract Icon Pill */}
        <div className="hidden lg:flex absolute right-12 bottom-12 items-center space-x-3 bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-4 shadow-2xl">
          <div className="flex flex-col items-center p-2 text-center">
            <span className="w-8 h-8 rounded-lg bg-indigo-500/20 text-indigo-300 flex items-center justify-center font-mono text-xs font-bold mb-1">
              IMG
            </span>
            <span className="text-[10px] text-slate-400">ResNet-50</span>
          </div>
          <span className="text-slate-500 font-bold">+</span>
          <div className="flex flex-col items-center p-2 text-center">
            <span className="w-8 h-8 rounded-lg bg-teal-500/20 text-teal-300 flex items-center justify-center font-mono text-xs font-bold mb-1">
              OCR
            </span>
            <span className="text-[10px] text-slate-400">EasyOCR</span>
          </div>
          <span className="text-slate-500 font-bold">+</span>
          <div className="flex flex-col items-center p-2 text-center">
            <span className="w-8 h-8 rounded-lg bg-teal-500/20 text-teal-300 flex items-center justify-center font-mono text-xs font-bold mb-1">
              TXT
            </span>
            <span className="text-[10px] text-slate-400">BERT</span>
          </div>
          <span className="text-slate-500 font-bold">→</span>
          <div className="flex flex-col items-center p-2 text-center">
            <span className="w-8 h-8 rounded-lg bg-amber-500/20 text-amber-300 flex items-center justify-center font-mono text-xs font-bold mb-1">
              BAM
            </span>
            <span className="text-[10px] text-slate-400">Hadamard</span>
          </div>
          <span className="text-slate-500 font-bold">→</span>
          <div className="flex flex-col items-center p-2 text-center">
            <span className="w-8 h-8 rounded-lg bg-emerald-500/20 text-emerald-300 flex items-center justify-center font-mono text-xs font-bold mb-1">
              PRED
            </span>
            <span className="text-[10px] text-slate-400">τ* = {projectConfig.calibratedThreshold}</span>
          </div>
        </div>
      </section>

      {/* Summary Metrics Cards */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Multimodal Model</span>
            <Layers className="w-4 h-4 text-indigo-600" />
          </div>
          <div className="text-lg font-bold text-slate-900">{projectConfig.modelArchitecture}</div>
          <p className="text-xs text-slate-500 mt-1">Bi-directional Attention (BAM)</p>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Input Modalities</span>
            <Eye className="w-4 h-4 text-teal-600" />
          </div>
          <div className="text-lg font-bold text-slate-900">Image & Optional Text</div>
          <p className="text-xs text-slate-500 mt-1">Automated OCR or Manual Caption</p>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Classification</span>
            <ShieldAlert className="w-4 h-4 text-amber-600" />
          </div>
          <div className="text-lg font-bold text-slate-900">Binary Detection</div>
          <p className="text-xs text-slate-500 mt-1">Hateful vs. Non-Hateful</p>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider">Model Status</span>
            <Cpu className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-lg font-bold text-slate-900 flex items-center space-x-2">
            <span>{health.model_loaded ? 'Available' : 'Offline'}</span>
            <span className={`w-2 h-2 rounded-full ${health.model_loaded ? 'bg-emerald-500' : 'bg-rose-500'}`} />
          </div>
          <p className="text-xs text-slate-500 mt-1">{health.device.toUpperCase()} PyTorch Execution</p>
        </div>
      </section>

      {/* How It Works — 4 Step Interactive Workflow */}
      <section className="bg-white rounded-3xl p-8 border border-slate-200/80 shadow-xs">
        <div className="max-w-xl mb-8">
          <h2 className="text-2xl font-bold text-slate-900 tracking-tight">How MemeGuard AI Works</h2>
          <p className="text-sm text-slate-500 mt-1 font-normal">
            A multimodal workflow designed to identify subtle toxicity, sarcasm, and cross-modal hate speech.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 relative">
          <div className="flex flex-col items-start p-5 rounded-2xl bg-slate-50/70 border border-slate-100 hover:border-indigo-200 transition-colors">
            <div className="w-10 h-10 rounded-xl bg-indigo-100 text-indigo-700 flex items-center justify-center mb-4 font-bold text-sm">
              <Upload className="w-5 h-5" />
            </div>
            <span className="text-xs font-semibold text-indigo-600 uppercase tracking-wider mb-1">Step 1</span>
            <h3 className="font-semibold text-slate-900 text-base mb-1.5">Upload Meme</h3>
            <p className="text-xs text-slate-500 leading-relaxed font-normal">
              Upload the meme graphic in PNG, JPG, or WEBP format for ResNet-50 visual representation.
            </p>
          </div>

          <div className="flex flex-col items-start p-5 rounded-2xl bg-slate-50/70 border border-slate-100 hover:border-indigo-200 transition-colors">
            <div className="w-10 h-10 rounded-xl bg-teal-100 text-teal-700 flex items-center justify-center mb-4 font-bold text-sm">
              <ScanText className="w-5 h-5" />
            </div>
            <span className="text-xs font-semibold text-teal-600 uppercase tracking-wider mb-1">Step 2</span>
            <h3 className="font-semibold text-slate-900 text-base mb-1.5">OCR Text Extraction</h3>
            <p className="text-xs text-slate-500 leading-relaxed font-normal">
              EasyOCR automatically extracts embedded meme text with human-in-the-loop editing for optional text input.
            </p>
          </div>

          <div className="flex flex-col items-start p-5 rounded-2xl bg-slate-50/70 border border-slate-100 hover:border-indigo-200 transition-colors">
            <div className="w-10 h-10 rounded-xl bg-amber-100 text-amber-700 flex items-center justify-center mb-4 font-bold text-sm">
              <Zap className="w-5 h-5" />
            </div>
            <span className="text-xs font-semibold text-amber-600 uppercase tracking-wider mb-1">Step 3</span>
            <h3 className="font-semibold text-slate-900 text-base mb-1.5">BAM Attention Fusion</h3>
            <p className="text-xs text-slate-500 leading-relaxed font-normal">
              Bi-directional Attention Mechanism evaluates cross-modal synergy and computes Hadamard bilinear interaction.
            </p>
          </div>

          <div className="flex flex-col items-start p-5 rounded-2xl bg-slate-50/70 border border-slate-100 hover:border-indigo-200 transition-colors">
            <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-700 flex items-center justify-center mb-4 font-bold text-sm">
              <CheckCircle className="w-5 h-5" />
            </div>
            <span className="text-xs font-semibold text-emerald-600 uppercase tracking-wider mb-1">Step 4</span>
            <h3 className="font-semibold text-slate-900 text-base mb-1.5">Calibrated Decision</h3>
            <p className="text-xs text-slate-500 leading-relaxed font-normal">
              Decision boundary is evaluated using empirical thresholding (τ* = {projectConfig.calibratedThreshold}) to mitigate False Negatives.
            </p>
          </div>
        </div>
      </section>

      {/* Responsible AI Notice */}
      <section className="flex items-start space-x-3.5 p-5 rounded-2xl bg-amber-50/80 border border-amber-200/80 text-amber-900 text-sm">
        <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <div>
          <h4 className="font-bold text-xs uppercase tracking-wider text-amber-800">
            Responsible AI & Academic Research Notice
          </h4>
          <p className="text-xs text-amber-700 mt-1 leading-relaxed">
            This tool provides an AI-generated classification and may make mistakes. Internet memes involve complex cultural sarcasm, vernacular idioms, and weak multimodal alignment. Predictions are provided for research analysis and must be reviewed by a human moderator before consequential moderation actions.
          </p>
        </div>
      </section>
    </div>
  );
};
