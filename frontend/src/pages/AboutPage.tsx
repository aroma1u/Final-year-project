import React from 'react';
import { 
  HelpCircle, 
  Upload, 
  FileText, 
  Cpu, 
  FileCheck2, 
  AlertTriangle,
  Sparkles
} from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Section A: Hero Section */}
      <section className="bg-white p-8 sm:p-10 rounded-3xl border border-slate-200/80 shadow-xs space-y-4">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-semibold">
          <Sparkles className="w-3.5 h-3.5" />
          <span>About MemeGuard AI</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight leading-tight">
          Understand the Content Behind the Image
        </h1>
        <p className="text-base sm:text-lg text-slate-600 leading-relaxed font-normal">
          MemeGuard AI helps users examine memes and image-based social media content by analyzing visual information and associated text to identify content that may be hateful.
        </p>
      </section>

      {/* Section B: Why MemeGuard AI? */}
      <section className="bg-white p-8 rounded-3xl border border-slate-200/80 shadow-xs space-y-4">
        <div className="w-10 h-10 rounded-xl bg-indigo-100 text-indigo-700 flex items-center justify-center">
          <HelpCircle className="w-5 h-5" />
        </div>
        <h2 className="text-xl font-bold text-slate-900">Why MemeGuard AI?</h2>
        <div className="space-y-3 text-sm text-slate-600 leading-relaxed font-normal">
          <p>
            The meaning of a meme often depends on the combination of its visual image and associated text. In many instances, text alone does not provide the complete context, and visual information alone can easily miss a harmful or satirical message embedded within the composition.
          </p>
          <p>
            MemeGuard AI combines visual feature representations and textual contextual embeddings through bi-directional cross-attention fusion to assist with hateful-content classification, detecting emergent toxicity that unimodal classifiers overlook.
          </p>
        </div>
      </section>

      {/* Section C: What Can Users Do? (Clean 4-Card Grid) */}
      <section className="space-y-4">
        <div className="px-1">
          <h2 className="text-xl font-bold text-slate-900">What Can Users Do?</h2>
          <p className="text-xs text-slate-500 mt-1">Core capabilities offered across the MemeGuard AI platform.</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Card 1 */}
          <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col justify-between hover:border-indigo-200 transition-colors">
            <div>
              <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center mb-3">
                <Upload className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-slate-900 text-sm mb-1">1. Upload an Image</h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                Submit a meme or supported social media image (PNG, JPG, JPEG, WEBP up to 10MB) for visual feature processing.
              </p>
            </div>
          </div>

          {/* Card 2 */}
          <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col justify-between hover:border-indigo-200 transition-colors">
            <div>
              <div className="w-10 h-10 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center mb-3">
                <FileText className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-slate-900 text-sm mb-1">2. Extract and Review Text</h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                Use OCR to extract visible text when the OCR service is available. Review and correct the extracted text or supply optional context when necessary.
              </p>
            </div>
          </div>

          {/* Card 3 */}
          <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col justify-between hover:border-indigo-200 transition-colors">
            <div>
              <div className="w-10 h-10 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center mb-3">
                <Cpu className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-slate-900 text-sm mb-1">3. Analyze Content</h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                Submit the image and associated text (or proceed with image-only analysis) to the trained multimodal classification model.
              </p>
            </div>
          </div>

          {/* Card 4 */}
          <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col justify-between hover:border-indigo-200 transition-colors">
            <div>
              <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center mb-3">
                <FileCheck2 className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-slate-900 text-sm mb-1">4. Read the Moderation Report</h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                Review the comprehensive classification result, available probabilities, extracted text, and supported evidence in a detailed report.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Section D: How to Use the Website (4-Step Workflow with Clear Icons) */}
      <section className="bg-white p-8 rounded-3xl border border-slate-200/80 shadow-xs space-y-6">
        <div>
          <h2 className="text-xl font-bold text-slate-900">How to Use the Website</h2>
          <p className="text-xs text-slate-500 mt-1">A streamlined workflow from upload to moderation decision.</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 relative">
          <div className="flex flex-col items-center text-center p-4 rounded-2xl bg-slate-50 border border-slate-100">
            <div className="w-12 h-12 rounded-2xl bg-indigo-600 text-white flex items-center justify-center font-bold text-sm mb-3 shadow-md shadow-indigo-100">
              <Upload className="w-6 h-6" />
            </div>
            <span className="text-[11px] font-bold text-indigo-600 uppercase tracking-wider mb-1">Step 1</span>
            <span className="font-semibold text-slate-900 text-sm">Upload Image</span>
            <p className="text-[11px] text-slate-500 mt-1">Drop or browse any meme image file</p>
          </div>

          <div className="flex flex-col items-center text-center p-4 rounded-2xl bg-slate-50 border border-slate-100">
            <div className="w-12 h-12 rounded-2xl bg-teal-600 text-white flex items-center justify-center font-bold text-sm mb-3 shadow-md shadow-teal-100">
              <FileText className="w-6 h-6" />
            </div>
            <span className="text-[11px] font-bold text-teal-600 uppercase tracking-wider mb-1">Step 2</span>
            <span className="font-semibold text-slate-900 text-sm">Review Text</span>
            <p className="text-[11px] text-slate-500 mt-1">Verify OCR or leave empty if none</p>
          </div>

          <div className="flex flex-col items-center text-center p-4 rounded-2xl bg-slate-50 border border-slate-100">
            <div className="w-12 h-12 rounded-2xl bg-amber-600 text-white flex items-center justify-center font-bold text-sm mb-3 shadow-md shadow-amber-100">
              <Cpu className="w-6 h-6" />
            </div>
            <span className="text-[11px] font-bold text-amber-600 uppercase tracking-wider mb-1">Step 3</span>
            <span className="font-semibold text-slate-900 text-sm">Analyze Content</span>
            <p className="text-[11px] text-slate-500 mt-1">Run multimodal inference</p>
          </div>

          <div className="flex flex-col items-center text-center p-4 rounded-2xl bg-slate-50 border border-slate-100">
            <div className="w-12 h-12 rounded-2xl bg-emerald-600 text-white flex items-center justify-center font-bold text-sm mb-3 shadow-md shadow-emerald-100">
              <FileCheck2 className="w-6 h-6" />
            </div>
            <span className="text-[11px] font-bold text-emerald-600 uppercase tracking-wider mb-1">Step 4</span>
            <span className="font-semibold text-slate-900 text-sm">Read Report</span>
            <p className="text-[11px] text-slate-500 mt-1">Examine metrics or download report</p>
          </div>
        </div>
      </section>

      {/* Section E: Responsible Use */}
      <section className="flex items-start space-x-3.5 p-6 rounded-3xl bg-amber-50/80 border border-amber-200/80 text-amber-900 text-sm">
        <AlertTriangle className="w-6 h-6 text-amber-600 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h4 className="font-bold text-xs uppercase tracking-wider text-amber-800">
            Responsible Use
          </h4>
          <p className="text-xs text-amber-800 leading-relaxed font-medium">
            MemeGuard AI is a research prototype. Its classifications may be incorrect or incomplete. Results should be reviewed by a human before making consequential moderation decisions.
          </p>
        </div>
      </section>
    </div>
  );
};
