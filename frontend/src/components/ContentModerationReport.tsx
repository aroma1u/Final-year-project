import React, { useState } from 'react';
import { 
  ShieldAlert, 
  ShieldCheck, 
  Copy, 
  Check, 
  Download, 
  RotateCcw, 
  Info, 
  Layers, 
  Calendar, 
  FileText, 
  HardDrive, 
  Maximize2, 
  FileCode,
  AlertTriangle
} from 'lucide-react';
import { PredictionResult, OCRStatus } from '../types';

interface ContentModerationReportProps {
  result: PredictionResult;
  imagePreview: string | null;
  imageFile: File | null;
  imageDimensions: { width: number; height: number } | null;
  originalOcrText: string;
  ocrStatus: OCRStatus;
  textUsed: string;
  onAnalyzeAnother: () => void;
}

export const ContentModerationReport: React.FC<ContentModerationReportProps> = ({
  result,
  imagePreview,
  imageFile,
  imageDimensions,
  originalOcrText,
  ocrStatus,
  textUsed,
  onAnalyzeAnother,
}) => {
  const [copiedText, setCopiedText] = useState<boolean>(false);
  const [isDownloading, setIsDownloading] = useState<boolean>(false);

  const isHateful = result.predicted_class === 1;
  const isImageOnly = !textUsed.trim();
  const wasEdited = originalOcrText.trim() !== '' && textUsed.trim() !== originalOcrText.trim();
  const isUserOnlyCaption = !originalOcrText.trim() && textUsed.trim() !== '';

  const handleCopyText = (textToCopy: string) => {
    if (!textToCopy) return;
    navigator.clipboard.writeText(textToCopy);
    setCopiedText(true);
    setTimeout(() => setCopiedText(false), 2000);
  };

  // Honest, data-driven synthesis for "Why Was This Content Classified This Way?"
  const constructWhyThisResult = () => {
    const probDiff = Math.abs(result.probability_hateful - result.calibrated_threshold * 100).toFixed(1);
    
    let textDescription = '';
    if (isImageOnly) {
      textDescription = 'This analysis was conducted without textual input (empty text stream). Textual features were represented solely through BERT tokenizer baseline [CLS]/[SEP] encodings.';
    } else if (wasEdited) {
      textDescription = `The analysis incorporated user-edited text ("${textUsed}"). The original OCR extraction detected ("${originalOcrText}"). Both streams informed the multimodal tokenization.`;
    } else if (isUserOnlyCaption) {
      textDescription = `The image contained no OCR-detected text; a user-supplied caption ("${textUsed}") was supplied to the BERT textual encoder.`;
    } else {
      textDescription = `The image embedded text ("${textUsed}") was automatically extracted via OCR and tokenized into contextual BERT representations.`;
    }

    let marginDescription = '';
    if (isHateful) {
      marginDescription = `The model computed a hateful probability of ${result.probability_hateful.toFixed(1)}%, which exceeds the calibrated decision threshold of ${(result.calibrated_threshold * 100).toFixed(1)}% by ${probDiff} percentage points.`;
    } else {
      marginDescription = `The model computed a hateful probability of ${result.probability_hateful.toFixed(1)}%, remaining below the calibrated decision threshold of ${(result.calibrated_threshold * 100).toFixed(1)}% by a margin of ${probDiff} percentage points.`;
    }

    let attentionDescription = '';
    if (result.attention) {
      const visualAttn = result.attention.visual.toFixed(1);
      const textAttn = result.attention.textual.toFixed(1);
      if (result.attention.visual > result.attention.textual) {
        attentionDescription = `The Bi-directional Attention Mechanism (BAM) assigned higher relative cross-modal attention to visual representations (${visualAttn}%) than textual representations (${textAttn}%).`;
      } else {
        attentionDescription = `The Bi-directional Attention Mechanism (BAM) assigned higher relative cross-modal attention to textual representations (${textAttn}%) than visual representations (${visualAttn}%).`;
      }
    }

    return {
      textDescription,
      marginDescription,
      attentionDescription,
      disclaimer: 'This classification is derived from statistical representations learned by the ResNet-50 visual backbone, BERT contextual embeddings, and BAM Hadamard bilinear fusion. The model predicts categorization without producing semantic causal reasoning.',
    };
  };

  const whyEvidence = constructWhyThisResult();

  // Generate self-contained, downloadable HTML report
  const handleDownloadReport = () => {
    setIsDownloading(true);
    try {
      const formattedDate = new Date(result.timestamp).toLocaleString();
      const filename = imageFile?.name || 'meme_input';
      const fileSize = imageFile ? `${(imageFile.size / 1024).toFixed(1)} KB` : 'N/A';
      const fileType = imageFile?.type || 'image/jpeg';
      const dimensions = imageDimensions ? `${imageDimensions.width} × ${imageDimensions.height} px` : 'N/A';

      const assessmentNote = isImageOnly
        ? `<div class="note-box">
             <strong>Assessment Note:</strong> The image was analyzed without readable or user-provided text. The model's prediction should be interpreted with caution because the current multimodal classifier was developed using visual and textual inputs.
           </div>`
        : '';

      const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MemeGuard AI - Content Moderation Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #1e293b; line-height: 1.5; padding: 32px; background: #f8fafc; margin: 0; }
    .container { max-width: 820px; margin: 0 auto; background: #ffffff; padding: 36px; border-radius: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07); border: 1px solid #e2e8f0; }
    .header { border-bottom: 2px solid #f1f5f9; padding-bottom: 20px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: flex-start; }
    .header h1 { font-size: 24px; font-weight: 800; color: #0f172a; margin: 0 0 6px 0; }
    .header p { font-size: 13px; color: #64748b; margin: 0; }
    .timestamp { font-size: 12px; color: #94a3b8; font-family: monospace; text-align: right; }
    .verdict-box { padding: 20px; border-radius: 14px; margin-bottom: 24px; display: flex; align-items: center; justify-content: space-between; ${
      isHateful 
        ? 'background-color: #fff1f2; border: 1px solid #fecdd3; color: #9f1239;'
        : 'background-color: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46;'
    } }
    .verdict-title { font-size: 20px; font-weight: 800; margin: 0 0 4px 0; }
    .verdict-desc { font-size: 13px; opacity: 0.9; margin: 0; }
    .confidence-badge { font-size: 26px; font-weight: 900; font-family: monospace; text-align: right; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 24px; }
    .card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px; }
    .card h3 { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #475569; margin: 0 0 12px 0; }
    .meta-row { display: flex; justify-content: space-between; font-size: 12px; padding: 4px 0; border-bottom: 1px dashed #e2e8f0; }
    .meta-row:last-child { border-bottom: none; }
    .meta-label { color: #64748b; }
    .meta-val { font-weight: 600; color: #1e293b; font-family: monospace; }
    .bar-wrap { background: #e2e8f0; height: 10px; border-radius: 5px; overflow: hidden; margin-top: 6px; }
    .bar-fill { height: 100%; border-radius: 5px; }
    .section-title { font-size: 16px; font-weight: 700; color: #0f172a; margin: 24px 0 12px 0; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }
    .evidence-p { font-size: 13px; color: #334155; margin-bottom: 8px; }
    .quote-box { background: #f1f5f9; border-left: 4px solid #6366f1; padding: 12px 16px; font-size: 13px; font-style: italic; color: #334155; border-radius: 0 8px 8px 0; margin-bottom: 14px; }
    .note-box { background: #fef3c7; border: 1px solid #fde68a; color: #92400e; padding: 14px; border-radius: 10px; font-size: 12px; margin: 18px 0; }
    .disclaimer-box { background: #f8fafc; border: 1px solid #cbd5e1; color: #475569; padding: 14px; border-radius: 10px; font-size: 12px; margin-top: 24px; text-align: center; }
    .print-btn { display: inline-block; padding: 10px 18px; background: #4f46e5; color: white; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 13px; margin-bottom: 18px; cursor: pointer; border: none; }
    @media print {
      body { background: #ffffff; padding: 0; }
      .container { border: none; box-shadow: none; padding: 0; }
      .no-print { display: none !important; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="no-print" style="text-align: right; margin-bottom: 12px;">
      <button class="print-btn" onclick="window.print()">Print / Save as PDF</button>
    </div>

    <div class="header">
      <div>
        <h1>Content Moderation Report</h1>
        <p>An analysis of the submitted image and its associated text.</p>
        <p style="font-size: 11px; color: #6366f1; font-weight: 600; margin-top: 4px;">MemeGuard AI Platform &bull; MSCMGTB</p>
      </div>
      <div class="timestamp">
        <div>${formattedDate}</div>
        <div>Model: ${result.model_name}</div>
        <div>Threshold (τ*): ${result.calibrated_threshold}</div>
      </div>
    </div>

    ${imagePreview ? `
    <div style="margin-bottom: 24px; text-align: center; background: #f8fafc; padding: 16px; border-radius: 14px; border: 1px solid #e2e8f0;">
      <img src="${imagePreview}" alt="Analyzed Meme Graphic" style="max-height: 260px; max-width: 100%; border-radius: 10px; border: 1px solid #cbd5e1; object-fit: contain;" />
    </div>` : ''}

    <div class="verdict-box">
      <div>
        <div class="verdict-title">${isHateful ? 'Classification: Hateful' : 'Classification: Non-Hateful'}</div>
        <p class="verdict-desc">
          ${isHateful 
            ? 'The model classified this content as hateful based on the visual and textual representations supplied to the classifier.'
            : 'The model did not classify this content as hateful based on the visual and textual representations supplied to the classifier.'}
        </p>
      </div>
      <div>
        <div class="confidence-badge">${result.confidence.toFixed(1)}%</div>
        <div style="font-size: 11px; text-align: right; opacity: 0.8;">Confidence</div>
      </div>
    </div>

    <div class="grid-2">
      <!-- Probabilities Card -->
      <div class="card">
        <h3>Class Probabilities</h3>
        <div class="meta-row">
          <span class="meta-label">Non-Hateful Probability:</span>
          <span class="meta-val">${result.probability_non_hateful.toFixed(2)}%</span>
        </div>
        <div class="bar-wrap">
          <div class="bar-fill" style="width: ${Math.min(100, Math.max(0, result.probability_non_hateful))}%; background: #10b981;"></div>
        </div>

        <div class="meta-row" style="margin-top: 10px;">
          <span class="meta-label">Hateful Probability:</span>
          <span class="meta-val">${result.probability_hateful.toFixed(2)}%</span>
        </div>
        <div class="bar-wrap">
          <div class="bar-fill" style="width: ${Math.min(100, Math.max(0, result.probability_hateful))}%; background: #ef4444;"></div>
        </div>

        <div class="meta-row" style="margin-top: 10px;">
          <span class="meta-label">Calibrated Decision Cutoff (τ*):</span>
          <span class="meta-val">${(result.calibrated_threshold * 100).toFixed(1)}%</span>
        </div>
      </div>

      <!-- Modality Attention Card -->
      <div class="card">
        <h3>Modality Analysis (BAM)</h3>
        <div class="meta-row">
          <span class="meta-label">Visual Attention:</span>
          <span class="meta-val">${result.attention.visual.toFixed(1)}%</span>
        </div>
        <div class="bar-wrap">
          <div class="bar-fill" style="width: ${result.attention.visual}%; background: #6366f1;"></div>
        </div>

        <div class="meta-row" style="margin-top: 10px;">
          <span class="meta-label">Textual Attention:</span>
          <span class="meta-val">${result.attention.textual.toFixed(1)}%</span>
        </div>
        <div class="bar-wrap">
          <div class="bar-fill" style="width: ${result.attention.textual}%; background: #0d9488;"></div>
        </div>
        <p style="font-size: 11px; color: #64748b; margin-top: 8px;">
          * Modality attention weights describe relative fusion coefficients from the Bi-directional Attention Mechanism, not definitive causal proofs.
        </p>
      </div>
    </div>

    <div class="grid-2">
      <!-- Image Metadata Card -->
      <div class="card">
        <h3>Image Details</h3>
        <div class="meta-row">
          <span class="meta-label">Filename:</span>
          <span class="meta-val">${filename}</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">Dimensions:</span>
          <span class="meta-val">${dimensions}</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">File Type:</span>
          <span class="meta-val">${fileType}</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">File Size:</span>
          <span class="meta-val">${fileSize}</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">OCR Text Status:</span>
          <span class="meta-val">${originalOcrText.trim() ? 'Text Extracted' : 'No Text Detected'}</span>
        </div>
      </div>

      <!-- Text Analysis Card -->
      <div class="card">
        <h3>Text Detected in the Image</h3>
        ${isImageOnly ? '<p style="font-size: 12px; color: #64748b;">No readable text was detected in this image. No text was available for textual analysis.</p>' : ''}
        ${originalOcrText.trim() ? `
          <div style="margin-bottom: 8px;">
            <span style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">Original OCR Text:</span>
            <div class="quote-box">"${originalOcrText}"</div>
          </div>
        ` : ''}
        ${wasEdited ? `
          <div>
            <span style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">Text Used for Analysis (User-Edited):</span>
            <div class="quote-box" style="border-left-color: #0d9488;">"${textUsed}"</div>
          </div>
        ` : ''}
        ${isUserOnlyCaption ? `
          <div>
            <span style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">User-Provided Caption Used for Analysis:</span>
            <div class="quote-box" style="border-left-color: #f59e0b;">"${textUsed}"</div>
          </div>
        ` : ''}
      </div>
    </div>

    <!-- Why Was This Content Classified This Way? -->
    <div class="section-title">Why Was This Content Classified This Way? (Why This Result?)</div>
    <p class="evidence-p">&bull; <strong>Text Condition:</strong> ${whyEvidence.textDescription}</p>
    <p class="evidence-p">&bull; <strong>Probability Margin:</strong> ${whyEvidence.marginDescription}</p>
    <p class="evidence-p">&bull; <strong>Attention Distribution:</strong> ${whyEvidence.attentionDescription}</p>
    <p class="evidence-p" style="color: #64748b; font-size: 12px; margin-top: 10px;">${whyEvidence.disclaimer}</p>

    <!-- Assessment Summary -->
    <div class="section-title">Assessment Summary</div>
    <p class="evidence-p">
      The submitted content was evaluated through the ResNet-50 + BERT + BAM multimodal pipeline. 
      The resulting classification is <strong>${isHateful ? 'Hateful' : 'Non-Hateful'}</strong> 
      with a decision confidence of <strong>${result.confidence.toFixed(2)}%</strong> 
      (calibrated decision boundary &tau;* = ${result.calibrated_threshold}).
    </p>
    ${assessmentNote}

    <!-- Human Review Notice -->
    <div class="disclaimer-box">
      <strong>Human Review Notice:</strong> AI-generated moderation results may contain errors. The prediction is a classification aid, not a definitive judgment. Review the original image and its context before taking action.
    </div>
  </div>
</body>
</html>`;

      const blob = new Blob([htmlContent], { type: 'text/html;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      const cleanName = filename.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_');
      link.download = `MemeGuard_Report_${cleanName}_${Date.now()}.html`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download report:', err);
      alert('Unable to generate report download. Please try again.');
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-md space-y-7 animate-in fade-in duration-300">
      {/* 7.1 Report Header & Top Action Buttons */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-slate-100">
        <div className="flex items-center space-x-4">
          {imagePreview && (
            <img
              src={imagePreview}
              alt="Meme Thumbnail"
              className="w-16 h-16 rounded-2xl object-cover bg-slate-900 border border-slate-200 shrink-0 shadow-2xs"
            />
          )}
          <div>
            <div className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-700 text-[11px] font-semibold mb-1">
              <Layers className="w-3 h-3" />
              <span>MemeGuard AI Moderation Engine</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">
              Content Moderation Report
            </h2>
            <p className="text-xs text-slate-500 font-normal">
              An analysis of the submitted image and its associated text.
            </p>
            <div className="text-[11px] text-slate-400 mt-1 flex items-center space-x-1">
              <Calendar className="w-3 h-3" />
              <span>{new Date(result.timestamp).toLocaleString()}</span>
            </div>
          </div>
        </div>

        {/* Section 8: Action Buttons (Top) */}
        <div className="flex items-center space-x-2.5 self-start sm:self-auto shrink-0">
          <button
            onClick={onAnalyzeAnother}
            className="inline-flex items-center space-x-1.5 px-3.5 py-2 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold transition-all cursor-pointer shadow-2xs"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
            <span>Analyze Another Image</span>
          </button>

          <button
            onClick={handleDownloadReport}
            disabled={isDownloading}
            className="inline-flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-all shadow-sm shadow-indigo-600/30 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>{isDownloading ? 'Generating...' : 'Download Report'}</span>
          </button>
        </div>
      </div>

      {/* 7.2 Primary Result — Hateful or Non-Hateful Banner */}
      <div
        className={`p-6 rounded-2xl border transition-all ${
          isHateful
            ? 'bg-rose-50/90 border-rose-200 text-rose-950'
            : 'bg-emerald-50/90 border-emerald-200 text-emerald-950'
        }`}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start space-x-4">
            <div
              className={`w-14 h-14 rounded-2xl flex items-center justify-center shrink-0 shadow-md ${
                isHateful ? 'bg-rose-600 text-white shadow-rose-200' : 'bg-emerald-600 text-white shadow-emerald-200'
              }`}
            >
              {isHateful ? <ShieldAlert className="w-8 h-8" /> : <ShieldCheck className="w-8 h-8" />}
            </div>
            <div>
              <div className="text-[11px] font-bold uppercase tracking-wider opacity-75">
                Primary Classification Result
              </div>
              <h3 className="text-xl sm:text-2xl font-black tracking-tight mt-0.5">
                {isHateful ? 'Classification: Hateful' : 'Classification: Non-Hateful'}
              </h3>
              <p className="text-xs mt-1.5 max-w-xl leading-relaxed opacity-90 font-normal">
                {isHateful
                  ? 'The model classified this content as hateful based on the visual and textual representations supplied to the classifier.'
                  : 'The model did not classify this content as hateful based on the visual and textual representations supplied to the classifier.'}
              </p>
            </div>
          </div>

          <div className="text-left sm:text-right shrink-0 border-t sm:border-t-0 pt-3 sm:pt-0 border-slate-200/40">
            <div className="text-[11px] font-semibold uppercase tracking-wider opacity-70">Decision Confidence</div>
            <div className="text-2xl sm:text-3xl font-black font-mono tracking-tight mt-0.5">
              {result.confidence.toFixed(1)}%
            </div>
            <div className="text-[10px] font-mono mt-0.5 opacity-80">
              Cutoff &tau;* = {(result.calibrated_threshold * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      {/* 7.3 Confidence & Class Probabilities Grid */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs font-bold text-slate-700 uppercase tracking-wider">
          <span>Confidence & Class Probabilities</span>
          <span className="text-[10px] text-slate-400 lowercase font-normal">
            (model predicted probability score)
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Non-Hateful Probability Card */}
          <div className="bg-slate-50 p-4 rounded-2xl border border-slate-100 space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-slate-700">Non-Hateful Probability</span>
              <span className="font-mono font-bold text-emerald-700 text-sm">
                {result.probability_non_hateful.toFixed(2)}%
              </span>
            </div>
            <div className="w-full h-2.5 bg-slate-200/80 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, Math.max(0, result.probability_non_hateful))}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-500">
              Likelihood that visual and textual streams contain benign content.
            </p>
          </div>

          {/* Hateful Probability Card */}
          <div className="bg-slate-50 p-4 rounded-2xl border border-slate-100 space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-slate-700">Hateful Probability</span>
              <span className="font-mono font-bold text-rose-700 text-sm">
                {result.probability_hateful.toFixed(2)}%
              </span>
            </div>
            <div className="w-full h-2.5 bg-slate-200/80 rounded-full overflow-hidden">
              <div
                className="h-full bg-rose-500 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, Math.max(0, result.probability_hateful))}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-500">
              Compared against calibrated threshold &tau;* = {(result.calibrated_threshold * 100).toFixed(1)}%.
            </p>
          </div>
        </div>
      </div>

      {/* 7.4 "Why This Result?" Section */}
      <div className="bg-slate-50/70 rounded-2xl p-5 border border-slate-200/80 space-y-3">
        <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-800">
          <Info className="w-4 h-4 text-indigo-600" />
          <span>Why Was This Content Classified This Way? (Why This Result?)</span>
        </div>

        <div className="space-y-2.5 text-xs text-slate-700 leading-relaxed font-normal">
          <div className="flex items-start space-x-2">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 shrink-0 mt-1.5" />
            <div>
              <strong className="text-slate-900">Text Condition: </strong>
              <span>{whyEvidence.textDescription}</span>
            </div>
          </div>

          <div className="flex items-start space-x-2">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 shrink-0 mt-1.5" />
            <div>
              <strong className="text-slate-900">Decision Margin: </strong>
              <span>{whyEvidence.marginDescription}</span>
            </div>
          </div>

          {whyEvidence.attentionDescription && (
            <div className="flex items-start space-x-2">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 shrink-0 mt-1.5" />
              <div>
                <strong className="text-slate-900">Attention Distribution: </strong>
                <span>{whyEvidence.attentionDescription}</span>
              </div>
            </div>
          )}

          <p className="text-[11px] text-slate-500 pt-1 border-t border-slate-200/60 font-normal">
            {whyEvidence.disclaimer}
          </p>
        </div>
      </div>

      {/* Two Column Section: 7.5 Extracted Text & 7.6 Image Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
        {/* 7.5 Extracted Text ("Text Detected in the Image") */}
        <div className="bg-white rounded-2xl p-5 border border-slate-200/80 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center space-x-1.5">
              <FileText className="w-4 h-4 text-teal-600" />
              <span>Text Detected in the Image</span>
            </span>

            {textUsed.trim() && (
              <button
                onClick={() => handleCopyText(textUsed)}
                className="inline-flex items-center space-x-1 text-[11px] text-indigo-600 hover:text-indigo-800 font-medium px-2 py-0.5 rounded-md hover:bg-indigo-50 transition-colors"
                title="Copy text"
              >
                {copiedText ? <Check className="w-3 h-3 text-emerald-600" /> : <Copy className="w-3 h-3" />}
                <span>{copiedText ? 'Copied' : 'Copy Text'}</span>
              </button>
            )}
          </div>

          {isImageOnly ? (
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 text-xs text-slate-500 space-y-1">
              <p className="font-semibold text-slate-700">No readable text was detected in this image.</p>
              <p className="text-[11px]">Text Evidence: No text was available for textual analysis.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {originalOcrText.trim() && (
                <div>
                  <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">
                    Original OCR Text
                  </span>
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 text-xs font-mono text-slate-800">
                    "{originalOcrText}"
                  </div>
                </div>
              )}

              {wasEdited && (
                <div>
                  <span className="text-[10px] uppercase font-bold text-teal-600 block mb-1">
                    Text Used for Analysis (User-Edited)
                  </span>
                  <div className="p-3 rounded-xl bg-teal-50/60 border border-teal-100 text-xs font-mono text-teal-900">
                    "{textUsed}"
                  </div>
                </div>
              )}

              {isUserOnlyCaption && (
                <div>
                  <span className="text-[10px] uppercase font-bold text-indigo-600 block mb-1">
                    User-Provided Caption Used for Analysis
                  </span>
                  <div className="p-3 rounded-xl bg-indigo-50/60 border border-indigo-100 text-xs font-mono text-indigo-900">
                    "{textUsed}"
                  </div>
                </div>
              )}
            </div>
          )}

          {ocrStatus === 'failed' && (
            <div className="p-2.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs">
              OCR service was unavailable or encountered an error during text extraction.
            </div>
          )}
        </div>

        {/* 7.6 Image Details ("Image Details") */}
        <div className="bg-white rounded-2xl p-5 border border-slate-200/80 space-y-3">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center space-x-1.5">
            <HardDrive className="w-4 h-4 text-indigo-600" />
            <span>Image Details</span>
          </span>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Filename</span>
              <span className="font-semibold text-slate-800 font-mono truncate max-w-[180px]">
                {imageFile?.name || 'meme_upload'}
              </span>
            </div>

            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500 flex items-center space-x-1">
                <Maximize2 className="w-3 h-3 text-slate-400" />
                <span>Dimensions</span>
              </span>
              <span className="font-semibold text-slate-800 font-mono">
                {imageDimensions ? `${imageDimensions.width} × ${imageDimensions.height} px` : '224 × 224 px'}
              </span>
            </div>

            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500 flex items-center space-x-1">
                <FileCode className="w-3 h-3 text-slate-400" />
                <span>Format</span>
              </span>
              <span className="font-semibold text-slate-800 font-mono uppercase">
                {imageFile?.type || 'image/jpeg'}
              </span>
            </div>

            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">File Size</span>
              <span className="font-semibold text-slate-800 font-mono">
                {imageFile ? `${(imageFile.size / 1024).toFixed(1)} KB` : 'N/A'}
              </span>
            </div>

            <div className="flex justify-between py-1.5">
              <span className="text-slate-500">OCR Text Status</span>
              <span className="font-semibold text-slate-800 font-mono">
                {originalOcrText.trim() ? 'Text Detected' : 'No Text Detected'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 7.7 Visual and Textual Analysis ("Modality Analysis") */}
      {result.attention && (
        <div className="space-y-3 pt-2">
          <div className="flex items-center justify-between text-xs font-bold text-slate-700 uppercase tracking-wider">
            <span className="flex items-center space-x-1.5">
              <Layers className="w-4 h-4 text-indigo-600" />
              <span>Modality Analysis (Bi-directional Attention Mechanism)</span>
            </span>
            <span className="text-[10px] text-slate-400 font-normal lowercase">
              (learned fusion weights)
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="p-4 bg-indigo-50/50 rounded-2xl border border-indigo-100/60 space-y-2">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-indigo-900">Visual Attention</span>
                <span className="font-mono font-bold text-indigo-700 text-sm">
                  {result.attention.visual.toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-2 bg-indigo-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-600 rounded-full transition-all duration-500"
                  style={{ width: `${result.attention.visual}%` }}
                />
              </div>
              <p className="text-[11px] text-slate-500">
                Weight assigned to ResNet-50 visual feature representation.
              </p>
            </div>

            <div className="p-4 bg-teal-50/50 rounded-2xl border border-teal-100/60 space-y-2">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-teal-900">Textual Attention</span>
                <span className="font-mono font-bold text-teal-700 text-sm">
                  {result.attention.textual.toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-2 bg-teal-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-teal-600 rounded-full transition-all duration-500"
                  style={{ width: `${result.attention.textual}%` }}
                />
              </div>
              <p className="text-[11px] text-slate-500">
                Weight assigned to BERT textual contextual embedding.
              </p>
            </div>
          </div>

          <p className="text-[11px] text-slate-400 italic">
            * Note: These values describe relative modality attention produced by the model during Hadamard bilinear fusion and are not definitive causal explanations.
          </p>
        </div>
      )}

      {/* 7.8 Final Assessment ("Assessment Summary") */}
      <div className="p-5 rounded-2xl bg-slate-50 border border-slate-200/80 space-y-3">
        <div className="text-xs font-bold uppercase tracking-wider text-slate-800">
          Assessment Summary
        </div>

        <p className="text-xs text-slate-700 leading-relaxed font-normal">
          The submitted content was evaluated through the {result.model_name} multimodal architecture. 
          The overall classification verdict is <strong>{isHateful ? 'Hateful' : 'Non-Hateful'}</strong> 
          with a decision confidence score of <strong>{result.confidence.toFixed(2)}%</strong> 
          (calibrated decision threshold &tau;* = {result.calibrated_threshold}).
        </p>

        {/* Exact Assessment Note when analyzed without text (empty text input) */}
        {isImageOnly && (
          <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200/80 text-amber-900 text-xs flex items-start space-x-2.5">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <span className="font-bold text-amber-800 uppercase tracking-wider text-[10px]">
                Assessment Note
              </span>
              <p className="font-medium text-amber-800 leading-relaxed">
                The image was analyzed without readable or user-provided text. The model's prediction should be interpreted with caution because the current multimodal classifier was developed using visual and textual inputs.
              </p>
            </div>
          </div>
        )}

        <div className="flex items-center space-x-4 text-[11px] text-slate-400 pt-1">
          <span className="flex items-center space-x-1">
            <Calendar className="w-3.5 h-3.5" />
            <span>{new Date(result.timestamp).toLocaleString()}</span>
          </span>
          <span>&bull;</span>
          <span>Analysis ID: {result.id || 'N/A'}</span>
        </div>
      </div>

      {/* 7.9 Human Review Notice */}
      <div className="p-4 rounded-2xl bg-amber-50/70 border border-amber-200/60 text-amber-900 text-xs flex items-start space-x-3">
        <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
        <p className="leading-relaxed font-normal text-amber-800">
          <strong>Human Review Notice:</strong> AI-generated moderation results may contain errors. The prediction is a classification aid, not a definitive judgment. Review the original image and its context before taking action.
        </p>
      </div>

      {/* Bottom Action Row */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-100">
        <button
          onClick={onAnalyzeAnother}
          className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold transition-all cursor-pointer shadow-2xs"
        >
          <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
          <span>Analyze Another Image</span>
        </button>

        <button
          onClick={handleDownloadReport}
          disabled={isDownloading}
          className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-all shadow-sm shadow-indigo-600/30 cursor-pointer"
        >
          <Download className="w-3.5 h-3.5" />
          <span>{isDownloading ? 'Generating...' : 'Download Report'}</span>
        </button>
      </div>
    </div>
  );
};
