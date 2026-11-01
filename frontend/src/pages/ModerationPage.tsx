import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, 
  FileText, 
  Image as ImageIcon, 
  Trash2, 
  RefreshCw, 
  Sparkles, 
  AlertCircle, 
  CheckCircle2, 
  ShieldCheck, 
  ArrowRight, 
  Info, 
  Clock, 
  ScanText, 
  Loader2 
} from 'lucide-react';
import { PredictionResult, SampleInput, HealthStatus, OCRStatus } from '../types';
import { apiService } from '../services/api';
import { ContentModerationReport } from '../components/ContentModerationReport';

interface ModerationPageProps {
  health: HealthStatus;
  onPredictionComplete: (result: PredictionResult) => void;
}

export const ModerationPage: React.FC<ModerationPageProps> = ({
  health,
  onPredictionComplete,
}) => {
  // Input states
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const [captionText, setCaptionText] = useState<string>('');
  const [originalOcrText, setOriginalOcrText] = useState<string>('');
  
  // OCR states
  const [ocrStatus, setOcrStatus] = useState<OCRStatus>('waiting');
  const [isExtractingText, setIsExtractingText] = useState<boolean>(false);
  const [ocrErrorMessage, setOcrErrorMessage] = useState<string | null>(null);

  // Execution states
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  
  // Samples
  const [samples, setSamples] = useState<SampleInput[]>([]);
  
  // Drag and drop state
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Load sample metadata on mount
  useEffect(() => {
    async function loadSamples() {
      const data = await apiService.getSamples();
      setSamples(data);
    }
    loadSamples();
  }, []);

  // Trigger OCR text extraction
  const triggerOCR = async (file: File) => {
    setIsExtractingText(true);
    setOcrStatus('extracting');
    setOcrErrorMessage(null);

    try {
      const ocrRes = await apiService.extractText(file);
      if (ocrRes.text_detected && ocrRes.extracted_text) {
        setOriginalOcrText(ocrRes.extracted_text);
        setCaptionText(ocrRes.extracted_text);
        setOcrStatus('extracted');
      } else {
        setOriginalOcrText('');
        setOcrStatus('no_text');
      }
    } catch (err) {
      setOcrStatus('failed');
      const msg = err instanceof Error ? err.message : 'OCR service is currently unavailable.';
      setOcrErrorMessage(`OCR notice: ${msg}. You can optionally type text manually or proceed with image-only analysis.`);
    } finally {
      setIsExtractingText(false);
    }
  };

  // Handle file selection
  const handleFileChange = (file: File) => {
    setErrorMessage(null);
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      setErrorMessage(`Unsupported format (${file.type}). Please upload PNG, JPG, or WEBP.`);
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setErrorMessage('Image size exceeds 10MB limit.');
      return;
    }

    setSelectedFile(file);
    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);
    setResult(null);

    // Read natural dimensions
    const img = new Image();
    img.onload = () => {
      setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = previewUrl;

    // Automatically trigger OCR on the newly uploaded image
    triggerOCR(file);
  };

  const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileChange(e.target.files[0]);
    }
  };

  // Drag & drop handlers
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const onDragLeave = () => {
    setIsDragging(false);
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  // Clear inputs & reset state cleanly
  const handleClear = () => {
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
    }
    setSelectedFile(null);
    setImagePreview(null);
    setImageDimensions(null);
    setCaptionText('');
    setOriginalOcrText('');
    setErrorMessage(null);
    setOcrErrorMessage(null);
    setOcrStatus('waiting');
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Load verified local sample
  const handleLoadSample = async (sample: SampleInput) => {
    setErrorMessage(null);
    setOcrErrorMessage(null);
    setResult(null);
    setOriginalOcrText(sample.text);
    setCaptionText(sample.text);
    setOcrStatus('extracted');

    try {
      const res = await fetch(sample.image_url);
      if (!res.ok) throw new Error('Sample image unreachable');
      const blob = await res.blob();
      const file = new File([blob], `${sample.id}.jpg`, { type: 'image/jpeg' });
      setSelectedFile(file);
      setImagePreview(sample.image_url);

      const img = new Image();
      img.onload = () => {
        setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.src = sample.image_url;
    } catch {
      setErrorMessage(`Sample image '${sample.title}' loaded. Click Analyze to process.`);
    }
  };

  // Analyze Content Submission — Optional Text Supported
  const handleAnalyze = async () => {
    setErrorMessage(null);

    // Section 6.3: Image Missing validation
    if (!selectedFile) {
      setErrorMessage('Please upload an image before starting the analysis.');
      return;
    }

    if (!health.model_loaded) {
      setErrorMessage('The model inference service is currently offline. Ensure backend is running.');
      return;
    }

    setIsAnalyzing(true);
    try {
      // Allow empty caption string for image-only inference
      const textToSend = captionText.trim();
      const prediction = await apiService.predictContent(selectedFile, textToSend);

      const isImageOnly = !textToSend;
      const fullResult: PredictionResult = {
        ...prediction,
        id: `analysis-${Date.now()}`,
        image_preview_url: imagePreview || undefined,
        input_text: textToSend,
        original_ocr_text: originalOcrText,
        text_used: textToSend,
        is_empty_text: isImageOnly,
        image_filename: selectedFile.name,
        image_dimensions: imageDimensions ? `${imageDimensions.width} × ${imageDimensions.height} px` : undefined,
        image_size: `${(selectedFile.size / 1024).toFixed(1)} KB`,
        image_type: selectedFile.type,
        ocr_status: ocrStatus,
      };

      setResult(fullResult);
      onPredictionComplete(fullResult);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Top Notification / Error Banner */}
      {errorMessage && (
        <div className="p-4 rounded-2xl bg-rose-50 border border-rose-200/80 text-rose-800 text-sm flex items-start space-x-3 transition-all animate-in fade-in">
          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="font-semibold text-xs uppercase tracking-wider text-rose-900">Notice</h4>
            <p className="text-xs text-rose-700 mt-0.5">{errorMessage}</p>
          </div>
          <button 
            onClick={() => setErrorMessage(null)} 
            className="text-rose-400 hover:text-rose-700 text-xs font-semibold cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Studio Grid: Left Input, Right Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* ========================================================================= */}
        {/* LEFT COLUMN: Input Workspace (7 cols on desktop)                          */}
        {/* ========================================================================= */}
        <div className="lg:col-span-6 space-y-6">
          {/* Card 1: Image Upload */}
          <div className="bg-white rounded-3xl p-6 sm:p-7 border border-slate-200/80 shadow-xs">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2">
                  <ImageIcon className="w-4 h-4 text-indigo-600" />
                  <span>1. Visual Input (Image / Meme)</span>
                </h2>
                <p className="text-xs text-slate-500 mt-0.5 font-normal">
                  ResNet-50 visual feature extractor (2048-dimensional representation)
                </p>
              </div>
              {selectedFile && (
                <button
                  onClick={handleClear}
                  className="text-xs text-rose-600 hover:text-rose-700 font-medium flex items-center space-x-1 cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Remove</span>
                </button>
              )}
            </div>

            {/* Drag & Drop Zone or Preview Card */}
            {!imagePreview ? (
              <div
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${
                  isDragging
                    ? 'border-indigo-500 bg-indigo-50/50 scale-[0.99]'
                    : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50/50'
                }`}
              >
                <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 shadow-xs">
                  <Upload className="w-6 h-6" />
                </div>
                <h3 className="text-sm font-semibold text-slate-800">
                  Drop your meme image here, or <span className="text-indigo-600 underline">browse</span>
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Supports PNG, JPG, JPEG, WEBP up to 10MB
                </p>
                <div className="inline-flex items-center space-x-1 text-[11px] text-indigo-600 font-medium mt-3 bg-indigo-50 px-2.5 py-1 rounded-full">
                  <ScanText className="w-3 h-3" />
                  <span>OCR will automatically extract embedded text upon upload</span>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative rounded-2xl overflow-hidden bg-slate-900 border border-slate-200 max-h-80 flex items-center justify-center">
                  <img
                    src={imagePreview}
                    alt="Uploaded meme preview"
                    className="max-h-80 w-auto object-contain"
                  />
                  {selectedFile && (
                    <div className="absolute bottom-2 left-2 right-2 bg-slate-900/80 backdrop-blur-xs px-3 py-1.5 rounded-xl text-white text-xs flex justify-between items-center">
                      <span className="truncate max-w-[200px]">{selectedFile.name}</span>
                      <span className="text-slate-300">
                        {imageDimensions ? `${imageDimensions.width}×${imageDimensions.height}px &bull; ` : ''}
                        {(selectedFile.size / 1024).toFixed(1)} KB
                      </span>
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-between">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="text-xs text-indigo-600 hover:text-indigo-800 font-medium flex items-center space-x-1 cursor-pointer"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>Replace Image</span>
                  </button>
                  {selectedFile && (
                    <button
                      onClick={() => triggerOCR(selectedFile)}
                      disabled={isExtractingText}
                      className="text-xs text-teal-700 hover:text-teal-900 font-medium flex items-center space-x-1 px-2.5 py-1 rounded-lg bg-teal-50 border border-teal-200/60 cursor-pointer disabled:opacity-50"
                    >
                      <ScanText className="w-3.5 h-3.5 text-teal-600" />
                      <span>{isExtractingText ? 'Extracting...' : 'Re-run OCR'}</span>
                    </button>
                  )}
                </div>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/jpg,image/webp"
              onChange={onFileInputChange}
              className="hidden"
            />
          </div>

          {/* Card 2: Section 6.2 Text Extracted from Image (Optional) */}
          <div className="bg-white rounded-3xl p-6 sm:p-7 border border-slate-200/80 shadow-xs">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
              <div>
                <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2">
                  <FileText className="w-4 h-4 text-teal-600" />
                  <span>2. Text Extracted from Image (Optional)</span>
                </h2>
                <p className="text-xs text-slate-500 mt-0.5 font-normal">
                  OCR automatically extracts readable text from your image. You can review or edit the extracted text, add an optional caption, or leave this field empty if no text is available.
                </p>
              </div>

              {/* Exact OCR Status Indicator */}
              <div className="shrink-0 flex items-center">
                {ocrStatus === 'waiting' && (
                  <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-medium bg-slate-100 text-slate-600 border border-slate-200">
                    <Clock className="w-3 h-3 mr-1.5 text-slate-400" />
                    <span>Waiting for image</span>
                  </span>
                )}
                {ocrStatus === 'extracting' && (
                  <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-medium bg-indigo-50 text-indigo-700 border border-indigo-200 animate-pulse">
                    <Loader2 className="w-3 h-3 mr-1.5 text-indigo-600 animate-spin" />
                    <span>Extracting text...</span>
                  </span>
                )}
                {ocrStatus === 'extracted' && (
                  <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-medium bg-emerald-50 text-emerald-800 border border-emerald-200">
                    <CheckCircle2 className="w-3 h-3 mr-1.5 text-emerald-600" />
                    <span>Text extracted</span>
                  </span>
                )}
                {ocrStatus === 'no_text' && (
                  <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-medium bg-amber-50 text-amber-800 border border-amber-200">
                    <Info className="w-3 h-3 mr-1.5 text-amber-600" />
                    <span>No text detected</span>
                  </span>
                )}
                {ocrStatus === 'failed' && (
                  <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-medium bg-rose-50 text-rose-800 border border-rose-200">
                    <AlertCircle className="w-3 h-3 mr-1.5 text-rose-600" />
                    <span>OCR notice</span>
                  </span>
                )}
              </div>
            </div>

            {/* Section 6.1: Friendly No-Text Notice */}
            {ocrStatus === 'no_text' && (
              <div className="mb-3 p-3 rounded-2xl bg-amber-50/80 border border-amber-200/80 text-amber-900 text-xs flex items-center space-x-2">
                <Info className="w-4 h-4 shrink-0 text-amber-600" />
                <span>No text detected in this image. You can optionally add a caption or context, or continue with image-only analysis.</span>
              </div>
            )}

            {/* Non-blocking OCR Notice */}
            {ocrErrorMessage && (
              <div className="mb-3 p-3 rounded-2xl bg-slate-50 border border-slate-200 text-slate-700 text-xs flex items-center space-x-2">
                <Info className="w-4 h-4 shrink-0 text-slate-500" />
                <span>{ocrErrorMessage}</span>
              </div>
            )}

            <div className="relative">
              <textarea
                value={captionText}
                onChange={(e) => setCaptionText(e.target.value.slice(0, 2000))}
                rows={4}
                disabled={isExtractingText}
                placeholder="Extracted text will appear here. You may edit it or enter an optional caption..."
                className={`w-full rounded-2xl border p-4 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-hidden focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-600 transition-all font-normal ${
                  isExtractingText 
                    ? 'bg-slate-50 border-indigo-200 text-slate-400 cursor-wait' 
                    : 'bg-white border-slate-200'
                }`}
              />

              {isExtractingText && (
                <div className="absolute inset-0 bg-white/70 backdrop-blur-xs rounded-2xl flex flex-col items-center justify-center text-indigo-700 text-xs font-semibold space-y-2 pointer-events-none">
                  <Loader2 className="w-6 h-6 animate-spin text-indigo-600" />
                  <span>Extracting text from image via EasyOCR...</span>
                </div>
              )}

              <div className="flex justify-between items-center mt-1.5 px-1 text-[11px] text-slate-400">
                <span>BERT Tokenizer (uncased, max 128 tokens)</span>
                <div className="flex items-center space-x-3">
                  {captionText && (
                    <button
                      onClick={() => setCaptionText('')}
                      className="text-xs text-slate-400 hover:text-slate-600 font-medium cursor-pointer"
                    >
                      Clear text
                    </button>
                  )}
                  <span className={captionText.length > 1800 ? 'text-amber-600 font-medium' : ''}>
                    {captionText.length} / 2000 chars
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Sample Demonstrations Selector */}
          <div className="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-xs">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold text-slate-700 flex items-center space-x-1.5">
                <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                <span>Demonstration Samples for Faculty Review</span>
              </span>
              <span className="text-[11px] text-slate-400">Click to populate image & OCR text</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {samples.map((sample) => (
                <button
                  key={sample.id}
                  onClick={() => handleLoadSample(sample)}
                  className="flex flex-col text-left p-3.5 rounded-2xl border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50/40 transition-all cursor-pointer group"
                >
                  <span className="text-xs font-semibold text-slate-900 group-hover:text-indigo-900">
                    {sample.title}
                  </span>
                  <span className="text-[11px] text-slate-500 italic truncate mt-1">
                    "{sample.text}"
                  </span>
                  <span className="text-[10px] font-medium text-indigo-600 mt-2">
                    {sample.category}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Action Row — Section 6.3: Enabled as long as image is present & model is loaded */}
          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing || isExtractingText || !selectedFile || !health.model_loaded}
              className={`flex-1 inline-flex items-center justify-center space-x-2 py-4 px-6 rounded-2xl font-bold text-sm text-white shadow-md transition-all cursor-pointer ${
                isAnalyzing || isExtractingText || !selectedFile || !health.model_loaded
                  ? 'bg-slate-300 text-slate-500 cursor-not-allowed shadow-none'
                  : 'bg-indigo-600 hover:bg-indigo-500 shadow-indigo-600/30 hover:shadow-indigo-600/50'
              }`}
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Analyzing visual and textual content...</span>
                </>
              ) : isExtractingText ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Waiting for OCR extraction...</span>
                </>
              ) : !selectedFile ? (
                <span>Upload an image to start analysis</span>
              ) : (
                <>
                  <span>Analyze Content</span>
                  <ArrowRight className="w-4 h-4 ml-1" />
                </>
              )}
            </button>

            <button
              onClick={handleClear}
              className="py-4 px-5 rounded-2xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-600 text-sm font-semibold transition-all cursor-pointer"
            >
              Clear Input
            </button>
          </div>
        </div>

        {/* ========================================================================= */}
        {/* RIGHT COLUMN: Real Inference Results / Detailed Moderation Report         */}
        {/* ========================================================================= */}
        <div className="lg:col-span-6 space-y-6">
          {result ? (
            <ContentModerationReport
              result={result}
              imagePreview={imagePreview}
              imageFile={selectedFile}
              imageDimensions={imageDimensions}
              originalOcrText={originalOcrText}
              ocrStatus={ocrStatus}
              textUsed={captionText}
              onAnalyzeAnother={handleClear}
            />
          ) : (
            /* Empty State */
            <div className="bg-white rounded-3xl p-8 border border-slate-200/80 shadow-xs text-center space-y-4">
              <div className="w-14 h-14 mx-auto rounded-3xl bg-slate-50 border border-slate-200/80 flex items-center justify-center text-slate-400">
                <ShieldCheck className="w-7 h-7 text-slate-400" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-800">
                  Awaiting Visual / Multimodal Input
                </h3>
                <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto leading-relaxed">
                  Upload an image to automatically extract text via OCR, or proceed with image-only analysis. Click "Analyze Content" to generate the detailed Content Moderation Report.
                </p>
              </div>

              <div className="pt-2 border-t border-slate-100 text-left text-xs text-slate-600 space-y-2">
                <div className="font-semibold text-slate-700 text-[11px] uppercase tracking-wider">
                  What your report includes:
                </div>
                <div className="flex items-center space-x-2 text-slate-500 text-[11px]">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                  <span>Primary binary classification with empirical threshold (τ* = 0.41)</span>
                </div>
                <div className="flex items-center space-x-2 text-slate-500 text-[11px]">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                  <span>Probability distribution and decision confidence scores</span>
                </div>
                <div className="flex items-center space-x-2 text-slate-500 text-[11px]">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                  <span>"Why This Result?" evidence synthesis and modality attention weights</span>
                </div>
                <div className="flex items-center space-x-2 text-slate-500 text-[11px]">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                  <span>Self-contained printable HTML / PDF moderation report download</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
