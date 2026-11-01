export type TabType = 'overview' | 'moderation' | 'history' | 'about';

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'offline';
  model_loaded: boolean;
  device: string;
  version: string;
  architecture: string;
  error_message?: string | null;
}

export interface ModalityAttention {
  visual: number;
  textual: number;
}

export interface ClassProbabilities {
  non_hateful: number;
  hateful: number;
}

export interface PredictionResult {
  success: boolean;
  label: 'Hateful Content Detected' | 'No Hateful Content Detected' | string;
  predicted_class: 0 | 1;
  probability_hateful: number;
  probability_non_hateful: number;
  confidence: number;
  calibrated_threshold: number;
  attention: ModalityAttention;
  model_name: string;
  timestamp: string;
  // Metadata for history & detailed moderation report
  image_preview_url?: string;
  input_text?: string;
  id?: string;
  original_ocr_text?: string;
  text_used?: string;
  is_empty_text?: boolean;
  image_filename?: string;
  image_dimensions?: string;
  image_size?: string;
  image_type?: string;
  ocr_status?: OCRStatus;
}

export interface ModelInfo {
  model_name: string;
  checkpoint_path: string;
  visual_backbone: string;
  textual_backbone: string;
  fusion_mechanism: string;
  calibrated_threshold: number;
  metrics: {
    validation_accuracy: number;
    validation_recall: number;
    validation_f1: number;
    test_auroc: number;
  };
  planned_stages: string[];
}

export interface SampleInput {
  id: string;
  title: string;
  text: string;
  image_url: string;
  category: string;
}

export type OCRStatus = 'waiting' | 'extracting' | 'extracted' | 'no_text' | 'failed';

export interface OCRExtractResponse {
  success: boolean;
  extracted_text: string;
  text_detected: boolean;
  raw_lines: string[];
  message: string;
}
