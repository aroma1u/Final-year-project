import { HealthStatus, ModelInfo, PredictionResult, SampleInput, OCRExtractResponse } from '../types';

const API_BASE = ''; // Uses Vite proxy in development or relative path

export const apiService = {
  async getHealth(): Promise<HealthStatus> {
    try {
      const response = await fetch(`${API_BASE}/api/health`, {
        headers: { 'Accept': 'application/json' },
      });
      if (!response.ok) {
        throw new Error(`Health check returned status ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      return {
        status: 'offline',
        model_loaded: false,
        device: 'unknown',
        version: '1.0.0',
        architecture: 'ResNet50 + BERT + BAM',
        error_message: error instanceof Error ? error.message : 'Backend unreachable',
      };
    }
  },

  async getModelInfo(): Promise<ModelInfo | null> {
    try {
      const response = await fetch(`${API_BASE}/api/model-info`);
      if (!response.ok) {
        throw new Error(`Failed to load model info: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.warn('[apiService] Could not fetch model info:', error);
      return null;
    }
  },

  async getSamples(): Promise<SampleInput[]> {
    try {
      const response = await fetch(`${API_BASE}/api/samples`);
      if (!response.ok) {
        throw new Error(`Failed to load samples: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.warn('[apiService] Could not fetch samples, using fallbacks:', error);
      return [
        {
          id: 'sample-benign',
          title: 'Sample A: Benign Social Commentary',
          text: "it's their character not their color that matters",
          image_url: '/static/samples/sample_benign.jpg',
          category: 'Demonstration (Benign Intent)',
        },
        {
          id: 'sample-irony',
          title: 'Sample B: Multimodal Juxtaposition Analysis',
          text: 'look at this sophisticated creature invading our country',
          image_url: '/static/samples/sample_irony.jpg',
          category: 'Demonstration (Multimodal Synergistic Risk)',
        },
      ];
    }
  },

  async predictContent(imageFile: File, text: string): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('text', text);

    const response = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorMessage = `Server error (${response.status})`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMessage = errorData.detail;
        }
      } catch {
        // Fallback to response text if not JSON
        const textErr = await response.text();
        if (textErr) errorMessage = textErr;
      }
      throw new Error(errorMessage);
    }

    return await response.json();
  },

  async extractText(imageFile: File): Promise<OCRExtractResponse> {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${API_BASE}/api/extract-text`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorMessage = `Server error (${response.status})`;
      try {
        const errorData = await response.json();
        if (errorData.detail) errorMessage = errorData.detail;
      } catch {
        const textErr = await response.text();
        if (textErr) errorMessage = textErr;
      }
      throw new Error(errorMessage);
    }

    return await response.json();
  },
};

