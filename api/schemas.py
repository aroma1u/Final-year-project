from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    device: str
    version: str = "1.0.0"
    architecture: str
    error_message: Optional[str] = None

class ModalityAttention(BaseModel):
    visual: float = Field(..., description="Percentage weight assigned to visual modality")
    textual: float = Field(..., description="Percentage weight assigned to textual modality")

class ClassProbabilities(BaseModel):
    non_hateful: float = Field(..., description="Probability of non-hateful content (%)")
    hateful: float = Field(..., description="Probability of hateful content (%)")

class PredictionResponse(BaseModel):
    success: bool
    label: str = Field(..., description="'Hateful Content Detected' or 'No Hateful Content Detected'")
    predicted_class: int = Field(..., description="0 for Non-Hateful, 1 for Hateful")
    probability_hateful: float
    probability_non_hateful: float
    confidence: float
    calibrated_threshold: float
    attention: ModalityAttention
    model_name: str
    timestamp: str

class ModelMetrics(BaseModel):
    validation_accuracy: float
    validation_recall: float
    validation_f1: float
    test_auroc: float

class ModelInfoResponse(BaseModel):
    model_name: str
    checkpoint_path: str
    visual_backbone: str
    textual_backbone: str
    fusion_mechanism: str
    calibrated_threshold: float
    metrics: ModelMetrics
    planned_stages: List[str]

class OCRExtractResponse(BaseModel):
    success: bool
    extracted_text: str = Field(..., description="Text extracted from image via OCR")
    text_detected: bool = Field(..., description="Whether readable text was found in image")
    raw_lines: List[str] = Field(default_factory=list, description="Extracted text split line by line")
    message: str = Field(..., description="Status description of the OCR run")

