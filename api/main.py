import os
import sys
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add root directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from api.schemas import HealthResponse, PredictionResponse, ModelInfoResponse, ModelMetrics, OCRExtractResponse
from api.inference_service import get_inference_service
from api.ocr_service import get_ocr_service

app = FastAPI(
    title="MemeGuard AI API",
    description="Backend API for Multimodal Social Media Content Moderation (MSCMGTB)",
    version="1.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample assets directory
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "api", "static", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "api", "static")), name="static")


@app.get("/")
def root():
    """Root endpoint providing service status and links to documentation and frontend."""
    return {
        "status": "online",
        "service": "MemeGuard AI API (MSCMGTB)",
        "documentation": "/docs",
        "health_check": "/api/health",
        "frontend_url": "http://localhost:5173"
    }


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Returns the operational status and model readiness of the MemeGuard AI engine."""
    service = get_inference_service()
    return HealthResponse(
        status="healthy" if service.model_loaded else "degraded",
        model_loaded=service.model_loaded,
        device=str(service.device),
        version="1.0.0",
        architecture="ResNet50 + BERT + BAM (Reconstructed)",
        error_message=service.error_message
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
def model_info():
    """Returns architectural metadata, empirical verification metrics, and operating thresholds."""
    service = get_inference_service()
    return ModelInfoResponse(
        model_name="MemeGuard AI Multimodal Model (ResNet50 + BERT + BAM)",
        checkpoint_path="checkpoints/best_multimodal_model.pth",
        visual_backbone="ResNet-50 (2048-dimensional visual feature representation)",
        textual_backbone="bert-base-uncased (768-dimensional contextual text representation)",
        fusion_mechanism="Bi-directional Attention Mechanism (BAM) with Cross-Modal Conditioning & Hadamard Bilinear Interaction",
        calibrated_threshold=service.optimal_threshold,
        metrics=ModelMetrics(
            validation_accuracy=54.15,
            validation_recall=83.29,
            validation_f1=61.08,
            test_auroc=64.39
        ),
        planned_stages=[
            "Graph Neural Network (GNN) Relational Context Modeling",
            "Genetic Algorithm (GA) Bio-Inspired Feature Mask Optimization",
            "Hybrid GA + GNN End-to-End Moderation Pipeline & Final Defense"
        ]
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_content(
    image: UploadFile = File(..., description="Meme image file (PNG, JPG, JPEG, WEBP)"),
    text: str = Form("", description="Optional text caption or social media message")
):
    """
    Analyzes submitted image and associated text through the trained multimodal pipeline.
    Text is optional; image-only and multimodal analysis are both supported.
    """
    service = get_inference_service()
    if not service.model_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Inference model is unavailable: {service.error_message}"
        )

    # 1. Validate File Extension
    filename = image.filename or ""
    allowed_exts = {".png", ".jpg", ".jpeg", ".webp"}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(allowed_exts))}"
        )

    # 2. Read and Validate File Size (Max 10MB)
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size exceeds maximum limit of 10MB.")

    # 3. Validate Text (Optional)
    clean_text = text.strip() if text else ""
    if len(clean_text) > 2000:
        raise HTTPException(status_code=400, detail="Text exceeds maximum limit of 2000 characters.")

    # 4. Perform Real Model Inference
    try:
        result = service.predict(image_bytes, clean_text)
        return PredictionResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {e}")


@app.post("/api/extract-text", response_model=OCRExtractResponse)
async def extract_text_from_image(
    image: UploadFile = File(..., description="Meme image file to perform OCR on")
):
    """
    Extracts text embedded inside a meme image using the deep-learning EasyOCR engine.
    Returns cleaned text, raw lines, and text detection indicator.
    """
    filename = image.filename or ""
    allowed_exts = {".png", ".jpg", ".jpeg", ".webp"}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(allowed_exts))}"
        )

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size exceeds maximum limit of 10MB.")

    ocr_service = get_ocr_service()
    if not ocr_service.is_available:
        raise HTTPException(
            status_code=503,
            detail=f"OCR service is currently unavailable: {ocr_service.error_message}"
        )

    try:
        result = ocr_service.extract_text(image_bytes)
        return OCRExtractResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR extraction failure: {e}")


@app.get("/api/samples")
def get_sample_inputs():
    """
    Returns verified local demonstration samples for faculty presentation.
    """
    return [
        {
            "id": "sample-benign",
            "title": "Sample A: Benign Social Commentary",
            "text": "it's their character not their color that matters",
            "image_url": "/static/samples/sample_benign.jpg",
            "category": "Demonstration (Benign Intent)"
        },
        {
            "id": "sample-irony",
            "title": "Sample B: Multimodal Juxtaposition Analysis",
            "text": "look at this sophisticated creature invading our country",
            "image_url": "/static/samples/sample_irony.jpg",
            "category": "Demonstration (Multimodal Synergistic Risk)"
        }
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
