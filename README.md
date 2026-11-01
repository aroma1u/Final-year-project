# MemeGuard AI: Multimodal Social Media Content Moderation (MSCMGTB)

> **B.Tech Final Year Computer Science & Engineering Capstone Project**  
> **Project Title:** Multimodal Social Media Content Moderation Using Hybrid Graph Theory and Bio-Inspired Optimization (**MSCMGTB**)  
> **Platform Name:** MemeGuard AI  
> **Current Stage:** 80% Project Implementation Milestone (Multimodal Inference + EasyOCR + Web Studio Deployed)  

---

## 1. System Overview

**MemeGuard AI** is an advanced artificial intelligence content moderation platform built to detect and analyze multimodal hateful content (internet memes, image-text pairs) where malice or irony emerges from the subtle interaction between visual and textual modalities.

A meme that appears benign in isolation (e.g., an innocent picture or a neutral phrase) can become toxic or derogatory when combined. MemeGuard AI addresses this challenge using deep cross-modal feature projection, bi-directional cross-attention mechanisms, and calibrated decision boundaries.

---

## 2. Multimodal Content Moderation Architecture (ResNet50 + BERT + BAM)

```
                    MULTIMODAL INPUT
                           |
             +-------------+-------------+
             |                           |
             v                           v
     Visual Modality             Textual Modality
     (Image: PNG/JPG)             (Caption / Text)
             |                           |
             v                           v
       ResNet-50 CNN             bert-base-uncased
      [N, 2048] features          [N, 768] features
             |                           |
             v                           v
     Visual Projection           Textual Projection
   (Linear 2048->512 + LN)     (Linear 768->512 + LN)
             |                           |
             +-------------+-------------+
                           |
                           v
         Bi-directional Attention Mechanism (BAM)
    Cross-modal conditioning: M_vt = [V_proj, T_proj]
     Learned Attention Weights: [alpha_v, alpha_t]
  Bilinear Interaction: [V', T', V' (x) T', |V' - T'|]
               Projection -> [N, 512]
                           |
                           v
                 Multimodal Classifier
           Linear(512->256) -> LN -> GELU
           Linear(256->128) -> LN -> GELU
                    Linear(128->2)
                           |
                           v
        Calibrated Decision Boundary (tau* = 0.41)
                           |
                           v
             Hateful / Non-Hateful Prediction
```

### Empirical Evaluation & Defect Baseline Elimination

In the original 50% milestone baseline, the classifier suffered from severe recall collapse (32.59% recall, missing 242 out of 359 hateful memes). Through the introduction of class-weighted Focal Loss ($\gamma = 1.5$), AdamW with cosine annealing, and calibrated decision thresholding ($\tau^* = 0.41$), the reconstructed model achieves:

| Metric | Baseline 50% Milestone | Reconstructed 50% Milestone | Improvement / Status |
| :--- | :--- | :--- | :--- |
| **Validation Accuracy** | 59.93% | **54.15%** | Calibrated boundary |
| **Validation Precision** | 56.25% | **48.23%** | Balanced trade-off |
| **Validation Recall (Sensitivity)** | **32.59%** | **83.29%** | **+50.70% (+155.6% relative)** |
| **Validation F1-Score** | 41.27% | **61.08%** | **+19.81% (+48.0% relative)** |
| **Test Set AUROC** | 58.12% | **64.39%** | **+6.27% discriminative power** |
| **Missed Hateful Memes (FN)** | 242 / 359 (67.4%) | **60 / 359 (16.7%)** | **75.2% reduction in false negatives** |

*Note: All metrics reflect real validation evaluation on the Facebook Hateful Memes dataset mirror; zero synthetic or hardcoded values are used.*

---

## 3. Technology Stack

### Frontend
- **Framework:** React 19 + TypeScript + Vite
- **Styling:** Tailwind CSS v4, Inter Typography
- **Icons:** Lucide React
- **Architecture:** Client-side state routing, polling health checks, local session history storage

### Backend & API
- **Framework:** Python FastAPI + Pydantic v2
- **ASGI Server:** Uvicorn
- **API Endpoints:** RESTful with CORS, multipart form uploads, typed JSON schemas

### Machine Learning & Inference
- **Framework:** PyTorch (TorchScript / CPU & CUDA compatible)
- **Computer Vision:** `torchvision.models.resnet50` (pretrained weights `DEFAULT`)
- **Optical Character Recognition (OCR):** `EasyOCR` (CRAFT detector + CRNN English recognizer)
- **Natural Language Processing:** Hugging Face `transformers` (`bert-base-uncased`)
- **Image Processing:** Pillow (PIL), NumPy

---

## 4. Quick Start & Execution Commands (Windows PowerShell)

### Prerequisites
- Python 3.10+ (Current runtime: Python 3.14)
- Node.js 18+ (Current runtime: Node.js v24.12.0, npm 11.6.2)

### Step 1: Clone / Open Project
```powershell
cd C:\mscgproject
```

### Step 2: Backend Setup & Execution
Install backend requirements (including EasyOCR):
```powershell
pip install -r requirements.txt
```

Start the FastAPI backend server:
```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
*The backend starts at `http://127.0.0.1:8000`.*
*Interactive API Swagger documentation is available at `http://127.0.0.1:8000/docs`.*

### Step 3: Frontend Setup & Execution
Open a second PowerShell terminal:
```powershell
cd C:\mscgproject\frontend
npm install
npm run dev
```
*The Vite development server runs at `http://localhost:5173/`.*
*Vite automatically proxies `/api` and `/static` requests to `http://127.0.0.1:8000`.*

---

## 5. REST API Specifications

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Returns backend health, model loaded status, device (`cpu`/`cuda`), and version |
| `GET` | `/api/model-info` | Returns model architecture parameters, calibrated threshold ($\tau^* = 0.41$), and metrics |
| `POST` | `/api/extract-text` | Extracts text embedded inside an uploaded meme image via EasyOCR (CRAFT + CRNN) |
| `POST` | `/api/predict` | Accepts multipart form upload (`image` binary, `text` caption) and returns prediction |
| `GET` | `/api/samples` | Returns local demonstration sample metadata for instant verification |


### Prediction API Response Example
```json
{
  "success": true,
  "label": "Hateful Content Detected",
  "predicted_class": 1,
  "probability_hateful": 49.92,
  "probability_non_hateful": 50.08,
  "confidence": 49.92,
  "calibrated_threshold": 0.41,
  "attention": {
    "visual": 39.95,
    "textual": 60.05
  },
  "model_name": "ResNet50 + BERT + BAM",
  "timestamp": "2026-10-24T17:30:47.267942+00:00"
}
```

---

## 6. Project Directory Structure

```
C:\mscgproject\
├── api\
│   ├── main.py                     # FastAPI application router & CORS setup
│   ├── inference_service.py        # Singleton PyTorch inference service (ResNet50 + BERT + BAM)
│   ├── schemas.py                  # Pydantic request and response schemas
│   ├── create_samples.py           # Verification sample generator
│   └── static\samples\             # Local demonstration images
├── checkpoints\
│   └── best_multimodal_model.pth   # Reconstructed 50% milestone PyTorch weights
├── dataset\
│   └── extract_gemini_embeddings.py# Modular Gemini Embedding-2-Flash script
├── docs\
│   └── SYSTEM_DESIGN.md            # Comprehensive system design document
├── features\
│   └── extract_features.py         # Batch feature extraction script
├── frontend\
│   ├── src\
│   │   ├── components\layout\      # Sidebar.tsx, Header.tsx
│   │   ├── pages\                  # Overview, Moderation, History, Architecture, About
│   │   ├── hooks\                  # useHealth.ts, useHistory.ts
│   │   ├── services\               # api.ts (Fetch API client)
│   │   ├── types\                  # TypeScript data interfaces
│   │   ├── App.tsx                 # Root router & layout
│   │   └── main.tsx                # React DOM entry
│   ├── package.json
│   └── vite.config.ts              # Vite configuration with API proxy
├── models\
│   ├── bam.py                      # Bi-directional Attention Mechanism & Projections
│   └── multimodal_model.py         # Multimodal classifier & baselines
├── results\                        # ROC curves, loss plots, and evaluation logs
├── train_50_percent.py             # Model training script with Focal Loss & AdamW
├── predict.py                      # Standalone CLI prediction script
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 7. Milestone Roadmap

| Milestone | Target Scope | Implementation Status |
| :--- | :--- | :--- |
| **0% - 20%** | Dataset ingestion, preprocessing, ResNet-50 visual extraction, BERT extraction | Completed |
| **20% - 50%** | BAM attention fusion, multimodal classifier, Focal Loss, Calibrated Thresholding | Completed |
| **50% - 80%** | EasyOCR text extraction, human-in-the-loop studio, FastAPI async inference, React 19 UI | **Completed & Tested (Current Milestone: 80%)** |
| **80% - 90%** | Graph Neural Network (GNN) for relational context modeling & entity interaction | Planned / In Development |
| **85% - 95%** | Genetic Algorithm (GA) for bio-inspired feature mask & hyperparameter optimization | Planned |
| **90% - 100%** | Full Hybrid GA + GNN end-to-end pipeline integration & faculty defense report | Planned |

---

## 8. Responsible AI Notice & Disclaimer

> **Responsible AI Notice:**  
> MemeGuard AI is a research demonstration system for academic capstone evaluation. Algorithmic content moderation systems may produce false positives or false negatives due to linguistic ambiguity, sarcasm, cultural nuances, or shifting internet slang. Model predictions should always be reviewed by a human moderator before any consequential administrative or moderation action is taken.
