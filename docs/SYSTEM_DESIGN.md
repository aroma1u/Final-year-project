# MemeGuard AI (MSCMGTB) — System Architecture & Design Document

## 1. System Overview
**MemeGuard AI** is a production-grade full-stack research demonstration platform for **"Multimodal Social Media Content Moderation Using Hybrid Graph Theory and Bio-Inspired Optimization" (MSCMGTB)**. It accepts multimodal inputs (an image and its associated text caption) and classifies them as **Hateful** or **Non-Hateful** using deep multimodal fusion.

---

## 2. Implemented vs. Planned Components (Milestone Realism)

| Component | Status | Description |
| :--- | :---: | :--- |
| **ResNet-50 Visual Extractor** | **Implemented** | 2048-dimensional global visual feature representation |
| **BERT Textual Extractor** | **Implemented** | `bert-base-uncased` 768-dimensional contextual text representation |
| **Modality Projections** | **Implemented** | Residual `Linear + LayerNorm + GELU` projection heads ($2048 \to 512, 768 \to 512$) |
| **Bi-directional Attention (BAM)** | **Implemented** | Cross-modal conditioned attention with temperature scaling and Hadamard bilinear fusion |
| **Multimodal Classifier MLP** | **Implemented** | Deep residual classifier ($512 \to 256 \to 128 \to 2$) with calibrated threshold $\tau^* = 0.41$ |
| **EasyOCR Text Extractor** | **Implemented** | CRAFT text detection and CRNN character recognition with human-in-the-loop editing |
| **FastAPI Inference Service** | **Implemented** | Production async backend serving real-time predictions |
| **React + Vite Modern Dashboard** | **Implemented** | 5-page responsive web application with Lucide icons and Tailwind CSS |
| **Graph Neural Network (GNN)** | *Planned (Stage 80%-90%)* | Relational graph topology and social context modeling |
| **Genetic Algorithm (GA)** | *Planned (Stage 85%-95%)* | Bio-inspired hyperparameter optimization and feature mask selection |
| **Hybrid GA + GNN Pipeline** | *Planned (Stage 90%-100%)* | Complete end-to-end bio-inspired graph multimodal moderation |

---

## 3. High-Level Architecture Diagram

```
                        ┌────────────────────────────────────────┐
                        │        Client Browser (React SPA)       │
                        │    - Content Moderation Studio         │
                        │    - Architecture & Attention Viewer   │
                        │    - Local History & About Project     │
                        └───────────────────┬────────────────────┘
                                            │ multipart/form-data
                                            ▼
                        ┌────────────────────────────────────────┐
                        │          FastAPI Backend Service       │
                        │    - CORS Middleware & Upload Guard    │
                        │    - GET  /api/health                  │
                        │    - GET  /api/model-info              │
                        │    - POST /api/predict                 │
                        └───────────────────┬────────────────────┘
                                            │
                                            ▼
                        ┌────────────────────────────────────────┐
                        │        Inference Service Engine        │
                        │    - Image Normalization (224x224)     │
                        │    - BERT Tokenizer (max_length=128)   │
                        │    - Device Selector (CUDA / CPU)      │
                        └───────────────────┬────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │                                             │
                     ▼                                             ▼
          ┌─────────────────────┐                       ┌─────────────────────┐
          │   ResNet-50 CNN     │                       │     BERT Model      │
          │  Visual Extractor   │                       │  (bert-base-uncased)│
          └──────────┬──────────┘                       └──────────┬──────────┘
                     │ (2048-d)                                    │ (768-d)
                     ▼                                             ▼
          ┌─────────────────────┐                       ┌─────────────────────┐
          │  Image Projection   │                       │   Text Projection   │
          │  (2048 -> 512, LN)  │                       │  (768 -> 512, LN)   │
          └──────────┬──────────┘                       └──────────┬──────────┘
                     │ (512-d)                                     │ (512-d)
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
                         ┌─────────────────────────────────────┐
                         │   Bi-directional Attention (BAM)    │
                         │ ┌─────────────────────────────────┐ │
                         │ │  Cross-Modal Conditioning (M_vt)│ │
                         │ │  Modality Attention [α_v, α_t]  │ │
                         │ │  Hadamard Interaction (V' ⊙ T') │ │
                         │ │  Absolute Difference |V' - T'|  │ │
                         │ └─────────────────────────────────┘ │
                         │    Fused Representation (512-d)     │
                         └──────────────────┬──────────────────┘
                                            │
                                            ▼
                         ┌─────────────────────────────────────┐
                         │      Deep Multimodal Classifier     │
                         │   Linear(512, 256) -> LayerNorm     │
                         │   GELU -> Linear(256, 128) -> 2     │
                         │   Calibrated Threshold (τ* = 0.41)  │
                         └──────────────────┬──────────────────┘
                                            │
                                            ▼
                         ┌─────────────────────────────────────┐
                         │ Prediction: Hateful / Non-Hateful   │
                         │ Confidence % & Attention Breakdown  │
                         └─────────────────────────────────────┘
```

---

## 4. API Endpoints Specification

### 4.1 Health Check: `GET /api/health`
Returns runtime status and model availability:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "version": "1.0.0",
  "architecture": "ResNet50 + BERT + BAM"
}
```

### 4.2 Model Info: `GET /api/model-info`
Returns architectural metadata, training specifications, and calibrated operating parameters:
```json
{
  "model_name": "MemeGuard AI Multimodal Model (ResNet50 + BERT + BAM)",
  "checkpoint_path": "checkpoints/best_multimodal_model.pth",
  "visual_backbone": "ResNet-50 (2048-d)",
  "textual_backbone": "bert-base-uncased (768-d)",
  "fusion_mechanism": "Bi-directional Attention Mechanism (BAM) with Hadamard Interaction",
  "calibrated_threshold": 0.41,
  "metrics": {
    "validation_accuracy": 54.15,
    "validation_recall": 83.29,
    "validation_f1": 61.08,
    "test_auroc": 64.39
  },
  "planned_stages": [
    "Graph Neural Network (GNN) Relational Modeling (Stage 80%-90%)",
    "Genetic Algorithm (GA) Bio-Inspired Feature Mask Optimization (Stage 85%-95%)",
    "Full Hybrid GA + GNN Pipeline Integration & Capstone Defense (Stage 90%-100%)"
  ]
}
```

### 4.3 Prediction: `POST /api/predict`
Accepts `multipart/form-data`:
- `image`: Image binary file (`.png`, `.jpg`, `.jpeg`, `.webp`, max 10MB)
- `text`: Associated meme caption string (max 2000 chars)

Response payload:
```json
{
  "success": true,
  "label": "Hateful Content Detected",
  "predicted_class": 1,
  "probability_hateful": 61.64,
  "probability_non_hateful": 38.36,
  "confidence": 61.64,
  "calibrated_threshold": 0.41,
  "attention": {
    "visual": 23.18,
    "textual": 76.82
  },
  "model_name": "ResNet50 + BERT + BAM",
  "timestamp": "2026-10-24T22:45:00Z"
}
```

---

## 5. UI/UX Architecture & Pages
- **Theme**: Refined light theme default (`#f8fafc` background, `#ffffff` card surface, `#312e81` deep indigo primary, `#0d9488` teal accent, `#0f172a` slate typography).
- **Page Structure**:
  1. **Overview Dashboard**: Hero, 4-step workflow, summary cards, ethical AI disclosure.
  2. **Content Moderation Studio**: Drag-and-drop file upload with visual preview, automated OCR extraction indicator, editable extracted text area, sample loaders, real-time prediction card with probability bars and BAM modality attention comparison.
  3. **Analysis History**: Client-side session log of all analyzed samples with inspectable metadata and clear-history control.
  4. **System Architecture**: Interactive responsive visual component pipeline highlighting implemented vs. planned stages.
  5. **About Project**: Capstone academic background, dataset specifics, limitations, and future work.

---

## 6. Optical Character Recognition (OCR) Integration Architecture

### 6.1 Purpose & Role: OCR vs. ResNet-50 Visual Features
In social media memes, text is baked directly into the visual pixel grid. It is critical to differentiate the roles of the two feature extractors:
- **ResNet-50 (Visual Backbone):** Extracts high-level visual semantic features (facial expressions, background imagery, objects, symbols, textures) into a 2048-dimensional dense representation. ResNet-50 is unaware of semantic grammar, vocabulary, or word order.
- **EasyOCR (Character Recognition Engine):** Uses a deep CRAFT text detector to locate bounding boxes of words and a CRNN (Convolutional Recurrent Neural Network) to decode pixel characters into structured English text strings.

### 6.2 Data Handoff: OCR to BERT Contextual Pipeline
```
  Uploaded Meme Image
          │
          ├──> ResNet-50 CNN (2048-d Visual Features)
          │
          └──> EasyOCR Engine (POST /api/extract-text)
                    │
                    ▼
          Extracted Text String
                    │
                    ▼
          Editable UI Textarea (User Review / Correction)
                    │
                    ▼
          bert-base-uncased Tokenizer (Max 128 tokens)
                    │
                    ▼
          BERT Transformer (768-d Contextual Text Features)
                    │
                    ▼
          Bi-directional Attention Mechanism (BAM Fusion)
```

### 6.3 State Preservation & Human-in-the-Loop Moderation
- When an image is uploaded or dropped into the Content Moderation Studio, OCR executes automatically in the background.
- The extracted text populates the editable textarea while keeping the image preview visible.
- If the user edits the text (e.g. correcting a rare slang spelling or OCR misread), their manual edits are strictly preserved and never silently overwritten.
- When "Analyze Content" is triggered, the final text from the textarea is submitted alongside the image to `/api/predict`.

### 6.4 OCR Endpoint Specification: `POST /api/extract-text`
Accepts `multipart/form-data`:
- `image`: Image binary file (`.png`, `.jpg`, `.jpeg`, `.webp`, max 10MB)

Response:
```json
{
  "success": true,
  "extracted_text": "[MemeGuard AI Sample - Diversity Meme]",
  "text_detected": true,
  "raw_lines": [
    "[MemeGuard AI Sample -",
    "Diversity Meme]"
  ],
  "message": "Text extraction completed successfully."
}
```

### 6.5 OCR Limitations & Mitigations
- **Artistic / Distorted Typography:** Heavily warped word art or extreme graffiti fonts can lead to character substitution. *Mitigation:* The frontend allows human-in-the-loop editing.
- **Low Contrast / Watermarks:** Faint gray text on busy backgrounds may have low detection confidence. *Mitigation:* The system detects `text_detected: false` and prompts the user for manual entry without blocking analysis.
- **Resolution Optimization:** Ultra-high-resolution uploads (>1600px) are scaled proportionally prior to CRAFT detection to prevent CPU memory bottlenecks while preserving character legibility.

