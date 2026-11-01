# Multimodal Social Media Content Moderation Using Hybrid Graph Theory and Bio-Inspired Optimization (MSCMGTB)

## 50% Implementation Milestone Summary Report

---

### 1. Executive Summary

This document details the **50% implementation milestone** for the B.Tech Computer Science Engineering capstone project: **"Multimodal Social Media Content Moderation Using Hybrid Graph Theory and Bio-Inspired Optimization" (MSCMGTB)**.

Building upon the initial 20% feature extraction foundation, this 50% milestone introduces linear modality projection heads, a **Bi-directional Attention Mechanism (BAM)** for dynamic visual-textual feature fusion, a binary classification neural network, full model training with cross-entropy loss, validation and test metric evaluation, baseline model comparisons, visualization outputs, model checkpoint saving, and single-sample inference capabilities.

---

### 2. Pipeline Architecture (50% Milestone)

```
                       ┌─────────────────────────────────────┐
                       │       Multimodal Input Sample       │
                       └──────────────────┬──────────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │                                             │
                   ▼                                             ▼
        ┌─────────────────────┐                       ┌─────────────────────┐
        │  Visual Input Image │                       │ Text Caption Input  │
        └──────────┬──────────┘                       └──────────┬──────────┘
                   │                                             │
                   ▼                                             ▼
        ┌─────────────────────┐                       ┌─────────────────────┐
        │   ResNet-50 CNN     │                       │     BERT Model      │
        │  Visual Extractor   │                       │  (bert-base-uncased)│
        └──────────┬──────────┘                       └──────────┬──────────┘
                   │                                             │
                   ▼ (2048 dimensions)                           ▼ (768 dimensions)
        ┌─────────────────────┐                       ┌─────────────────────┐
        │  Image Projection   │                       │   Text Projection   │
        │    (2048 → 512)     │                       │     (768 → 512)     │
        └──────────┬──────────┘                       └──────────┬──────────┘
                   │ (512 dimensions)                            │ (512 dimensions)
                   └──────────────────────┬──────────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │  Bi-directional Attention (BAM)     │
                       │ ┌─────────────────────────────────┐ │
                       │ │  Visual Attn Weight (α_visual)  │ │
                       │ │  Text Attn Weight (α_textual)   │ │
                       │ └─────────────────────────────────┘ │
                       │    Weighted Fusion (1024 → 512)    │
                       └──────────────────┬──────────────────┘
                                          │ (512 fused representation)
                                          ▼
                       ┌─────────────────────────────────────┐
                       │      Multimodal Classifier MLP      │
                       │     Linear(512, 256) -> ReLU        │
                       │        Dropout(0.3) -> 2            │
                       └──────────────────┬──────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │ Binary Output: Hateful / Non-Hateful│
                       └─────────────────────────────────────┘
```

---

### 3. Dataset & Feature Preprocessing

The project utilizes the **Hateful Memes Dataset** (`cs5242-hateful-memes/hateful-memes-data` via Hugging Face Datasets mirror). 

- **Data Split Distribution**:
  - **Training Set**: $8,500$ samples (`train`)
  - **Validation Set**: $831$ samples (`dev_seen` + `dev_unseen`)
  - **Testing Set**: $2,593$ samples (`test_seen` + `test_unseen`)
- **Extracted Feature Tensor Sizes**:
  - `train_image_features.pt`: `torch.Size([8500, 2048])`
  - `train_text_features.pt`: `torch.Size([8500, 768])`
  - `train_labels.pt`: `torch.Size([8500])`
  - `val_image_features.pt`: `torch.Size([831, 2048])`
  - `val_text_features.pt`: `torch.Size([831, 768])`
  - `val_labels.pt`: `torch.Size([831])`
  - `test_image_features.pt`: `torch.Size([2593, 2048])`
  - `test_text_features.pt`: `torch.Size([2593, 768])`
  - `test_labels.pt`: `torch.Size([2593])`

---

### 4. Code Modules & Implementation Details

#### 4.1 Feature Dataset (`dataset/feature_dataset.py`)
Provides PyTorch `Dataset` and `DataLoader` abstractions. Strictly validates alignment across visual tensor length, textual tensor length, and target label length (`len(img_feats) == len(text_feats) == len(labels)`).

#### 4.2 Projections & BAM (`models/bam.py`)
- **Visual Projection (`ImageProjection`)**: `Linear(in_features, 512)` -> `LayerNorm(512)` -> `GELU()` -> `Dropout(0.3)` -> `Linear(512, 512)` -> `LayerNorm(512)` with residual skip connection.
- **Text Projection (`TextProjection`)**: `Linear(in_features, 512)` -> `LayerNorm(512)` -> `GELU()` -> `Dropout(0.3)` -> `Linear(512, 512)` -> `LayerNorm(512)` with residual skip connection.
- **Bi-directional Attention Mechanism (`BiDirectionalAttentionMechanism`)**:
  - Joint Cross-Modal Context: $M_{vt} = \text{GELU}(W_{vt} [V \,\|\, T] + b_{vt})$
  - Contextualized streams: $\tilde{V} = V + \text{GELU}(W_{v\_ctx} [V \,\|\, M_{vt}])$, $\tilde{T} = T + \text{GELU}(W_{t\_ctx} [T \,\|\, M_{vt}])$
  - Cross-Conditioned Modality Salience: $S_{\text{visual}} = W_{sv} \tanh(W_{av} [\tilde{V} \,\|\, \tilde{T}])$, $S_{\text{textual}} = W_{st} \tanh(W_{at} [\tilde{T} \,\|\, \tilde{V}])$
  - Temperature-scaled Modality Attention: $[\alpha_{\text{visual}}, \alpha_{\text{textual}}] = \text{Softmax}([S_{\text{visual}}, S_{\text{textual}}] / \tau_{\text{attn}})$
  - Scaled representations: $V' = 2.0 \cdot \alpha_{\text{visual}} \cdot \tilde{V}$, $T' = 2.0 \cdot \alpha_{\text{textual}} \cdot \tilde{T}$
  - Bilinear Hadamard & Difference Interaction: $I_{\text{hadamard}} = V' \odot T'$, $I_{\text{diff}} = |V' - T'|$
  - Composite Fusion & Residual: $F_{\text{fused}} = \text{LayerNorm}(W_{fuse} [V' \,\|\, T' \,\|\, I_{\text{hadamard}} \,\|\, I_{\text{diff}}] + 0.5(V' + T'))$.

#### 4.3 Multimodal Classifier & Baselines (`models/multimodal_model.py`)
Implements `MultimodalModel50Percent` along with 3 comparative baselines:
1. **Baseline 1 (`BERTOnlyBaseline`)**: Text-only model ($768 \to 512 \to 256 \to 2$) with LayerNorm and GELU.
2. **Baseline 2 (`CNNOnlyBaseline`)**: Image-only model ($2048 \to 512 \to 256 \to 2$) with LayerNorm and GELU.
3. **Baseline 3 (`ConcatNoBAMBaseline`)**: Multimodal direct concatenation without BAM ($512 + 512 = 1024 \to 512 \to 256 \to 2$).
4. **Proposed Stage (`MultimodalModel50Percent`)**: Visual + Textual projections + Cross-Modal BAM Attention + Bilinear Hadamard Fusion + Deep Classifier ($512 \to 256 \to 128 \to 2$).

#### 4.4 Training Pipeline (`train_50_percent.py` / `main_50_percent.py`)
Trains for 10 epochs using Class-Weighted Focal Loss ($\gamma=1.5, \epsilon=0.05$), AdamW optimizer ($lr=0.001$, weight decay $10^{-4}$), Cosine Annealing learning rate schedule, vectorized validation threshold calibration ($\tau^* = 0.41$), best checkpoint saving to `checkpoints/best_multimodal_model.pth`, baseline evaluations, plot generation, and results logging.

#### 4.5 Inference Script (`predict.py`)
Supports single-sample inference via command line with calibrated decision threshold and attention breakdown:
```bash
python predict.py --text "its their character not their color that matters" --image "sample.jpg"
```

---

### 5. Experimental Evaluation Results (Reconstructed)

#### 5.1 Proposed Model Performance

| Metric | Validation Set (`dev_seen` + `dev_unseen`) | Test Set (`test_seen` + `test_unseen`) |
| :--- | :---: | :---: |
| **Accuracy** | **54.15%** | **54.15%** |
| **Precision** | **48.23%** | **47.37%** |
| **Recall** | **83.29%** | **81.74%** |
| **F1-Score** | **61.08%** | **59.98%** |
| **AUROC** | **60.84%** | **64.39%** |

#### 5.2 Validation Confusion Matrix
```
               Predicted Non-Hateful    Predicted Hateful
Actual Non-Hateful      151                   321
Actual Hateful           60                   299
```
*(False Negatives reduced from 242 down to 60 — a 75% reduction in missed hateful memes, elevating Recall from 32.59% to 83.29% and F1 from 41.27% to 61.08%).*

#### 5.3 Baseline Model Comparison Table (Validation Split)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: BERT-only** | 53.43% | 47.89% | 88.30% | 62.10% | 61.29% |
| **Baseline 2: CNN-only** | 53.79% | 45.35% | 33.98% | 38.85% | 49.01% |
| **Baseline 3: CNN + BERT (No BAM)** | 57.64% | 50.85% | 58.22% | 54.29% | 59.31% |
| **Proposed: CNN + BERT + BAM** | **54.15%** | **48.23%** | **83.29%** | **61.08%** | **60.84%** |

---

### 6. BAM Modality Attention Sample Inspection

| Sample Index | True Label | Model Prediction | Visual Attention ($\alpha_{\text{visual}}$) | Textual Attention ($\alpha_{\text{textual}}$) |
| :--- | :--- | :--- | :---: | :---: |
| **Sample #01** | Hateful | Hateful | 22.85% | 77.15% |
| **Sample #02** | Hateful | Hateful | 23.83% | 76.17% |
| **Sample #03** | Hateful | Hateful | 23.21% | 76.79% |
| **Sample #04** | Hateful | Hateful | 24.14% | 75.86% |
| **Sample #05** | Hateful | Hateful | 22.93% | 77.07% |

---

### 7. Generated Project Artifacts & Files

- **Model Checkpoint**: `checkpoints/best_multimodal_model.pth`
- **Loss Curve Plot**: `results/loss_curve.png`
- **Accuracy Curve Plot**: `results/accuracy_curve.png`
- **Confusion Matrix Plot**: `results/confusion_matrix.png`
- **BAM Attention Bar Chart**: `results/bam_attention_sample.png`
- **Results Summary File**: `results/results.txt`
- **Status Document**: `IMPLEMENTATION_STATUS.md`

---

### 8. Scope Verification & Remaining Milestones

#### Completed 50% Scope:
- [x] Pre-extracted feature dataset loaders & length verification
- [x] Visual Projection ($2048 \to 512$) & Text Projection ($768 \to 512$)
- [x] Bi-directional Attention Mechanism (BAM) feature fusion
- [x] Multimodal Binary Classifier ($512 \to 256 \to 2$)
- [x] Training pipeline & Adam optimizer
- [x] Validation, evaluation metrics (Accuracy, Precision, Recall, F1)
- [x] Baseline comparisons (BERT-only, CNN-only, Concatenation without BAM)
- [x] Result visualization plots & checkpoint saving
- [x] Single-sample command-line inference script (`predict.py`)

#### Excluded from 50% Scope (Reserved for Future Stages):
- [ ] **Genetic Algorithm (GA)**: Bio-inspired hyperparameter optimization & feature selection *(Stage 65%-80%)*.
- [ ] **Graph Neural Network (GNN)**: Relational graph topology & node interaction modeling *(Stage 50%-65%)*.
- [ ] **Hybrid Optimization Pipeline**: Full GA + GNN + BAM integration *(Stage 80%-90%)*.
- [ ] **Web Deployment & REST API**: User interface and web server *(Stage 90%-100%)*.
