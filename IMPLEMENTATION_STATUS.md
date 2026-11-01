# MSCMGTB Implementation Status

**Project Title:** Multimodal Social Media Content Moderation Using Hybrid Graph Theory and Bio-Inspired Optimization (MSCMGTB)

---

## 1. Completed 20% Milestone

The initial 20% stage successfully established the multimodal data loading and feature extraction pipeline:

- **Dataset Acquisition:** Integrated Hugging Face Datasets loader for the Hateful Memes dataset (`cs5242-hateful-memes/hateful-memes-data`).
- **Data Preprocessing:** Standardized visual transforms (224×224 resize, normalization) and textual tokenization (BERT `bert-base-uncased` tokenizer with padding/truncation).
- **CNN Visual Feature Extraction:** Extracted 2048-dimensional visual feature representations using a pre-trained ResNet-50 CNN backbone.
- **BERT Textual Feature Extraction:** Extracted 768-dimensional textual representations from the `[CLS]` token output of `bert-base-uncased`.
- **Feature Storage & Verification:** Verified alignment across visual features `[100, 2048]`, text features `[100, 768]`, and target labels `[100]`.

---

## 2. Completed 50% Milestone (Current Stage)

The 50% stage completes full feature extraction, multimodal attention fusion, model training, baseline comparisons, evaluation, and visualization:

- **Full Feature Preparation:** Extracted ResNet-50 visual (2048-d) and BERT textual (768-d) features across all dataset splits:
  - `train`: 8,500 samples
  - `validation` (`dev_seen` + `dev_unseen`): 831 samples
  - `testing` (`test_seen` + `test_unseen`): 2,593 samples
- **Feature Alignment & PyTorch Dataset:** Developed `FeatureDataset` class with strict length validation (`len(img_features) == len(text_features) == len(labels)`).
- **Visual Feature Projection (`ImageProjection`):** Reduced 2048-d visual features to 512 dimensions via `Linear(2048, 512)` -> `BatchNorm1d` -> `ReLU` -> `Dropout(0.3)`.
- **Text Feature Projection (`TextProjection`):** Reduced 768-d textual features to 512 dimensions via `Linear(768, 512)` -> `BatchNorm1d` -> `ReLU` -> `Dropout(0.3)`.
- **Bi-directional Attention Mechanism (`BAM`):** Designed and implemented BAM in `models/bam.py`. It computes dynamic softmax-normalized attention scores (`image_attention`, `text_attention`) to weight modality representations before concatenation (`1024 -> 512`).
- **Multimodal Classification Model:** Combined projection heads, BAM fusion, and a classification MLP (`512 -> 256 -> 2`) predicting binary Hateful / Non-Hateful classes.
- **Training Pipeline (`train_50_percent.py` / `main_50_percent.py`):** Configured 10-epoch training loop with CrossEntropyLoss and Adam optimizer.
- **Validation & Baseline Comparisons:** Evaluated proposed model against 3 baseline models on identical splits:
  1. Baseline 1: BERT-only
  2. Baseline 2: CNN-only
  3. Baseline 3: CNN + BERT without BAM (direct concatenation)
  4. Proposed: CNN + BERT + BAM
- **Model Checkpointing:** Saved best model state based on validation F1 score to `checkpoints/best_multimodal_model.pth`.
- **Performance Evaluation & Visualization:** Computed Accuracy, Precision, Recall, F1 score, Confusion Matrix, Training/Validation Loss & Accuracy curves (`results/loss_curve.png`, `results/accuracy_curve.png`), and BAM Attention weights visualization (`results/bam_attention_sample.png`).
- **Single Sample Inference (`predict.py`):** Standalone script supporting `--image` and `--text` command-line inputs to return predicted class, confidence %, and modality attention weights.

---

## 3. Architecture Overview (50% Stage)

```
Visual Input (Image)                      Textual Input (Caption)
        │                                            │
        ▼                                            ▼
ResNet-50 Encoder                            BERT Encoder
        │                                            │
        ▼                                            ▼
2048-d Visual Feature                         768-d Textual Feature
        │                                            │
        ▼                                            ▼
Visual Projection (2048 → 512)               Text Projection (768 → 512)
        │                                            │
        └─────────────────────┬──────────────────────┘
                              │
                              ▼
               Bi-directional Attention Mechanism (BAM)
                 ┌────────────┴────────────┐
                 ▼                         ▼
          Visual Weight (α_v)       Text Weight (α_t)
                 └────────────┬────────────┘
                              │
                              ▼
                  512-d Fused Representation
                              │
                              ▼
                 Classifier MLP (512 → 256 → 2)
                              │
                              ▼
                Prediction: Hateful / Non-Hateful
```

---

## 4. Not Yet Implemented (Future Milestones)

The following advanced components will be developed in subsequent stages and are **NOT** included in the 50% milestone:

- ✗ **Bio-Inspired Optimization (Genetic Algorithm / GA):** Feature selection and hyperparameter search optimization.
- ✗ **Relational Graph Neural Network (GNN):** Graph construction and relational interaction modeling between multimodal elements.
- ✗ **Hybrid Graph-Optimization Integration (50% to 80% stage):** Seamless combination of BAM representations with Graph topology and GA.
- ✗ **Final Web Application / API Deployment:** Production REST API or Web UI.
