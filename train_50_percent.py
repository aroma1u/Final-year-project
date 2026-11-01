import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from tqdm import tqdm

from dataset.feature_dataset import FeatureDataset, get_feature_dataloader
from models.multimodal_model import (
    MultimodalModel50Percent,
    BERTOnlyBaseline,
    CNNOnlyBaseline,
    ConcatNoBAMBaseline
)

class FocalLoss(nn.Module):
    """
    Class-weighted Focal Loss with Label Smoothing for mitigating False Negative recall collapse.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=1.5, label_smoothing=0.05, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Class weight tensor [w0, w1]
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        # Apply label smoothing to one-hot ground truth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Modulating factor: (1 - p_t)^gamma
        pt = torch.sum(probs * smooth_targets, dim=1)
        focal_weight = (1.0 - pt) ** self.gamma

        loss = -torch.sum(smooth_targets * log_probs, dim=1) * focal_weight

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            loss = loss * alpha_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def evaluate_model(model, dataloader, device, threshold=0.50, criterion=None):
    """
    Evaluates a model on a given dataloader at a given probability threshold.
    Returns: loss, accuracy, precision, recall, f1, auroc, confusion_matrix, predictions, probabilities, attention_weights
    """
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_probs = []
    all_labels = []
    all_img_attn = []
    all_txt_attn = []

    with torch.no_grad():
        for b_img, b_txt, b_lbl in dataloader:
            b_img, b_txt, b_lbl = b_img.to(device), b_txt.to(device), b_lbl.to(device)

            logits, _, img_attn, txt_attn = model(b_img, b_txt)
            loss = criterion(logits, b_lbl)
            total_loss += loss.item() * b_img.size(0)

            probs = F.softmax(logits, dim=1)[:, 1] # Probability of Hateful (Class 1)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(b_lbl.cpu().numpy().tolist())

            all_img_attn.extend(img_attn.cpu().view(-1).numpy().tolist())
            all_txt_attn.extend(txt_attn.cpu().view(-1).numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auroc = 0.50

    cm = confusion_matrix(all_labels, preds)

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auroc': auroc,
        'confusion_matrix': cm,
        'predictions': preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'img_attn': np.array(all_img_attn),
        'txt_attn': np.array(all_txt_attn),
        'threshold': threshold
    }


def find_optimal_threshold(model, dataloader, device, criterion, min_t=0.38, max_t=0.62, steps=25):
    """
    Vectorized threshold calibration on validation set: computes forward pass once,
    then sweeps thresholds to maximize balanced performance (F1 with accuracy constraint).
    """
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_probs = []
    all_labels = []
    all_img_attn = []
    all_txt_attn = []

    with torch.no_grad():
        for b_img, b_txt, b_lbl in dataloader:
            b_img, b_txt, b_lbl = b_img.to(device), b_txt.to(device), b_lbl.to(device)
            logits, _, img_attn, txt_attn = model(b_img, b_txt)
            loss = criterion(logits, b_lbl)
            total_loss += loss.item() * b_img.size(0)
            probs = F.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(b_lbl.cpu().numpy().tolist())
            all_img_attn.extend(img_attn.cpu().view(-1).numpy().tolist())
            all_txt_attn.extend(txt_attn.cpu().view(-1).numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    thresholds = np.linspace(min_t, max_t, steps)
    best_score = -1.0
    best_t = 0.50

    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        acc = accuracy_score(all_labels, preds)
        rec = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        # Balanced criterion: maximize F1 while penalizing accuracy drops below 55%
        penalty = 1.0 if acc >= 0.55 else (acc / 0.55) ** 2
        score = f1 * penalty

        if score > best_score:
            best_score = score
            best_t = t

    best_preds = (all_probs >= best_t).astype(int)
    cm = confusion_matrix(all_labels, best_preds)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auroc = 0.50

    best_metrics = {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': accuracy_score(all_labels, best_preds),
        'precision': precision_score(all_labels, best_preds, zero_division=0),
        'recall': recall_score(all_labels, best_preds, zero_division=0),
        'f1': f1_score(all_labels, best_preds, zero_division=0),
        'auroc': auroc,
        'confusion_matrix': cm,
        'predictions': best_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'img_attn': np.array(all_img_attn),
        'txt_attn': np.array(all_txt_attn),
        'threshold': best_t
    }

    return best_t, best_metrics


def train_proposed_model(train_loader, val_loader, device, epochs=10, lr=0.001, class_weights=None):
    print("\n" + "=" * 65)
    print("TRAINING RECONSTRUCTED MULTIMODAL MODEL (ResNet50 + BERT + BAM)")
    print("Optimization: AdamW + CosineAnnealingLR + Class-Weighted Focal Loss")
    print("=" * 65)

    model = MultimodalModel50Percent(img_in_dim=2048, text_in_dim=768, proj_dim=512, hidden_dim=256).to(device)
    
    criterion = FocalLoss(alpha=class_weights, gamma=1.5, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': [],
        'val_rec': []
    }

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    best_val_f1 = -1.0
    best_ckpt_path = os.path.join("checkpoints", "best_multimodal_model.pth")
    best_threshold = 0.50

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []

        start_time = time.time()
        for b_img, b_txt, b_lbl in train_loader:
            b_img, b_txt, b_lbl = b_img.to(device), b_txt.to(device), b_lbl.to(device)

            optimizer.zero_grad()
            logits, _, _, _ = model(b_img, b_txt)
            loss = criterion(logits, b_lbl)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * b_img.size(0)
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy().tolist())
            train_labels.extend(b_lbl.cpu().numpy().tolist())

        scheduler.step()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(train_labels, train_preds)

        # Fast vectorized validation evaluation
        opt_t, opt_val_eval = find_optimal_threshold(model, val_loader, device, criterion)

        elapsed = time.time() - start_time

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(opt_val_eval['loss'])
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(opt_val_eval['accuracy'])
        history['val_f1'].append(opt_val_eval['f1'])
        history['val_rec'].append(opt_val_eval['recall'])

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {opt_val_eval['loss']:.4f}, Val Acc: {opt_val_eval['accuracy']*100:.2f}%, "
            f"Val Recall: {opt_val_eval['recall']*100:.2f}%, Val F1: {opt_val_eval['f1']*100:.2f}% (tau*={opt_t:.2f})"
        )

        if opt_val_eval['f1'] > best_val_f1:
            best_val_f1 = opt_val_eval['f1']
            best_threshold = opt_t
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimal_threshold': float(best_threshold),
                'val_metrics': opt_val_eval,
                'config': {
                    'img_in_dim': 2048,
                    'text_in_dim': 768,
                    'proj_dim': 512,
                    'hidden_dim': 256,
                    'num_classes': 2
                }
            }
            torch.save(checkpoint, best_ckpt_path)

    print(f"\nModel training completed! Best Validation F1 Score: {best_val_f1*100:.2f}% (Optimal Threshold: {best_threshold:.2f})")
    print(f"Best checkpoint saved to: {best_ckpt_path}")

    # Load best checkpoint
    best_ckpt = torch.load(best_ckpt_path, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    best_threshold = best_ckpt['optimal_threshold']

    return model, history, best_threshold


def train_and_eval_baseline(model_class, model_name, train_loader, val_loader, test_loader, device, criterion, epochs=10, lr=0.001):
    print(f"\nTraining Baseline: {model_name}...")
    model = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        for b_img, b_txt, b_lbl in train_loader:
            b_img, b_txt, b_lbl = b_img.to(device), b_txt.to(device), b_lbl.to(device)
            optimizer.zero_grad()
            logits, _, _, _ = model(b_img, b_txt)
            loss = criterion(logits, b_lbl)
            loss.backward()
            optimizer.step()

    opt_t, val_metrics = find_optimal_threshold(model, val_loader, device, criterion)
    test_metrics = evaluate_model(model, test_loader, device, threshold=opt_t, criterion=criterion)
    return val_metrics, test_metrics, opt_t


def generate_plots(history, val_metrics, test_metrics):
    os.makedirs("results", exist_ok=True)

    # 1. Loss Curve
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss')
    plt.title('MSCMGTB - Loss vs Epoch (Reconstructed)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Focal Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/loss_curve.png', dpi=300)
    plt.close()

    # 2. Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Training Accuracy')
    plt.plot(epochs, [a * 100 for a in history['val_acc']], 'r-s', label='Validation Accuracy (Calibrated)')
    plt.title('MSCMGTB - Accuracy vs Epoch (Reconstructed)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/accuracy_curve.png', dpi=300)
    plt.close()

    # 3. Confusion Matrix
    plt.figure(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=val_metrics['confusion_matrix'], display_labels=['Non-Hateful', 'Hateful'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Validation Confusion Matrix (tau*={val_metrics['threshold']:.2f})", fontsize=14)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300)
    plt.close()

    # 4. BAM Attention Weights Sample Bar Chart
    plt.figure(figsize=(10, 5))
    sample_indices = range(10)
    img_attns = val_metrics['img_attn'][:10]
    txt_attns = val_metrics['txt_attn'][:10]

    bar_width = 0.35
    x = np.arange(len(sample_indices))

    plt.bar(x - bar_width/2, img_attns, bar_width, label='Visual Attention (CNN ResNet50)', color='#1f77b4')
    plt.bar(x + bar_width/2, txt_attns, bar_width, label='Textual Attention (BERT)', color='#ff7f0e')

    plt.xlabel('Validation Sample Index', fontsize=12)
    plt.ylabel('Attention Weight', fontsize=12)
    plt.title('Bi-directional Attention Mechanism (BAM) Modality Weights', fontsize=14)
    plt.xticks(x, [f'Sample #{i+1}' for i in sample_indices], rotation=45)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/bam_attention_sample.png', dpi=300)
    plt.close()

    print("Generated updated visualization plots in 'results/':")
    print("  - results/loss_curve.png")
    print("  - results/accuracy_curve.png")
    print("  - results/confusion_matrix.png")
    print("  - results/bam_attention_sample.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("MSCMGTB - 50% RECONSTRUCTED IMPLEMENTATION PIPELINE")
    print("=" * 65)
    print(f"Execution Device: {device}")

    # Check required feature files
    train_img_path = "features/train_image_features.pt"
    train_txt_path = "features/train_text_features.pt"
    train_lbl_path = "features/train_labels.pt"

    val_img_path = "features/val_image_features.pt"
    val_txt_path = "features/val_text_features.pt"
    val_lbl_path = "features/val_labels.pt"

    test_img_path = "features/test_image_features.pt"
    test_txt_path = "features/test_text_features.pt"
    test_lbl_path = "features/test_labels.pt"

    for path in [train_img_path, train_txt_path, train_lbl_path, val_img_path, val_txt_path, val_lbl_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required feature file: {path}. Run features/extract_features.py first!")

    # Load feature tensors
    train_img = torch.load(train_img_path, weights_only=False)
    train_txt = torch.load(train_txt_path, weights_only=False)
    train_lbl = torch.load(train_lbl_path, weights_only=False)

    val_img = torch.load(val_img_path, weights_only=False)
    val_txt = torch.load(val_txt_path, weights_only=False)
    val_lbl = torch.load(val_lbl_path, weights_only=False)

    test_img = torch.load(test_img_path, weights_only=False)
    test_txt = torch.load(test_txt_path, weights_only=False)
    test_lbl = torch.load(test_lbl_path, weights_only=False)

    print(f"\nFeature Datasets Loaded Successfully:")
    print(f"  Training samples   : {len(train_lbl)} (Image: {train_img.shape}, Text: {train_txt.shape})")
    print(f"  Validation samples : {len(val_lbl)} (Image: {val_img.shape}, Text: {val_txt.shape})")
    print(f"  Test samples       : {len(test_lbl)} (Image: {test_img.shape}, Text: {test_txt.shape})")

    # Compute class weights to counter the 1.815:1 imbalance
    n_0 = (train_lbl == 0).sum().item()
    n_1 = (train_lbl == 1).sum().item()
    total = len(train_lbl)
    w0 = total / (2.0 * n_0)
    w1 = total / (2.0 * n_1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
    print(f"\nClass Distribution in Training Set: Non-Hateful={n_0}, Hateful={n_1}")
    print(f"Class-Balancing Weights: Non-Hateful={w0:.4f}, Hateful={w1:.4f} (Ratio: {w1/w0:.3f})")

    batch_size = 32
    train_loader = get_feature_dataloader(train_img, train_txt, train_lbl, batch_size=batch_size, shuffle=True)
    val_loader = get_feature_dataloader(val_img, val_txt, val_lbl, batch_size=batch_size, shuffle=False)
    test_loader = get_feature_dataloader(test_img, test_txt, test_lbl, batch_size=batch_size, shuffle=False)

    # Train proposed model (CNN + BERT + Reconstructed BAM)
    model, history, optimal_t = train_proposed_model(
        train_loader,
        val_loader,
        device,
        epochs=10,
        lr=0.001,
        class_weights=class_weights
    )

    criterion = FocalLoss(alpha=class_weights, gamma=1.5, label_smoothing=0.05)

    # Validation and Test Evaluation of Proposed Model
    val_metrics = evaluate_model(model, val_loader, device, threshold=optimal_t, criterion=criterion)
    test_metrics = evaluate_model(model, test_loader, device, threshold=optimal_t, criterion=criterion)

    # Baseline Experiments
    print("\nRunning Baseline Comparison Experiments...")
    b1_val, b1_test, _ = train_and_eval_baseline(BERTOnlyBaseline, "Baseline 1 (BERT-only)", train_loader, val_loader, test_loader, device, criterion)
    b2_val, b2_test, _ = train_and_eval_baseline(CNNOnlyBaseline, "Baseline 2 (CNN-only)", train_loader, val_loader, test_loader, device, criterion)
    b3_val, b3_test, _ = train_and_eval_baseline(ConcatNoBAMBaseline, "Baseline 3 (CNN + BERT without BAM)", train_loader, val_loader, test_loader, device, criterion)

    # Generate Plots
    generate_plots(history, val_metrics, test_metrics)

    # Display BAM Attention Values for 5 Validation Samples
    print("\n" + "=" * 65)
    print("BI-DIRECTIONAL ATTENTION MECHANISM (BAM) SAMPLE INSPECTION")
    print("=" * 65)
    label_map = {0: "Non-Hateful", 1: "Hateful"}
    for i in range(min(5, len(val_metrics['labels']))):
        true_l = label_map[val_metrics['labels'][i]]
        pred_l = label_map[val_metrics['predictions'][i]]
        img_a = val_metrics['img_attn'][i]
        txt_a = val_metrics['txt_attn'][i]
        print(f"Sample #{i+1:02d} | True Label: {true_l:<12} | Prediction: {pred_l:<12} | Image Attn: {img_a:.4f} | Text Attn: {txt_a:.4f}")

    # Output Results Table
    print("\n" + "=" * 78)
    print(f"FINAL VALIDATION RESULTS COMPARISON TABLE (Calibrated Threshold tau* = {optimal_t:.2f})")
    print("=" * 78)
    print(f"{'Model':<32} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'AUROC':<10}")
    print("-" * 88)
    print(f"{'Baseline 1: BERT-only':<32} | {b1_val['accuracy']*100:6.2f}%    | {b1_val['precision']*100:6.2f}%    | {b1_val['recall']*100:6.2f}%    | {b1_val['f1']*100:6.2f}%    | {b1_val['auroc']*100:6.2f}%")
    print(f"{'Baseline 2: CNN-only':<32} | {b2_val['accuracy']*100:6.2f}%    | {b2_val['precision']*100:6.2f}%    | {b2_val['recall']*100:6.2f}%    | {b2_val['f1']*100:6.2f}%    | {b2_val['auroc']*100:6.2f}%")
    print(f"{'Baseline 3: CNN + BERT (No BAM)':<32} | {b3_val['accuracy']*100:6.2f}%    | {b3_val['precision']*100:6.2f}%    | {b3_val['recall']*100:6.2f}%    | {b3_val['f1']*100:6.2f}%    | {b3_val['auroc']*100:6.2f}%")
    print(f"{'Proposed: CNN + BERT + BAM':<32} | {val_metrics['accuracy']*100:6.2f}%    | {val_metrics['precision']*100:6.2f}%    | {val_metrics['recall']*100:6.2f}%    | {val_metrics['f1']*100:6.2f}%    | {val_metrics['auroc']*100:6.2f}%")
    print("=" * 88)

    # Save results.txt
    os.makedirs("results", exist_ok=True)
    with open("results/results.txt", "w") as f:
        f.write("============================================================\n")
        f.write("MSCMGTB - 50% IMPLEMENTATION EXPERIMENTAL RESULTS (RECONSTRUCTED)\n")
        f.write("============================================================\n\n")
        f.write("DATASET INFORMATION:\n")
        f.write(f"  Training samples   : {len(train_lbl)}\n")
        f.write(f"  Validation samples : {len(val_lbl)} (dev_seen + dev_unseen)\n")
        f.write(f"  Test samples       : {len(test_lbl)} (test_seen + test_unseen)\n\n")
        f.write("MODEL ARCHITECTURE:\n")
        f.write("  Visual Encoder     : ResNet50 (2048-dimensional visual features)\n")
        f.write("  Visual Projection  : Linear(2048, 512) -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm\n")
        f.write("  Text Encoder       : BERT bert-base-uncased (768-dimensional textual features)\n")
        f.write("  Text Projection    : Linear(768, 512) -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm\n")
        f.write("  Multimodal Fusion  : Cross-Modal Conditioned BAM + Hadamard Bilinear Interaction (2048 -> 512)\n")
        f.write("  Classifier         : Linear(512, 256) -> LayerNorm -> GELU -> Dropout -> Linear(256, 128) -> Linear(128, 2)\n\n")
        f.write("TRAINING PARAMETERS:\n")
        f.write("  Batch size         : 32\n")
        f.write("  Learning rate      : 0.001\n")
        f.write("  Epochs             : 10\n")
        f.write("  Optimizer          : AdamW (weight_decay=1e-4)\n")
        f.write("  Scheduler          : CosineAnnealingLR (eta_min=1e-5)\n")
        f.write("  Loss Function      : Class-Weighted Focal Loss (gamma=1.5, label_smoothing=0.05)\n")
        f.write(f"  Calibrated Threshold : {optimal_t:.4f}\n\n")
        f.write("PROPOSED MODEL VALIDATION RESULTS:\n")
        f.write(f"  Accuracy           : {val_metrics['accuracy']*100:.2f}%\n")
        f.write(f"  Precision          : {val_metrics['precision']*100:.2f}%\n")
        f.write(f"  Recall             : {val_metrics['recall']*100:.2f}%\n")
        f.write(f"  F1-score           : {val_metrics['f1']*100:.2f}%\n")
        f.write(f"  AUROC              : {val_metrics['auroc']*100:.2f}%\n\n")
        f.write("CONFUSION MATRIX (VALIDATION):\n")
        f.write(f"{val_metrics['confusion_matrix']}\n\n")
        f.write("PROPOSED MODEL TEST RESULTS:\n")
        f.write(f"  Accuracy           : {test_metrics['accuracy']*100:.2f}%\n")
        f.write(f"  Precision          : {test_metrics['precision']*100:.2f}%\n")
        f.write(f"  Recall             : {test_metrics['recall']*100:.2f}%\n")
        f.write(f"  F1-score           : {test_metrics['f1']*100:.2f}%\n")
        f.write(f"  AUROC              : {test_metrics['auroc']*100:.2f}%\n\n")
        f.write("BASELINE COMPARISON TABLE (VALIDATION):\n")
        f.write(f"{'Model':<32} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'AUROC':<10}\n")
        f.write("-" * 88 + "\n")
        f.write(f"{'BERT-only':<32} | {b1_val['accuracy']*100:6.2f}%    | {b1_val['precision']*100:6.2f}%    | {b1_val['recall']*100:6.2f}%    | {b1_val['f1']*100:6.2f}%    | {b1_val['auroc']*100:6.2f}%\n")
        f.write(f"{'CNN-only':<32} | {b2_val['accuracy']*100:6.2f}%    | {b2_val['precision']*100:6.2f}%    | {b2_val['recall']*100:6.2f}%    | {b2_val['f1']*100:6.2f}%    | {b2_val['auroc']*100:6.2f}%\n")
        f.write(f"{'CNN + BERT (No BAM)':<32} | {b3_val['accuracy']*100:6.2f}%    | {b3_val['precision']*100:6.2f}%    | {b3_val['recall']*100:6.2f}%    | {b3_val['f1']*100:6.2f}%    | {b3_val['auroc']*100:6.2f}%\n")
        f.write(f"{'CNN + BERT + BAM (Proposed)':<32} | {val_metrics['accuracy']*100:6.2f}%    | {val_metrics['precision']*100:6.2f}%    | {val_metrics['recall']*100:6.2f}%    | {val_metrics['f1']*100:6.2f}%    | {val_metrics['auroc']*100:6.2f}%\n\n")
        f.write("BEST CHECKPOINT PATH:\n")
        f.write("  checkpoints/best_multimodal_model.pth\n")

    print(f"\nSaved reconstructed experimental results log to 'results/results.txt'.")


if __name__ == "__main__":
    main()
