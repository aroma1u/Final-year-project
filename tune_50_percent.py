import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from dataset.feature_dataset import get_feature_dataloader
from models.multimodal_model import MultimodalModel50Percent

def evaluate_with_threshold(model, dataloader, device, threshold=0.50):
    """
    Evaluates model predictions at a specific classification threshold for class 1 (Hateful).
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for b_img, b_txt, b_lbl in dataloader:
            b_img, b_txt, b_lbl = b_img.to(device), b_txt.to(device), b_lbl.to(device)
            logits, _, _, _ = model(b_img, b_txt)
            probs = F.softmax(logits, dim=1)[:, 1] # Probability of Hateful (class 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(b_lbl.cpu().numpy())
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    preds = (all_probs >= threshold).astype(int)
    
    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    cm = confusion_matrix(all_labels, preds)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'probs': all_probs,
        'labels': all_labels,
        'threshold': threshold
    }

def find_best_threshold(model, dataloader, device, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.50, 11)
        
    best_t_metrics = None
    best_f1 = -1.0
    
    for t in thresholds:
        metrics = evaluate_with_threshold(model, dataloader, device, threshold=t)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_t_metrics = metrics
            
    return best_t_metrics

def train_and_tune_experiment(
    exp_id,
    use_class_weights,
    lr,
    max_epochs,
    train_loader,
    val_loader,
    train_labels,
    device,
    patience=5
):
    print(f"\n" + "=" * 65)
    print(f"EXPERIMENT {exp_id}: Class-Weighted={use_class_weights}, LR={lr}, Max Epochs={max_epochs}")
    print("=" * 65)

    model = MultimodalModel50Percent(img_in_dim=2048, text_in_dim=768, proj_dim=512, hidden_dim=256).to(device)
    
    if use_class_weights:
        # Calculate class weights from actual training label distribution
        count_0 = (train_labels == 0).sum().item()
        count_1 = (train_labels == 1).sum().item()
        total = len(train_labels)
        
        w0 = total / (2.0 * count_0)
        w1 = total / (2.0 * count_1)
        class_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
        print(f"  Calculated Training Class Weights: Class 0 (Non-Hateful)={w0:.4f}, Class 1 (Hateful)={w1:.4f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        print("  Using Unweighted CrossEntropyLoss.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Early stopping and LR scheduler variables
    best_val_f1 = -1.0
    best_model_state = None
    no_improve_count = 0
    actual_epochs_trained = 0

    for epoch in range(1, max_epochs + 1):
        actual_epochs_trained = epoch
        model.train()
        running_loss = 0.0
        
        for b_img, b_txt, b_lbl in train_loader:
            b_img, b_txt, b_lbl = b_img.to(device), b_txt.to(device), b_lbl.to(device)
            
            optimizer.zero_grad()
            logits, _, _, _ = model(b_img, b_txt)
            loss = criterion(logits, b_lbl)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * b_img.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Evaluate validation performance at standard 0.50 threshold for early stopping
        val_eval = evaluate_with_threshold(model, val_loader, device, threshold=0.50)
        val_f1 = val_eval['f1']
        
        print(f"  Epoch {epoch:02d}/{max_epochs:02d} - Loss: {epoch_loss:.4f} | Val Acc: {val_eval['accuracy']*100:.2f}%, Val F1 (t=0.5): {val_f1*100:.2f}%")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (no val F1 improvement for {patience} epochs).")
                break

    # Restore best epoch weights
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    # Perform threshold tuning on validation set (0.30 to 0.50)
    best_val_metrics = find_best_threshold(model, val_loader, device)
    
    print(f"  Best Validation Result after Threshold Search:")
    print(f"    Selected Threshold : {best_val_metrics['threshold']:.2f}")
    print(f"    Val Accuracy       : {best_val_metrics['accuracy']*100:.2f}%")
    print(f"    Val Precision      : {best_val_metrics['precision']*100:.2f}%")
    print(f"    Val Recall         : {best_val_metrics['recall']*100:.2f}%")
    print(f"    Val F1-Score       : {best_val_metrics['f1']*100:.2f}%")

    return {
        'exp_id': exp_id,
        'class_weighting': 'Yes' if use_class_weights else 'No',
        'lr': lr,
        'epochs': actual_epochs_trained,
        'threshold': best_val_metrics['threshold'],
        'val_accuracy': best_val_metrics['accuracy'],
        'val_precision': best_val_metrics['precision'],
        'val_recall': best_val_metrics['recall'],
        'val_f1': best_val_metrics['f1'],
        'model_state': best_model_state,
        'val_metrics': best_val_metrics
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("MSCMGTB - CONTROLLED HYPERPARAMETER & THRESHOLD TUNING")
    print("=" * 65)
    print(f"Device: {device}")

    # Load feature datasets
    train_img = torch.load("features/train_image_features.pt", weights_only=False)
    train_txt = torch.load("features/train_text_features.pt", weights_only=False)
    train_lbl = torch.load("features/train_labels.pt", weights_only=False)

    val_img = torch.load("features/val_image_features.pt", weights_only=False)
    val_txt = torch.load("features/val_text_features.pt", weights_only=False)
    val_lbl = torch.load("features/val_labels.pt", weights_only=False)

    test_img = torch.load("features/test_image_features.pt", weights_only=False)
    test_txt = torch.load("features/test_text_features.pt", weights_only=False)
    test_lbl = torch.load("features/test_labels.pt", weights_only=False)

    batch_size = 32
    train_loader = get_feature_dataloader(train_img, train_txt, train_lbl, batch_size=batch_size, shuffle=True)
    val_loader = get_feature_dataloader(val_img, val_txt, val_lbl, batch_size=batch_size, shuffle=False)
    test_loader = get_feature_dataloader(test_img, test_txt, test_lbl, batch_size=batch_size, shuffle=False)

    # 1 & 2. Class imbalance inspection
    n_non_hateful = (train_lbl == 0).sum().item()
    n_hateful = (train_lbl == 1).sum().item()
    print(f"\nTraining Class Distribution:")
    print(f"  Non-Hateful (0) : {n_non_hateful} ({n_non_hateful/len(train_lbl)*100:.2f}%)")
    print(f"  Hateful (1)     : {n_hateful} ({n_hateful/len(train_lbl)*100:.2f}%)")
    print(f"  Imbalance Ratio : 1 : {n_non_hateful/n_hateful:.2f}")

    # Define hyperparameter grid
    # Learning rates: 0.001, 0.0005, 0.0001
    # Class weighting: False vs True
    lrs = [0.001, 0.0005, 0.0001]
    weighting_opts = [False, True]
    max_epochs = 20

    results_records = []
    best_overall_exp = None
    best_overall_val_f1 = -1.0

    exp_counter = 1
    for use_weights in weighting_opts:
        for lr in lrs:
            exp_result = train_and_tune_experiment(
                exp_id=exp_counter,
                use_class_weights=use_weights,
                lr=lr,
                max_epochs=max_epochs,
                train_loader=train_loader,
                val_loader=val_loader,
                train_labels=train_lbl,
                device=device,
                patience=5
            )
            
            record = {
                'Experiment': f"Exp_{exp_counter}",
                'Class_Weighting': exp_result['class_weighting'],
                'Learning_Rate': exp_result['lr'],
                'Epochs_Trained': exp_result['epochs'],
                'Best_Threshold': f"{exp_result['threshold']:.2f}",
                'Val_Accuracy': f"{exp_result['val_accuracy']*100:.2f}%",
                'Val_Precision': f"{exp_result['val_precision']*100:.2f}%",
                'Val_Recall': f"{exp_result['val_recall']*100:.2f}%",
                'Val_F1': f"{exp_result['val_f1']*100:.2f}%"
            }
            results_records.append(record)

            if exp_result['val_f1'] > best_overall_val_f1:
                best_overall_val_f1 = exp_result['val_f1']
                best_overall_exp = exp_result

            exp_counter += 1

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/tuning_results.csv"
    df_results = pd.DataFrame(results_records)
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved all experimental results to '{csv_path}'.")

    # Generate f1_comparison.png plot
    plt.figure(figsize=(10, 6))
    exp_labels = [f"Exp {r['Experiment']} (LR={r['Learning_Rate']}, W={r['Class_Weighting']})" for r in results_records]
    f1_scores = [float(r['Val_F1'].replace('%', '')) for r in results_records]
    
    colors = ['#2ca02c' if f == max(f1_scores) else '#1f77b4' for f in f1_scores]
    bars = plt.bar(range(len(f1_scores)), f1_scores, color=colors, width=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xlabel('Controlled Experiment Configurations', fontsize=12)
    plt.ylabel('Validation F1-Score (%)', fontsize=12)
    plt.title('MSCMGTB - Validation F1 Comparison Across Hyperparameters & Weights', fontsize=13)
    plt.xticks(range(len(f1_scores)), [f"Exp {i+1}" for i in range(len(f1_scores))], rotation=0)
    plt.ylim(0, max(f1_scores) + 8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/f1_comparison.png", dpi=300)
    plt.close()
    print("Saved comparison chart to 'results/f1_comparison.png'.")

    # Save Best Tuned Checkpoint based on Validation F1
    os.makedirs("checkpoints", exist_ok=True)
    best_ckpt_path = "checkpoints/best_tuned_multimodal_model.pth"
    best_model_obj = MultimodalModel50Percent(img_in_dim=2048, text_in_dim=768, proj_dim=512, hidden_dim=256)
    best_model_obj.load_state_dict(best_overall_exp['model_state'])
    
    checkpoint_data = {
        'experiment_id': best_overall_exp['exp_id'],
        'model_state_dict': best_overall_exp['model_state'],
        'best_threshold': best_overall_exp['threshold'],
        'val_metrics': best_overall_exp['val_metrics'],
        'learning_rate': best_overall_exp['lr'],
        'class_weighting': best_overall_exp['class_weighting']
    }
    torch.save(checkpoint_data, best_ckpt_path)
    print(f"Saved best tuned checkpoint to '{best_ckpt_path}'.")

    # Final Single Evaluation on Test Set using Best Tuned Model & Selected Validation Threshold
    best_model_obj.to(device)
    test_eval_best = evaluate_with_threshold(best_model_obj, test_loader, device, threshold=best_overall_exp['threshold'])

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model Configuration':<35} | {'Val Acc':<9} | {'Val Prec':<9} | {'Val Rec':<9} | {'Val F1':<9}")
    print("-" * 75)
    print(f"{'Original CNN+BERT+BAM (lr=0.001)':<35} | {58.12:6.2f}%  | {51.96:6.2f}%  | {40.67:6.2f}%  | {45.62:6.2f}%")
    print(f"{'Best Tuned CNN+BERT+BAM (Exp #' + str(best_overall_exp['exp_id']) + ')':<35} | {best_overall_exp['val_accuracy']*100:6.2f}%  | {best_overall_exp['val_precision']*100:6.2f}%  | {best_overall_exp['val_recall']*100:6.2f}%  | {best_overall_exp['val_f1']*100:6.2f}%")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("BEST TUNED MODEL TEST SET EVALUATION (Single Final Evaluation)")
    print("=" * 70)
    print(f"  Optimal Threshold Selected from Validation : {best_overall_exp['threshold']:.2f}")
    print(f"  Test Accuracy  : {test_eval_best['accuracy']*100:.2f}%")
    print(f"  Test Precision : {test_eval_best['precision']*100:.2f}%")
    print(f"  Test Recall    : {test_eval_best['recall']*100:.2f}%")
    print(f"  Test F1-Score  : {test_eval_best['f1']*100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
