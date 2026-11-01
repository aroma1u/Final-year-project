import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from models.multimodal import MemeMultimodalClassifier

def load_feature_dataset(split: str):
    feat_path = os.path.join("features", f"{split}_features.pt")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature file not found: {feat_path}. Run features/extract_features.py first.")
    
    data = torch.load(feat_path)
    dataset = TensorDataset(data['img_features'], data['text_features'], data['labels'])
    return dataset

def train_model(epochs: int = 10, batch_size: int = 32, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Multimodal Baseline Model on device: {device}")
    
    # Load dataset features
    train_ds = load_feature_dataset('train')
    val_ds = load_feature_dataset('dev_seen')
    val_unseen_ds = load_feature_dataset('dev_unseen')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    val_unseen_loader = DataLoader(val_unseen_ds, batch_size=batch_size, shuffle=False)

    model = MemeMultimodalClassifier(img_feature_dim=512, text_feature_dim=768, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs("results", exist_ok=True)
    best_val_acc = 0.0

    print("\nStarting Training Loop...")
    print("-" * 65)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start_t = time.time()

        for img_feats, text_feats, labels in train_loader:
            img_feats, text_feats, labels = img_feats.to(device), text_feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(img_feats, text_feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * img_feats.size(0)

        train_loss = total_loss / len(train_ds)

        # Validation evaluation
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for img_feats, text_feats, labels in val_loader:
                img_feats, text_feats = img_feats.to(device), text_feats.to(device)
                logits = model(img_feats, text_feats)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_probs)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        elapsed = time.time() - start_t

        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | ROC-AUC: {roc_auc:.4f} | F1: {macro_f1:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            ckpt_path = os.path.join("results", "best_multimodal_model.pt")
            torch.save(model.state_dict(), ckpt_path)

    print("-" * 65)
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
