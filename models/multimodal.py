import torch
import torch.nn as nn
from .cnn import MemeCNNClassifier
from .bert import MemeBERTClassifier

class MemeMultimodalClassifier(nn.Module):
    """
    Multimodal Fusion Neural Network for Hateful Memes Classification.
    Fuses visual features (e.g. ResNet 512-d) and textual features (e.g. BERT 768-d).
    """
    def __init__(
        self,
        img_feature_dim: int = 512,
        text_feature_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout_prob: float = 0.3
    ):
        super(MemeMultimodalClassifier, self).__init__()
        fused_dim = img_feature_dim + text_feature_dim
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, img_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            img_features (torch.Tensor): Tensor of shape (batch_size, img_feature_dim)
            text_features (torch.Tensor): Tensor of shape (batch_size, text_feature_dim)
        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes)
        """
        fused_representation = torch.cat((img_features, text_features), dim=1)
        logits = self.fusion_head(fused_representation)
        return logits

class EndToEndMultimodalModel(nn.Module):
    """
    End-to-End Multimodal Model combining CNN and BERT encoders directly.
    """
    def __init__(self, cnn_backbone: str = 'resnet18', bert_model_name: str = 'bert-base-uncased', num_classes: int = 2):
        super(EndToEndMultimodalModel, self).__init__()
        self.cnn_encoder = MemeCNNClassifier(backbone=cnn_backbone, extract_features=True)
        self.bert_encoder = MemeBERTClassifier(model_name=bert_model_name, extract_features=True)
        
        img_dim = 512 if cnn_backbone == 'resnet18' else 2048
        text_dim = 768  # Standard BERT hidden size
        
        self.fusion_classifier = MemeMultimodalClassifier(
            img_feature_dim=img_dim,
            text_feature_dim=text_dim,
            num_classes=num_classes
        )

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        img_feats = self.cnn_encoder(images)
        text_feats = self.bert_encoder(input_ids, attention_mask)
        logits = self.fusion_classifier(img_feats, text_feats)
        return logits

if __name__ == "__main__":
    model = MemeMultimodalClassifier(img_feature_dim=512, text_feature_dim=768)
    dummy_img = torch.randn(8, 512)
    dummy_txt = torch.randn(8, 768)
    out = model(dummy_img, dummy_txt)
    print(f"MemeMultimodalClassifier test output shape: {out.shape}")
