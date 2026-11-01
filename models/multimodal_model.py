import os
import sys
import torch
import torch.nn as nn

try:
    from models.bam import ImageProjection, TextProjection, BiDirectionalAttentionMechanism
except ModuleNotFoundError:
    from bam import ImageProjection, TextProjection, BiDirectionalAttentionMechanism

class MultimodalModel50Percent(nn.Module):
    """
    Multimodal Social Media Content Moderation Model (50% Milestone Reconstructed).
    Combines visual features (2048-d ResNet50 or 512-d Gemini) and textual features (768-d BERT or 512-d Gemini)
    using Cross-Modal Bi-directional Attention (BAM) with Hadamard interaction fusion and deep classification head.
    """
    def __init__(
        self,
        img_in_dim: int = 2048,
        text_in_dim: int = 768,
        proj_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout_prob: float = 0.3
    ):
        super(MultimodalModel50Percent, self).__init__()
        
        self.img_projection = ImageProjection(in_features=img_in_dim, out_features=proj_dim, dropout_prob=dropout_prob)
        self.text_projection = TextProjection(in_features=text_in_dim, out_features=proj_dim, dropout_prob=dropout_prob)
        
        self.bam = BiDirectionalAttentionMechanism(feature_dim=proj_dim, hidden_dim=128, dropout_prob=dropout_prob)
        
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, img_features: torch.Tensor, text_features: torch.Tensor):
        """
        Forward pass.
        Args:
            img_features (torch.Tensor): Visual feature tensor [N, img_in_dim]
            text_features (torch.Tensor): Textual feature tensor [N, text_in_dim]
        Returns:
            logits (torch.Tensor): Logits tensor [N, 2]
            fused_features (torch.Tensor): Fused representation [N, 512]
            img_attn (torch.Tensor): Visual attention weight [N, 1]
            text_attn (torch.Tensor): Textual attention weight [N, 1]
        """
        # Feature projections: img_in_dim -> 512, text_in_dim -> 512
        v_proj = self.img_projection(img_features)
        t_proj = self.text_projection(text_features)
        
        # Bi-directional Cross-Modal Attention with Hadamard Fusion
        fused_features, img_attn, text_attn = self.bam(v_proj, t_proj)
        
        # Deep Classifier head: 512 -> 256 -> 128 -> 2
        logits = self.classifier(fused_features)
        
        return logits, fused_features, img_attn, text_attn


# Baseline Model Implementations
class BERTOnlyBaseline(nn.Module):
    """Baseline 1: Text-only model using BERT features (768-d)."""
    def __init__(self, text_in_dim: int = 768, proj_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2):
        super(BERTOnlyBaseline, self).__init__()
        self.text_projection = TextProjection(text_in_dim, proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, img_features: torch.Tensor, text_features: torch.Tensor):
        t_proj = self.text_projection(text_features)
        logits = self.classifier(t_proj)
        dummy_attn = torch.ones((text_features.size(0), 1), device=text_features.device)
        return logits, t_proj, dummy_attn, dummy_attn


class CNNOnlyBaseline(nn.Module):
    """Baseline 2: Image-only model using CNN ResNet50 features (2048-d)."""
    def __init__(self, img_in_dim: int = 2048, proj_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2):
        super(CNNOnlyBaseline, self).__init__()
        self.img_projection = ImageProjection(img_in_dim, proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, img_features: torch.Tensor, text_features: torch.Tensor):
        v_proj = self.img_projection(img_features)
        logits = self.classifier(v_proj)
        dummy_attn = torch.ones((img_features.size(0), 1), device=img_features.device)
        return logits, v_proj, dummy_attn, dummy_attn


class ConcatNoBAMBaseline(nn.Module):
    """Baseline 3: Multimodal CNN + BERT without BAM attention (Direct concatenation 512+512=1024 -> 512 -> 256 -> 2)."""
    def __init__(self, img_in_dim: int = 2048, text_in_dim: int = 768, proj_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2):
        super(ConcatNoBAMBaseline, self).__init__()
        self.img_projection = ImageProjection(img_in_dim, proj_dim)
        self.text_projection = TextProjection(text_in_dim, proj_dim)
        self.concat_proj = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, img_features: torch.Tensor, text_features: torch.Tensor):
        v_proj = self.img_projection(img_features)
        t_proj = self.text_projection(text_features)
        cat_feat = torch.cat([v_proj, t_proj], dim=1)
        fused = self.concat_proj(cat_feat)
        logits = self.classifier(fused)
        dummy_attn = torch.ones((img_features.size(0), 1), device=img_features.device)
        return logits, fused, dummy_attn, dummy_attn


if __name__ == "__main__":
    model = MultimodalModel50Percent()
    dummy_img = torch.randn(8, 2048)
    dummy_txt = torch.randn(8, 768)
    logits, fused, img_w, txt_w = model(dummy_img, dummy_txt)
    print(f"Logits shape: {logits.shape}")
    print(f"Fused features shape: {fused.shape}")
    print(f"Image attention shape: {img_w.shape}")
    print(f"Text attention shape: {txt_w.shape}")
