import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageProjection(nn.Module):
    """
    Upgraded Visual Feature Projection Head with Residual Skip Connections and LayerNorm.
    Projects in_features (2048-d ResNet50 or 512-d Gemini) into 512-d representation space.
    """
    def __init__(self, in_features: int = 2048, out_features: int = 512, dropout_prob: float = 0.3):
        super(ImageProjection, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
        )
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.act(self.net(x) + residual)
        return self.dropout(out)


class TextProjection(nn.Module):
    """
    Upgraded Textual Feature Projection Head with Residual Skip Connections and LayerNorm.
    Projects in_features (768-d BERT or 512-d Gemini) into 512-d representation space.
    """
    def __init__(self, in_features: int = 768, out_features: int = 512, dropout_prob: float = 0.3):
        super(TextProjection, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
        )
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.act(self.net(x) + residual)
        return self.dropout(out)


class BiDirectionalAttentionMechanism(nn.Module):
    """
    Reconstructed Bi-directional Attention Mechanism (BAM) with Cross-Modal Conditioning
    and Bilinear Hadamard Interaction Fusion.
    
    Eliminates the unimodal zero-sum gating flaw by conditioning attention on joint cross-modal
    representations, while modeling element-wise multiplicative (Hadamard) and discrepancy interactions.
    Outputs normalized modality attention weights [alpha_visual, alpha_textual] for interpretability.
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 128, dropout_prob: float = 0.3, temperature: float = 2.0):
        super(BiDirectionalAttentionMechanism, self).__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # 1. Cross-Modal Joint Representation Layer
        self.cross_dense = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )
        
        # 2. Contextualization Projections
        self.v_ctx_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )
        self.t_ctx_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )
        
        # 3. Cross-Conditioned Modality Attention Scoring Heads
        # Normalized with LayerNorm and temperature to prevent probability saturation
        self.img_attn_head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.text_attn_head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # Small weight initialization to ensure balanced initial modality exploration
        nn.init.normal_(self.img_attn_head[-1].weight, std=0.02)
        nn.init.normal_(self.text_attn_head[-1].weight, std=0.02)
        
        # 4. Multimodal Hadamard Interaction Fusion Network
        # Concatenates: [V', T', V' * T', |V' - T'|] -> 4 * feature_dim (2048-d) -> feature_dim (512-d)
        self.fusion_proj = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(feature_dim, feature_dim)
        )
        self.fusion_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, img_features: torch.Tensor, text_features: torch.Tensor):
        """
        Forward pass with bi-directional conditioning and Hadamard interaction.
        Args:
            img_features (torch.Tensor): Visual representation [N, 512]
            text_features (torch.Tensor): Textual representation [N, 512]
        Returns:
            fused_features (torch.Tensor): Fused multimodal representation [N, 512]
            image_attn (torch.Tensor): Interpretable visual attention weight [N, 1]
            text_attn (torch.Tensor): Interpretable text attention weight [N, 1]
        """
        # Step 1: Compute joint cross-modal context
        joint_state = torch.cat([img_features, text_features], dim=1) # [N, 1024]
        cross_context = self.cross_dense(joint_state)                 # [N, 512]
        
        # Step 2: Contextualize unimodal streams with cross-modal state
        v_ctx = img_features + self.v_ctx_proj(torch.cat([img_features, cross_context], dim=1)) # [N, 512]
        t_ctx = text_features + self.t_ctx_proj(torch.cat([text_features, cross_context], dim=1)) # [N, 512]
        
        # Step 3: Compute cross-conditioned modality attention scores
        v_score = self.img_attn_head(torch.cat([v_ctx, t_ctx], dim=1))   # [N, 1]
        t_score = self.text_attn_head(torch.cat([t_ctx, v_ctx], dim=1))  # [N, 1]
        
        # Temperature-scaled Softmax across the 2 modalities
        raw_scores = torch.cat([v_score, t_score], dim=1) / self.temperature # [N, 2]
        attn_weights = F.softmax(raw_scores, dim=1)                           # [N, 2]
        
        image_attn = attn_weights[:, 0:1] # [N, 1]
        text_attn = attn_weights[:, 1:2]  # [N, 1]
        
        # Step 4: Scale contextualized features
        v_weighted = v_ctx * (image_attn * 2.0) # [N, 512]
        t_weighted = t_ctx * (text_attn * 2.0)  # [N, 512]
        
        # Step 5: Bilinear Hadamard and Difference Interactions
        i_hadamard = v_weighted * t_weighted            # [N, 512] - captures multiplicative co-occurrence
        i_diff = torch.abs(v_weighted - t_weighted)     # [N, 512] - captures cross-modal discordance/irony
        
        # Step 6: Full Interaction Tensor Fusion (2048 -> 512)
        composite = torch.cat([v_weighted, t_weighted, i_hadamard, i_diff], dim=1) # [N, 2048]
        projected = self.fusion_proj(composite)                                     # [N, 512]
        
        # Step 7: Residual Skip Connection with LayerNorm
        residual = 0.5 * (v_weighted + t_weighted)
        fused_features = self.fusion_norm(projected + residual)
        fused_features = self.dropout(fused_features)
        
        return fused_features, image_attn, text_attn


if __name__ == "__main__":
    img_proj = ImageProjection(2048, 512)
    txt_proj = TextProjection(768, 512)
    bam = BiDirectionalAttentionMechanism(512)
    
    dummy_img_raw = torch.randn(4, 2048)
    dummy_txt_raw = torch.randn(4, 768)
    
    img_rep = img_proj(dummy_img_raw)
    txt_rep = txt_proj(dummy_txt_raw)
    
    fused, img_w, txt_w = bam(img_rep, txt_rep)
    
    print(f"Image Projection: {dummy_img_raw.shape} -> {img_rep.shape}")
    print(f"Text Projection:  {dummy_txt_raw.shape} -> {txt_rep.shape}")
    print(f"BAM Output:       {fused.shape}")
    print(f"Image Attention:  {img_w.squeeze().tolist()}")
    print(f"Text Attention:   {txt_w.squeeze().tolist()}")
