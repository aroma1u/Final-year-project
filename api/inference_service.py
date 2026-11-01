import os
import sys
import io
import datetime
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.multimodal_model import MultimodalModel50Percent

class InferenceService:
    """
    Singleton Inference Engine for MemeGuard AI.
    Loads ResNet-50 visual extractor, BERT textual extractor, and trained BAM multimodal classifier.
    Performs inference with calibrated decision thresholding.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(InferenceService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, checkpoint_path: str = "checkpoints/best_multimodal_model.pth"):
        if self._initialized:
            return

        self.checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.error_message = None
        self.optimal_threshold = 0.41

        try:
            print(f"[InferenceService] Initializing on device: {self.device}")
            
            # 1. ResNet-50 Visual Extractor
            self.cnn = models.resnet50(weights="DEFAULT")
            self.cnn.fc = torch.nn.Identity()
            self.cnn = self.cnn.to(self.device)
            self.cnn.eval()

            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            # 2. BERT Textual Extractor
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.bert = AutoModel.from_pretrained("bert-base-uncased")
            self.bert = self.bert.to(self.device)
            self.bert.eval()

            # 3. Multimodal Classifier (CNN + BERT + BAM)
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

            self.model = MultimodalModel50Percent(
                img_in_dim=2048,
                text_in_dim=768,
                proj_dim=512,
                hidden_dim=256,
                num_classes=2
            ).to(self.device)

            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            self.optimal_threshold = float(checkpoint.get('optimal_threshold', 0.41))
            self.model_loaded = True
            self._initialized = True
            print(f"[InferenceService] Checkpoint loaded successfully! (Calibrated Threshold: {self.optimal_threshold:.2f})")

        except Exception as e:
            self.error_message = str(e)
            print(f"[InferenceService ERROR] Failed to initialize model: {e}")

    def predict(self, image_bytes: bytes, text: str) -> dict:
        """
        Executes real multimodal inference on the submitted image binary and text caption.
        """
        if not self.model_loaded:
            raise RuntimeError(f"Model is not loaded: {self.error_message}")

        # 1. Process & Validate Image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image format or corrupted file: {e}")

        img_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        # 2. Process Text (Optional - supports multimodal or image-only inference)
        clean_text = text.strip() if text else ""

        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 3. Model Forward Pass
        with torch.no_grad():
            # Feature extraction
            img_feature = self.cnn(img_tensor) # [1, 2048]
            bert_output = self.bert(**inputs)
            text_feature = bert_output.last_hidden_state[:, 0, :] # [1, 768]

            # Multimodal BAM fusion and classification
            logits, fused_feats, img_attn, text_attn = self.model(img_feature, text_feature)
            probs = F.softmax(logits, dim=1)
            
            p_non_hateful = probs[0, 0].item() * 100.0
            p_hateful = probs[0, 1].item() * 100.0

            # Calibrated Decision Rule:
            # Predict Hateful (class 1) if p_hateful >= optimal_threshold
            prob_hateful_fraction = p_hateful / 100.0
            is_hateful = prob_hateful_fraction >= self.optimal_threshold
            predicted_class = 1 if is_hateful else 0
            label = "Hateful Content Detected" if is_hateful else "No Hateful Content Detected"
            
            confidence = p_hateful if is_hateful else p_non_hateful

            img_attn_pct = img_attn.item() * 100.0
            txt_attn_pct = text_attn.item() * 100.0

        return {
            "success": True,
            "label": label,
            "predicted_class": predicted_class,
            "probability_hateful": round(p_hateful, 2),
            "probability_non_hateful": round(p_non_hateful, 2),
            "confidence": round(confidence, 2),
            "calibrated_threshold": round(self.optimal_threshold, 4),
            "attention": {
                "visual": round(img_attn_pct, 2),
                "textual": round(txt_attn_pct, 2)
            },
            "model_name": "ResNet50 + BERT + BAM",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }


# Global accessor
def get_inference_service() -> InferenceService:
    return InferenceService()
