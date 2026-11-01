import torch
import torch.nn as nn
from torchvision import models, transforms

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = models.resnet50(weights="DEFAULT")

# Remove final classification layer
model.fc = torch.nn.Identity()

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class MemeCNNClassifier(nn.Module):
    """
    CNN-based Image Classifier / Feature Extractor for Meme Images.
    Uses a pre-trained ResNet backbone (ResNet-50 by default).
    """
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True, num_classes: int = 2, extract_features: bool = False):
        super(MemeCNNClassifier, self).__init__()
        self.extract_features = extract_features
        
        if backbone == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.cnn = models.resnet50(weights=weights)
            in_features = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        elif backbone == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.cnn = models.resnet18(weights=weights)
            in_features = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images)
        if self.extract_features:
            return features
        logits = self.classifier(features)
        return logits

if __name__ == "__main__":
    print(f"ResNet50 CNN model initialized on device: {device}")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        extracted_feats = model(dummy_input)
    print(f"Extracted feature shape: {extracted_feats.shape}")  # Expect [2, 2048]
