import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from models.multimodal_model import MultimodalModel50Percent

def load_inference_models(device):
    """Loads ResNet50 visual extractor, BERT textual extractor, and trained multimodal model."""
    # 1. ResNet50 visual extractor
    cnn = models.resnet50(weights="DEFAULT")
    cnn.fc = torch.nn.Identity()
    cnn = cnn.to(device)
    cnn.eval()

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 2. BERT textual extractor
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased")
    bert = bert.to(device)
    bert.eval()

    # 3. Multimodal Model Checkpoint
    ckpt_path = os.path.join("checkpoints", "best_multimodal_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Saved model checkpoint not found at '{ckpt_path}'. Run training first!")

    multimodal_model = MultimodalModel50Percent(
        img_in_dim=2048,
        text_in_dim=768,
        proj_dim=512,
        hidden_dim=256,
        num_classes=2
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    multimodal_model.load_state_dict(checkpoint['model_state_dict'])
    multimodal_model.eval()

    optimal_threshold = checkpoint.get('optimal_threshold', 0.50)

    return cnn, image_transform, tokenizer, bert, multimodal_model, optimal_threshold


def predict_sample(image_path: str, text: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nLoading models for inference...")
    cnn, image_transform, tokenizer, bert, multimodal_model, optimal_threshold = load_inference_models(device)

    # 1. Process Image
    if not os.path.exists(image_path):
        print(f"Warning: Image file '{image_path}' not found. Creating a synthetic input for demonstration.")
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    else:
        img = Image.open(image_path).convert('RGB')

    img_tensor = image_transform(img).unsqueeze(0).to(device)

    # 2. Process Text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 3. Extract Features
    with torch.no_grad():
        img_feature = cnn(img_tensor) # [1, 2048]
        bert_output = bert(**inputs)
        text_feature = bert_output.last_hidden_state[:, 0, :] # [1, 768]

        # 4. Multimodal Model Inference
        logits, fused_feats, img_attn, text_attn = multimodal_model(img_feature, text_feature)
        probs = F.softmax(logits, dim=1)
        prob_hateful = probs[0, 1].item()
        
        # Decision using calibrated threshold
        pred_idx = 1 if prob_hateful >= optimal_threshold else 0
        confidence = (prob_hateful if pred_idx == 1 else (1.0 - prob_hateful)) * 100.0

    classes = {0: "Non-Hateful", 1: "Hateful"}
    prediction = classes[pred_idx]
    img_attn_pct = img_attn.item() * 100.0
    txt_attn_pct = text_attn.item() * 100.0

    print("\n" + "=" * 55)
    print("MULTIMODAL CONTENT MODERATION INFERENCE (RECONSTRUCTED)")
    print("=" * 55)
    print(f"Input Text           : {text}")
    print(f"Input Image          : {image_path}")
    print("-" * 55)
    print(f"Prediction           : {prediction}")
    print(f"P(Hateful)           : {prob_hateful * 100:.2f}%")
    print(f"Calibrated Threshold : {optimal_threshold:.2f}")
    print(f"Confidence           : {confidence:.2f}%")
    print(f"Visual Attention     : {img_attn_pct:.2f}%")
    print(f"Textual Attention    : {txt_attn_pct:.2f}%")
    print("=" * 55)

    return {
        'prediction': prediction,
        'probability_hateful': prob_hateful,
        'threshold': optimal_threshold,
        'confidence': confidence,
        'image_attention': img_attn_pct,
        'text_attention': txt_attn_pct
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Content Moderation Single Sample Inference")
    parser.add_argument("--image", type=str, default="sample_image.jpg", help="Path to input image file")
    parser.add_argument("--text", type=str, default="its their character not their color that matters", help="Input text caption")
    args = parser.parse_args()

    predict_sample(args.image, args.text)
