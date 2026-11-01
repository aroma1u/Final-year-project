import torch
from datasets import load_dataset
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ==========================================
# 1. DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MSCMGTB - 20% IMPLEMENTATION")
print("=" * 60)

print("Device:", device)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

dataset = load_dataset(
    "cs5242-hateful-memes/hateful-memes-data"
)

print(dataset)

train_dataset = dataset["train"]

print("\nTraining samples:", len(train_dataset))


# ==========================================
# 3. CNN MODEL
# ==========================================

print("\nLoading CNN...")

cnn = models.resnet50(weights="DEFAULT")

# Remove final classification layer
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

print("CNN loaded successfully.")


# ==========================================
# 4. BERT MODEL
# ==========================================

print("\nLoading BERT...")

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

bert = AutoModel.from_pretrained(
    "bert-base-uncased"
)

bert = bert.to(device)
bert.eval()

print("BERT loaded successfully.")


# ==========================================
# 5. TEST ONE SAMPLE
# ==========================================

print("\n" + "=" * 60)
print("TESTING ONE MULTIMODAL SAMPLE")
print("=" * 60)

sample = train_dataset[0]

image = sample["image"].convert("RGB")
text = sample["text"]
label = sample["label"]

print("\nSample ID:", sample["id"])
print("Text:", text)
print("Label:", label)


# ==========================================
# 6. IMAGE FEATURE EXTRACTION
# ==========================================

image_tensor = image_transform(image)

image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.to(device)

with torch.no_grad():

    image_features = cnn(image_tensor)

print("\nImage feature shape:")
print(image_features.shape)


# ==========================================
# 7. TEXT FEATURE EXTRACTION
# ==========================================

inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():

    outputs = bert(**inputs)

text_features = outputs.last_hidden_state[:, 0, :]

print("\nText feature shape:")
print(text_features.shape)


# ==========================================
# 8. FINAL OUTPUT
# ==========================================

print("\n" + "=" * 60)
print("20% PIPELINE TEST COMPLETED")
print("=" * 60)

print("\nImage:")
print("Input -> ResNet50 ->", image_features.shape)

print("\nText:")
print("Input -> BERT ->", text_features.shape)

print("\nLabel:", label)

print("\nNext stage:")
print("CNN Features + BERT Features -> BAM Fusion")


# ==========================================
# 9. PROCESS MULTIPLE SAMPLES
# ==========================================

NUM_SAMPLES = 100

image_features_list = []
text_features_list = []
labels_list = []

print("\nProcessing", NUM_SAMPLES, "samples...")

for i in tqdm(range(NUM_SAMPLES)):

    sample = train_dataset[i]

    # Image
    image = sample["image"].convert("RGB")

    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        image_feature = cnn(image_tensor)

    # Text
    text = sample["text"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = bert(**inputs)

    text_feature = outputs.last_hidden_state[:, 0, :]

    # Save
    image_features_list.append(
        image_feature.cpu()
    )

    text_features_list.append(
        text_feature.cpu()
    )

    labels_list.append(sample["label"])


# Combine
image_features = torch.cat(
    image_features_list,
    dim=0
)

text_features = torch.cat(
    text_features_list,
    dim=0
)

labels = torch.tensor(labels_list)


print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 60)

print("Image features:", image_features.shape)
print("Text features:", text_features.shape)
print("Labels:", labels.shape)

import os
os.makedirs("features", exist_ok=True)

torch.save(
    image_features,
    "features/image_features.pt"
)

torch.save(
    text_features,
    "features/text_features.pt"
)

torch.save(
    labels,
    "features/labels.pt"
)

print("\nFeatures saved successfully!")
