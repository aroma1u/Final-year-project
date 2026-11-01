import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = AutoModel.from_pretrained(
    "bert-base-uncased"
)

model = model.to(device)
model.eval()

class MemeBERTClassifier(nn.Module):
    """
    BERT-based Text Classifier / Feature Extractor for Meme Text.
    Uses Hugging Face transformers backbone (bert-base-uncased by default).
    """
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 2, extract_features: bool = False):
        super(MemeBERTClassifier, self).__init__()
        self.extract_features = extract_features
        self.model_name = model_name
        self.bert = AutoModel.from_pretrained(model_name)
        in_features = self.bert.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        
        if self.extract_features:
            return cls_rep
        logits = self.classifier(cls_rep)
        return logits

def get_bert_tokenizer(model_name: str = 'bert-base-uncased'):
    return AutoTokenizer.from_pretrained(model_name)

if __name__ == "__main__":
    print(f"BERT model initialized on device: {device}")
    dummy_text = ["its their character not their color that matters"]
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_representation = outputs.last_hidden_state[:, 0, :]
    print(f"BERT [CLS] token feature shape: {cls_representation.shape}")  # Expect [1, 768]
