import torch
from torch.utils.data import Dataset, DataLoader

class FeatureDataset(Dataset):
    """
    PyTorch Dataset for pre-extracted multimodal feature representations.
    Validates alignment across image features, text features, and labels.
    """
    def __init__(self, image_features: torch.Tensor, text_features: torch.Tensor, labels: torch.Tensor, ids=None):
        if not (len(image_features) == len(text_features) == len(labels)):
            raise ValueError(
                f"Feature alignment error! Mismatched tensor lengths: "
                f"image_features ({len(image_features)}), text_features ({len(text_features)}), labels ({len(labels)})"
            )
            
        self.image_features = image_features
        self.text_features = text_features
        self.labels = labels.long()
        self.ids = ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.image_features[idx], self.text_features[idx], self.labels[idx]

def get_feature_dataloader(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    dataset = FeatureDataset(image_features, text_features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    dummy_img = torch.randn(100, 2048)
    dummy_txt = torch.randn(100, 768)
    dummy_lbl = torch.randint(0, 2, (100,))
    ds = FeatureDataset(dummy_img, dummy_txt, dummy_lbl)
    print(f"FeatureDataset test passed! Total samples: {len(ds)}")
