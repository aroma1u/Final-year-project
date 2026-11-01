import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from torchvision import transforms

class HatefulMemesDataset(Dataset):
    """
    PyTorch Dataset wrapper for Hateful Memes loaded from Hugging Face datasets.
    """
    def __init__(self, split_name: str = 'train', img_transform=None):
        self.ds = load_dataset("cs5242-hateful-memes/hateful-memes-data", split=split_name)
        self.img_transform = img_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item['image']
        
        # Ensure RGB mode for PIL Image
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.img_transform:
            image = self.img_transform(image)
            
        text = item['text']
        label = item['label']
        sample_id = item['id']

        return {
            'id': sample_id,
            'image': image,
            'text': text,
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_dataloader(split_name: str = 'train', batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
    dataset = HatefulMemesDataset(split_name=split_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

if __name__ == "__main__":
    ds = HatefulMemesDataset(split_name='dev_seen')
    print(f"HatefulMemesDataset dev_seen split length: {len(ds)}")
    sample = ds[0]
    print(f"Sample keys: {list(sample.keys())}, Image tensor shape: {sample['image'].shape}")
