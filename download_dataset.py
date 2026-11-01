import os
import sys
import time
from datasets import load_dataset
from PIL import Image

def main():
    print("=" * 60)
    print("Loading dataset 'cs5242-hateful-memes/hateful-memes-data' from Hugging Face...")
    print("=" * 60)
    
    start_time = time.time()
    try:
        ds = load_dataset("cs5242-hateful-memes/hateful-memes-data")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
        
    elapsed = time.time() - start_time
    print(f"\nDataset loaded successfully in {elapsed:.2f} seconds!\n")
    
    print("Dataset Overview:")
    print("-" * 40)
    print(ds)
    print("\nAvailable Splits:")
    for split_name in ds.keys():
        print(f" - {split_name}: {len(ds[split_name])} examples")
        
    print("\nSplits Feature Schema:")
    print("-" * 40)
    first_split = list(ds.keys())[0]
    print(ds[first_split].features)
    
    print("\nSample Inspection (First 3 examples from train split):")
    print("-" * 60)
    train_ds = ds['train'] if 'train' in ds else ds[first_split]
    for i in range(min(3, len(train_ds))):
        item = train_ds[i]
        print(f"Example #{i+1}:")
        print(f"  ID: {item.get('id', 'N/A')}")
        print(f"  Text: {item.get('text', 'N/A')}")
        print(f"  Label: {item.get('label', 'N/A')}")
        print(f"  Img attribute type: {type(item.get('image'))}")
        
        # Verify image opening if img is PIL Image or file path
        img_val = item.get('image')
        if isinstance(img_val, Image.Image):
            print(f"  Image dimensions: {img_val.size}, mode: {img_val.mode}")
        elif isinstance(img_val, str) and os.path.exists(img_val):
            with Image.open(img_val) as img:
                print(f"  Image path: {img_val}, dimensions: {img.size}, mode: {img.mode}")
        elif img_val is not None:
            print(f"  Image data: {str(img_val)[:100]}")
        print()
        
    print("\nLabel Distribution Summary:")
    print("-" * 40)
    for split_name, dataset in ds.items():
        if 'label' in dataset.column_names:
            labels = dataset['label']
            label_counts = {}
            for l in labels:
                label_counts[l] = label_counts.get(l, 0) + 1
            print(f"Split '{split_name}': {label_counts}")
        else:
            print(f"Split '{split_name}': No 'label' column found")
            
    print("=" * 60)
    print("Dataset Step 1 Verification Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
