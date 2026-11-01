import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.loader import HatefulMemesDataset
from models.cnn import MemeCNNClassifier
from models.bert import MemeBERTClassifier, get_bert_tokenizer

def extract_and_save_features(batch_size: int = 32, force_reextract: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for feature extraction")
    
    os.makedirs("features", exist_ok=True)

    print("Initializing CNN (ResNet50 - 2048d) and BERT (bert-base-uncased - 768d) feature extractors...")
    cnn_model = MemeCNNClassifier(backbone='resnet50', pretrained=True, extract_features=True).to(device)
    bert_model = MemeBERTClassifier(model_name='bert-base-uncased', extract_features=True).to(device)
    tokenizer = get_bert_tokenizer('bert-base-uncased')

    cnn_model.eval()
    bert_model.eval()

    splits = ['train', 'dev_seen', 'dev_unseen', 'test_seen', 'test_unseen']
    
    extracted_data = {}

    for split in splits:
        output_file = os.path.join("features", f"{split}_features.pt")
        
        # Check if 2048d features exist
        if os.path.exists(output_file) and not force_reextract:
            data = torch.load(output_file)
            img_shape = data['img_features'].shape
            if img_shape[1] == 2048:
                print(f"Valid 2048d features for split '{split}' already exist at {output_file}. Loading...")
                extracted_data[split] = data
                continue

        print(f"\nExtracting ResNet50 (2048d) + BERT (768d) features for split '{split}'...")
        dataset = HatefulMemesDataset(split_name=split)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        img_feats_list = []
        text_feats_list = []
        labels_list = []
        ids_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting {split}"):
                images = batch['image'].to(device)
                texts = batch['text']
                labels = batch['label']
                sample_ids = batch['id']

                # Extract image features (ResNet50: 2048d)
                img_feats = cnn_model(images).cpu()

                # Tokenize and extract BERT text features (BERT: 768d)
                inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                text_feats = bert_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).cpu()

                img_feats_list.append(img_feats)
                text_feats_list.append(text_feats)
                labels_list.append(labels)
                ids_list.extend(sample_ids)

        all_img_feats = torch.cat(img_feats_list, dim=0)
        all_text_feats = torch.cat(text_feats_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        saved_data = {
            'ids': ids_list,
            'img_features': all_img_feats,
            'text_features': all_text_feats,
            'labels': all_labels
        }
        torch.save(saved_data, output_file)
        extracted_data[split] = saved_data
        print(f"Saved {len(ids_list)} samples to {output_file} (img_shape: {all_img_feats.shape}, text_shape: {all_text_feats.shape})")

    # Create combined val (dev_seen + dev_unseen) and test (test_seen + test_unseen) files and save individual tensors
    print("\nPreparing combined train / val / test feature tensors...")
    
    # Train
    train_data = extracted_data['train']
    torch.save(train_data['img_features'], "features/train_image_features.pt")
    torch.save(train_data['text_features'], "features/train_text_features.pt")
    torch.save(train_data['labels'], "features/train_labels.pt")

    # Val (dev_seen + dev_unseen)
    val_img_features = torch.cat([extracted_data['dev_seen']['img_features'], extracted_data['dev_unseen']['img_features']], dim=0)
    val_text_features = torch.cat([extracted_data['dev_seen']['text_features'], extracted_data['dev_unseen']['text_features']], dim=0)
    val_labels = torch.cat([extracted_data['dev_seen']['labels'], extracted_data['dev_unseen']['labels']], dim=0)
    
    torch.save(val_img_features, "features/val_image_features.pt")
    torch.save(val_text_features, "features/val_text_features.pt")
    torch.save(val_labels, "features/val_labels.pt")

    # Test (test_seen + test_unseen)
    test_img_features = torch.cat([extracted_data['test_seen']['img_features'], extracted_data['test_unseen']['img_features']], dim=0)
    test_text_features = torch.cat([extracted_data['test_seen']['text_features'], extracted_data['test_unseen']['text_features']], dim=0)
    test_labels = torch.cat([extracted_data['test_seen']['labels'], extracted_data['test_unseen']['labels']], dim=0)

    torch.save(test_img_features, "features/test_image_features.pt")
    torch.save(test_text_features, "features/test_text_features.pt")
    torch.save(test_labels, "features/test_labels.pt")

    print("Feature preparation complete:")
    print(f"  Train: img {train_data['img_features'].shape}, text {train_data['text_features'].shape}, labels {train_data['labels'].shape}")
    print(f"  Val  : img {val_img_features.shape}, text {val_text_features.shape}, labels {val_labels.shape}")
    print(f"  Test : img {test_img_features.shape}, text {test_text_features.shape}, labels {test_labels.shape}")

if __name__ == "__main__":
    extract_and_save_features()
