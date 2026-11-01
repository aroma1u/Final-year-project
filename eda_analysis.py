import json
import os
from collections import Counter
from datasets import load_dataset
from PIL import Image

def run_eda():
    print("=" * 60)
    print("Running Exploratory Data Analysis (EDA) on Hateful Memes...")
    print("=" * 60)
    
    ds = load_dataset("cs5242-hateful-memes/hateful-memes-data")
    stats = {}

    for split_name in ds.keys():
        split_ds = ds[split_name]
        texts = split_ds['text']
        labels = split_ds['label'] if 'label' in split_ds.column_names else []
        
        # Text statistics
        word_counts = [len(text.split()) for text in texts]
        char_counts = [len(text) for text in texts]
        
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        max_words = max(word_counts) if word_counts else 0
        min_words = min(word_counts) if word_counts else 0

        # Vocabulary top words
        all_words = [w.lower() for text in texts for w in text.split()]
        vocab_counter = Counter(all_words)

        # Image aspect ratios & dimensions sample
        image_dims = []
        sample_size = min(100, len(split_ds))
        for i in range(sample_size):
            img = split_ds[i]['image']
            if isinstance(img, Image.Image):
                image_dims.append(img.size)

        split_stat = {
            "num_rows": len(split_ds),
            "text_stats": {
                "avg_words_per_meme": round(avg_words, 2),
                "min_words": min_words,
                "max_words": max_words,
                "total_vocab_words": len(vocab_counter),
                "top_10_words": vocab_counter.most_common(10)
            },
            "class_distribution": dict(Counter(labels)) if labels else {},
            "sample_image_dims": {
                "min_width": min(d[0] for d in image_dims) if image_dims else None,
                "max_width": max(d[0] for d in image_dims) if image_dims else None,
                "min_height": min(d[1] for d in image_dims) if image_dims else None,
                "max_height": max(d[1] for d in image_dims) if image_dims else None,
            }
        }
        stats[split_name] = split_stat

        print(f"\n--- Split: '{split_name}' ---")
        print(f"  Total samples: {split_stat['num_rows']}")
        print(f"  Avg words/meme: {split_stat['text_stats']['avg_words_per_meme']}")
        print(f"  Word count range: [{min_words}, {max_words}]")
        print(f"  Class Distribution: {split_stat['class_distribution']}")
        print(f"  Top 5 words: {[w[0] for w in split_stat['text_stats']['top_10_words'][:5]]}")

    os.makedirs("results", exist_ok=True)
    results_file = os.path.join("results", "eda_summary.json")
    with open(results_file, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print(f"EDA Summary saved to {results_file}")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()
