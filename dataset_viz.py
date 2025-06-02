import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sns.set(style="whitegrid")

# List of dataset folders inside ./datasets
dataset_folders = [
    'ghostbuster-data_split',
    'face2_zh_split',
    'mul_split'
]

DATA_ROOT = './datasets'
VIZ_ROOT = './viz'

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def visualize_dataset(folder_name, viz_dir):
    folder_path = os.path.join(DATA_ROOT, folder_name)
    splits = ['train.jsonl', 'val.jsonl', 'test.jsonl']
    all_data = {}

    for split_file in splits:
        path = os.path.join(folder_path, split_file)
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist, skipping.")
            continue
        data = load_jsonl(path)
        all_data[split_file.split('.')[0]] = data

    # Plot number of samples per split
    counts = {split: len(data) for split, data in all_data.items()}
    if counts:
        plt.figure(figsize=(6,4))
        sns.barplot(x=list(counts.keys()), y=list(counts.values()), color="skyblue")
        plt.title(f"Number of samples per split in {folder_name}")
        plt.ylabel("Number of samples")
        plt.savefig(os.path.join(viz_dir, f'{folder_name}_sample_counts.png'))
        plt.close()

    # Plot label distributions per split and print counts
    for split, data in all_data.items():
        labels = []
        for entry in data:
            for key in ['label', 'category', 'class']:
                if key in entry:
                    labels.append(entry[key])
                    break
        
        if labels:
            label_counts = Counter(labels)
            print(f"\nLabel distribution in {folder_name} - {split}:")
            for label, count in label_counts.items():
                print(f"{label}: {count}")

            plt.figure(figsize=(8,4))
            sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()), color="skyblue")
            plt.title(f"Label distribution in {folder_name} - {split}")
            plt.xticks(rotation=45)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{folder_name}_label_distribution_{split}.png'))
            plt.close()
        else:
            print(f"No labels found for {folder_name} - {split}")

    # Combined label distribution for all splits
    combined_labels = []
    for data in all_data.values():
        for entry in data:
            for key in ['label', 'category', 'class']:
                if key in entry:
                    combined_labels.append(entry[key])
                    break

    if combined_labels:
        combined_counts = Counter(combined_labels)
        print(f"\nCombined label distribution in {folder_name} (train+val+test):")
        for label, count in combined_counts.items():
            print(f"{label}: {count}")

        plt.figure(figsize=(8,4))
        sns.barplot(x=list(combined_counts.keys()), y=list(combined_counts.values()), color="skyblue")
        plt.title(f"Combined label distribution in {folder_name} (train+val+test)")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'{folder_name}_label_distribution_combined.png'))
        plt.close()

if __name__ == "__main__":
    os.makedirs(VIZ_ROOT, exist_ok=True)

    for folder in dataset_folders:
        print(f"\nVisualizing {folder}...")
        visualize_dataset(folder, VIZ_ROOT)
    print(f"All visualizations saved in '{VIZ_ROOT}' folder.")
