import json
import os

def count_jsonl_lines(jsonl_path, label=None):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        if label is None:
            return sum(1 for _ in f)
        else:
            return sum(1 for line in f if json.loads(line)['label'] == label)

def count_txt_lines(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def validate_counts(dataset_name, jsonl_base, output_dir, checks):
    print(f"\nValidating {dataset_name} dataset:")
    all_passed = True
    for split, label, txt_file in checks:
        jsonl_path = os.path.join(jsonl_base, f"{split}.jsonl")
        txt_path = os.path.join(output_dir, txt_file)

        jsonl_count = count_jsonl_lines(jsonl_path, label)
        txt_count = count_txt_lines(txt_path)

        status = "✅ OK" if jsonl_count == txt_count else "❌ MISMATCH"
        if status == "❌ MISMATCH":
            all_passed = False
        print(f"  {txt_file:25}  JSONL: {jsonl_count:<5} | TXT: {txt_count:<5} => {status}")
    
    if all_passed:
        print(f"✅ All counts matched for {dataset_name}.")
    else:
        print(f"⚠️  Some mismatches found in {dataset_name}.")

# Config
output_dir = './preprocess_outputs'

validate_counts(
    "English",
    './../datasets/ghostbuster-data_split',
    output_dir,
    [
        ('train', 0, 'train_human_eng.txt'), ('train', 1, 'train_model_eng.txt'),
        ('val', 0, 'val_human_eng.txt'),     ('val', 1, 'val_model_eng.txt'),
        ('test', None, 'test_eng.txt'),
    ]
)

validate_counts(
    "Chinese",
    './../datasets/face2_zh_split',
    output_dir,
    [
        ('train', 0, 'train_human_chn.txt'), ('train', 1, 'train_model_chn.txt'),
        ('val', 0, 'val_human_chn.txt'),     ('val', 1, 'val_model_chn.txt'),
        ('test', None, 'test_chn.txt'),
    ]
)

validate_counts(
    "Multilingual",
    './../datasets/mul_split',
    output_dir,
    [
        ('train', 0, 'train_human_mul.txt'), ('train', 1, 'train_model_mul.txt'),
        ('val', 0, 'val_human_mul.txt'),     ('val', 1, 'val_model_mul.txt'),
        ('test', None, 'test_mul.txt'),
    ]
)
