import json
import os

def extract_texts(jsonl_file, output_file, label=None):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(jsonl_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out:
        lines_written = 0
        for line in f:
            data = json.loads(line)
            if label is None or data['label'] == label:
                text = data['text'].replace('\n', ' ')  # Ensure single-line output
                out.write(text + '\n')
                lines_written += 1
        print(f"Written {lines_written} lines to {output_file}")
    return lines_written

# Folder to store extracted text files
output_dir = './preprocess_outputs'

# English dataset
base_eng = './../datasets/ghostbuster-data_split'
for split, label, suffix in [
    ('train', 0, 'train_human_eng.txt'), ('train', 1, 'train_model_eng.txt'),
    ('val', 0, 'val_human_eng.txt'), ('val', 1, 'val_model_eng.txt'),
    ('test', None, 'test_eng.txt')
]:
    extract_texts(f'{base_eng}/{split}.jsonl', f'{output_dir}/{suffix}', label)

# Chinese dataset 
base_chn = './../datasets/face2_zh_split'
for split, label, suffix in [
    ('train', 0, 'train_human_chn.txt'), ('train', 1, 'train_model_chn.txt'),
    ('val', 0, 'val_human_chn.txt'), ('val', 1, 'val_model_chn.txt'),
    ('test', None, 'test_chn.txt')
]:
    extract_texts(f'{base_chn}/{split}.jsonl', f'{output_dir}/{suffix}', label)

# Multilingual dataset 
base_mul = './../datasets/mul_split'
for split, label, suffix in [
    ('train', 0, 'train_human_mul.txt'), ('train', 1, 'train_model_mul.txt'),
    ('val', 0, 'val_human_mul.txt'), ('val', 1, 'val_model_mul.txt'),
    ('test', None, 'test_mul.txt')
]:
    extract_texts(f'{base_mul}/{split}.jsonl', f'{output_dir}/{suffix}', label)

print("Text extraction complete. Files saved to:", output_dir)
