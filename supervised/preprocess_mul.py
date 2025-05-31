import os
import glob
import json
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = " ".join(text.split())
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return text

def load_english_domain(domain):
    base_path = f'./../datasets/ghostbuster-data_reformed/{domain}'
    human_files = glob.glob(f'{base_path}/human/*.txt')
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f != 'human']
    llm_files = []
    for sf in subfolders:
        llm_files.extend(glob.glob(f'{base_path}/{sf}/*.txt'))
    data = []
    for file in human_files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            data.append({'text': text, 'label': 0})
    for file in llm_files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            data.append({'text': text, 'label': 1})
    return data

def load_chinese_domain(domain):
    if domain == "news":
        human_file = "news-zh.json"
        generated_file = "news-zh.qwen2-72b-base.json"
    elif domain == "wiki":
        human_file = "wiki-zh.json"
        generated_file = "wiki-zh.qwen2-72b-base.json"
    else:
        raise ValueError("Invalid domain")
    # Load human texts
    file_path = f'./../datasets/face2_zh_json/human/zh_unicode/{human_file}'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    human_texts = [clean_text(item["input"] + item["output"]) for item in data]
    # Load generated texts
    file_path = f'./../datasets/face2_zh_json/generated/zh_qwen2/{generated_file}'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    generated_texts = [clean_text(data["input"][key] + data["output"][key]) for key in data["input"]]
    data = [{'text': text, 'label': 0} for text in human_texts] + [{'text': text, 'label': 1} for text in generated_texts]
    return data

# Load English 'essay' and Chinese 'news' for training
english_train_data = load_english_domain('essay')
chinese_train_data = load_chinese_domain('news')
train_data_combined = english_train_data + chinese_train_data

# Split into train and val
train_data, val_data = train_test_split(
    train_data_combined,
    test_size=0.2,
    stratify=[d['label'] for d in train_data_combined],
    random_state=42
)

# Load English 'wp' and Chinese 'wiki' for testing
english_test_data = load_english_domain('wp')
chinese_test_data = load_chinese_domain('wiki')
test_data = english_test_data + chinese_test_data

# Save to JSONL files
os.makedirs('./../datasets/mul_split', exist_ok=True)
with open('./../datasets/mul_split/train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
with open('./../datasets/mul_split/val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
with open('./../datasets/mul_split/test.jsonl', 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("Preprocessing complete. Datasets saved to ./../datasets/mul_split.")
