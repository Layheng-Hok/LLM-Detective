import json
import os
from sklearn.model_selection import train_test_split

# Clean function
def clean_text(text):
    text = " ".join(text.split())
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return text

# Data loading function
def load_texts(domain, label):
    """Load texts for a given domain and label (0 for human, 1 for generated)."""
    if domain == "news":
        human_file = "news-zh.json"
        generated_file = "news-zh.qwen2-72b-base.json"
    elif domain == "wiki":
        human_file = "wiki-zh.json"
        generated_file = "wiki-zh.qwen2-72b-base.json"
    elif domain == "webnovel":
        human_file = "webnovel.json"
        generated_file = "webnovel.qwen2-72b-base.json"
    else:
        raise ValueError("Invalid domain")

    if label == 0:  # Human-written
        file_path = f'./datasets/face2_zh_json/human/zh_unicode/{human_file}'
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [clean_text(item["input"] + item["output"]) for item in data]
    elif label == 1:  # LLM-generated
        file_path = f'./datasets/face2_zh_json/generated/zh_qwen2/{generated_file}'
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [clean_text(data["input"][key] + data["output"][key]) for key in data["input"]]
    else:
        raise ValueError("Invalid label")
    return texts

# Load and prepare "news" domain data for training and validation
human_texts_news = load_texts('news', 0)
generated_texts_news = load_texts('news', 1)
data_news = (
    [{'text': text, 'label': 0} for text in human_texts_news] +
    [{'text': text, 'label': 1} for text in generated_texts_news]
)
train_data, val_data = train_test_split(
    data_news,
    test_size=0.2,
    stratify=[d['label'] for d in data_news],
    random_state=42
)

# Load "wiki" domain data for testing (OOD)
human_texts_wiki = load_texts('wiki', 0)
generated_texts_wiki = load_texts('wiki', 1)
test_data = (
    [{'text': text, 'label': 0} for text in human_texts_wiki] +
    [{'text': text, 'label': 1} for text in generated_texts_wiki]
)

# Save datasets to JSONL files
os.makedirs('./datasets/face2_zh_split', exist_ok=True)
for dataset, filename in [
    (train_data, 'train.jsonl'),
    (val_data, 'val.jsonl'),
    (test_data, 'test.jsonl')
]:
    with open(f'./datasets/face2_zh_split/{filename}', 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("Preprocessing complete. Datasets saved to ./datasets/face2_zh_split.")
