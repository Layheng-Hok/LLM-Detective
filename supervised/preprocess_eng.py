import os
import glob
import json
import random
from sklearn.model_selection import train_test_split

def load_domain_data(domain, balance=False):
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
    if balance:
        human_data = [d for d in data if d['label'] == 0]
        llm_data = [d for d in data if d['label'] == 1]
        if len(llm_data) > len(human_data):
            llm_data = random.sample(llm_data, len(human_data))
        data = human_data + llm_data
        random.shuffle(data)
    return data

# Load and split essay data for training and validation with balancing
essay_data = load_domain_data('essay', balance=True)
train_data, val_data = train_test_split(
    essay_data, 
    test_size=0.2, 
    stratify=[d['label'] for d in essay_data], 
    random_state=42
)

# Load wp data for testing with balancing
test_data = load_domain_data('wp', balance=True)

os.makedirs('./../datasets/ghostbuster-data_split', exist_ok=True)

# Save datasets to JSON lines files
with open('./../datasets/ghostbuster-data_split/train.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

with open('./../datasets/ghostbuster-data_split/val.jsonl', 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + '\n')

with open('./../datasets/ghostbuster-data_split/test.jsonl', 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')

print("Preprocessing complete. Datasets saved to ./../datasets/ghostbuster-data_split.")
