import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.special import softmax

# Load datasets
train_dataset = load_dataset('json', data_files='./../datasets/face2_zh_split/train.jsonl')['train']
val_dataset = load_dataset('json', data_files='./../datasets/face2_zh_split/val.jsonl')['train']

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('./pretrained/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('./pretrained/bert-base-chinese', num_labels=2)
model.to('cuda')
print(f"Model is on device: {next(model.parameters()).device}")

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Define metrics computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probas = softmax(pred.predictions, axis=1)[:, 1]
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds),
        'auroc': roc_auc_score(labels, probas)
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./checkpoints/bert-chn',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model('./finetuned/bert-chn')
tokenizer.save_pretrained('./finetuned/bert-chn')

print("Training complete. Model saved to ./finetuned/bert-chn.")
