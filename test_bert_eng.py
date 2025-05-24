import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.special import softmax

# Load datasets
val_dataset = load_dataset('json', data_files='./datasets/ghostbuster-data_split/val.jsonl')['train']
test_dataset = load_dataset('json', data_files='./datasets/ghostbuster-data_split/test.jsonl')['train']

# Load tokenizer and fine-tuned model
tokenizer = BertTokenizer.from_pretrained('./finetuned/bert-eng')
model = BertForSequenceClassification.from_pretrained('./finetuned/bert-eng', num_labels=2)
model.to('cuda')
print(f"Model is on device: {next(model.parameters()).device}")

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Define metrics computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probas = softmax(pred.predictions, axis=1)[:,1]
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds),
        'auroc': roc_auc_score(labels, probas)
    }

# Training arguments for evaluation
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=16,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
)

# Evaluate on validation set (in-domain)
val_pred = trainer.predict(val_dataset)
val_metrics = compute_metrics(val_pred)
print("In-domain (essay validation) results:", val_metrics)

# Evaluate on test set (OOD)
test_pred = trainer.predict(test_dataset)
test_metrics = compute_metrics(test_pred)
print("OOD (wp test) results:", test_metrics)
