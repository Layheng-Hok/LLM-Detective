import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import os

print("Evaluation of NLL-based zero-shot detection:")

DIR = './fft_outputs'

def load_spectrum(file):
    """Load spectrum data from a CSV file."""
    return pd.read_csv(file)

def get_power_k(df, k):
    """Compute the sum of power for the first k frequencies per text."""
    powers = []
    for sid, group in df.groupby('sid'):
        power_k = np.sum(group['power'].values[:k])
        powers.append(power_k)
    return np.array(powers)

def evaluate_dataset(human_val_file, model_val_file, test_file, test_jsonl):
    # Add directory prefix
    human_val_file = os.path.join(DIR, human_val_file)
    model_val_file = os.path.join(DIR, model_val_file)
    test_file = os.path.join(DIR, test_file)

    # Load validation and test spectrum data
    val_human_df = load_spectrum(human_val_file)
    val_model_df = load_spectrum(model_val_file)
    test_df = load_spectrum(test_file)

    # Select optimal k using validation data
    best_acc, best_k, best_higher, best_threshold = 0, None, None, None
    for k in range(1, 51):
        val_human_powers = get_power_k(val_human_df, k)
        val_model_powers = get_power_k(val_model_df, k)
        
        # Compute threshold as the midpoint between means
        mean_human = np.mean(val_human_powers)
        mean_model = np.mean(val_model_powers)
        threshold = (mean_human + mean_model) / 2
        
        # Test assumption: higher = 'human'
        pred_human_human = val_human_powers >= threshold
        pred_model_human = val_model_powers < threshold
        acc_human = (np.mean(pred_human_human) * len(val_human_powers) + 
                     np.mean(pred_model_human) * len(val_model_powers)) / \
                    (len(val_human_powers) + len(val_model_powers))
        
        # Test assumption: higher = 'model'
        pred_human_model = val_human_powers < threshold
        pred_model_model = val_model_powers >= threshold
        acc_model = (np.mean(pred_human_model) * len(val_human_powers) + 
                     np.mean(pred_model_model) * len(val_model_powers)) / \
                    (len(val_human_powers) + len(val_model_powers))
        
        # Update best if acc_human is better
        if acc_human > best_acc:
            best_acc = acc_human
            best_k = k
            best_higher = 'human'
            best_threshold = threshold
        
        # Update best if acc_model is better
        if acc_model > best_acc:
            best_acc = acc_model
            best_k = k
            best_higher = 'model'
            best_threshold = threshold

    print(f'\nBest k: {best_k}, Validation Accuracy: {best_acc:.4f}, Higher: {best_higher}')

    # Classify test data
    test_powers = get_power_k(test_df, best_k)
    if best_higher == 'model':
        preds = test_powers >= best_threshold
        scores = test_powers  # Higher score -> more likely model
    else:
        preds = test_powers < best_threshold
        scores = -test_powers  # Lower power -> more likely model

    # Load true labels
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        true_labels = [json.loads(line)['label'] for line in f]
    true_labels = np.array(true_labels)

    # Compute metrics
    acc = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    auroc = roc_auc_score(true_labels, scores)

    return {
        'Accuracy': acc, 'Precision': precision, 'Recall': recall,
        'F1': f1, 'AUROC': auroc, 'Best k': best_k, 'Higher': best_higher
    }

# Evaluate English dataset
eng_results = evaluate_dataset(
    'val_human_eng.nll.nllzs.fftnorm.txt', 'val_model_eng.nll.nllzs.fftnorm.txt',
    'test_eng.nll.nllzs.fftnorm.txt', './../datasets/ghostbuster-data_split/test.jsonl'
)
print("English Results:")
for metric, value in eng_results.items():
    print(f"{metric}: {value}")

# Evaluate Chinese dataset
chn_results = evaluate_dataset(
    'val_human_chn.nll.nllzs.fftnorm.txt', 'val_model_chn.nll.nllzs.fftnorm.txt',
    'test_chn.nll.nllzs.fftnorm.txt', './../datasets/face2_zh_split/test.jsonl'
)
print("Chinese Results:")
for metric, value in chn_results.items():
    print(f"{metric}: {value}")

# Evaluate multilingual dataset
mul_results = evaluate_dataset(
    'val_human_mul.nll.nllzs.fftnorm.txt', 'val_model_mul.nll.nllzs.fftnorm.txt',
    'test_mul.nll.nllzs.fftnorm.txt', './../datasets/mul_split/test.jsonl'
)
print("Multilingual Results:")
for metric, value in mul_results.items():
    print(f"{metric}: {value}")
