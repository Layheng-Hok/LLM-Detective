import os
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Ensure output directory exists
output_dir = './viz/test'
os.makedirs(output_dir, exist_ok=True)

# Full names for datasets
log_paths = {
    'English': './logs/nlp-test-bert-eng-job45830.o',
    'Chinese': './logs/nlp-test-bert-chn-job44179.o',
    'Multilingual': './logs/nlp-test-bert-mul-job45829.o'
}

# Regex to extract final result line
result_pattern = re.compile(
    r"results:\s*\{[^}]*accuracy':\s*([\d.]+)[^}]*"
    r"precision':\s*([\d.]+)[^}]*"
    r"recall':\s*([\d.]+)[^}]*"
    r"f1':\s*([\d.]+)[^}]*"
    r"auroc':\s*np\.float64\(([\d.]+)\)"
)

# Paths to outputs
output_csv = os.path.join(output_dir, 'ood_results.csv')
plot_grouped_path = os.path.join(output_dir, 'ood_results_grouped.png')
plot_subplots_path = os.path.join(output_dir, 'ood_results_subplots.png')

# Collect data
rows = []

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['dataset', 'accuracy', 'precision', 'recall', 'f1', 'auroc'])

    for dataset, path in log_paths.items():
        with open(path, 'r') as file:
            content = file.read()
            match = result_pattern.search(content)
            if match:
                metrics = list(map(float, match.groups()))
                row = [dataset] + metrics
                writer.writerow(row)
                rows.append(row)
            else:
                print(f"[WARN] No results found in {path}")

print(f"✅ Results saved to {output_csv}")

# Create DataFrame
df = pd.DataFrame(rows, columns=['dataset', 'accuracy', 'precision', 'recall', 'f1', 'auroc'])
df.set_index('dataset', inplace=True)

# --- Plot 1: Grouped Bar Chart ---
ax = df.plot(kind='bar', figsize=(10, 6), colormap='Set2', rot=0)
plt.title('OOD Evaluation Metrics by Dataset')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Metric')
plt.tight_layout()
plt.savefig(plot_grouped_path)
plt.close()
print(f"📊 Grouped bar chart saved to {plot_grouped_path}")

# --- Plot 2: One plot per metric ---
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
for metric in metrics:
    fig, ax = plt.subplots(figsize=(6, 4))
    df[metric].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title(f'{metric.capitalize()} Across Languages')
    ax.set_xlabel('Dataset')
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_xticklabels(df.index, rotation=45)

    plot_metric_path = os.path.join(output_dir, f'ood_{metric}.png')
    plt.tight_layout()
    plt.savefig(plot_metric_path)
    plt.close()
    print(f"📈 {metric.capitalize()} plot saved to {plot_metric_path}")
