import os
import re
import csv

# Ensure output directory exists
output_dir = './viz/test'
os.makedirs(output_dir, exist_ok=True)

# Full names for datasets
log_paths = {
    'English': './supervised/logs/nlp-test-bert-eng-job45830.o',
    'Chinese': './supervised/logs/nlp-test-bert-chn-job44179.o',
    'Multilingual': './supervised/logs/nlp-test-bert-mul-job45829.o'
}

# Regex to extract final result line
result_pattern = re.compile(
    r"results:\s*\{[^}]*accuracy':\s*([\d.]+)[^}]*"
    r"precision':\s*([\d.]+)[^}]*"
    r"recall':\s*([\d.]+)[^}]*"
    r"f1':\s*([\d.]+)[^}]*"
    r"auroc':\s*np\.float64\(([\d.]+)\)"
)

# Output CSV path
output_csv = os.path.join(output_dir, 'ood_results.csv')

# Write extracted results to CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['dataset', 'accuracy', 'precision', 'recall', 'f1', 'auroc'])

    for dataset, path in log_paths.items():
        try:
            with open(path, 'r') as file:
                content = file.read()
                match = result_pattern.search(content)
                if match:
                    metrics = list(map(float, match.groups()))
                    row = [dataset] + metrics
                    writer.writerow(row)
                else:
                    print(f"[WARN] No results found in {path}")
        except FileNotFoundError:
            print(f"[ERROR] File not found: {path}")

print(f"Results saved to {output_csv}")
