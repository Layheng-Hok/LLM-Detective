import os
import re
import json
import csv
import matplotlib.pyplot as plt

# === Log file paths ===
log_files = {
    "English": "./supervised/logs/nlp-train-bert-eng-job45791.o",
    "Chinese": "./supervised/logs/nlp-train-bert-chn-job44165.o",
    "Multilingual": "./supervised/logs/nlp-train-bert-mul-job45792.o",
}

# === Regular expression to extract JSON-like lines ===
json_pattern = re.compile(r"\{.*?\}")

# === Output directories ===
base_output_dir = "./viz/train"
plots_dir = os.path.join(base_output_dir, "plots")
csvs_dir = os.path.join(base_output_dir, "csvs")

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(csvs_dir, exist_ok=True)

# === Metrics to ignore ===
ignored_metrics = {
    "train_loss",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "learning_rate",
}

# === Parse a single log file and extract metrics per epoch ===
def parse_log(file_path):
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            match = json_pattern.search(line)
            if match:
                try:
                    fixed = match.group().replace("'", '"')
                    log_entry = json.loads(fixed)

                    epoch = log_entry.get("epoch")
                    if epoch is None:
                        continue

                    for key, value in log_entry.items():
                        if key == "epoch" or key in ignored_metrics:
                            continue
                        if key not in metrics:
                            metrics[key] = {}
                        metrics[key][epoch] = value
                except json.JSONDecodeError:
                    continue
    return metrics

# === Parse all logs ===
all_metrics = {name: parse_log(path) for name, path in log_files.items()}

# === Gather all unique metric names ===
metric_names = set()
for metrics in all_metrics.values():
    metric_names.update(metrics.keys())

# === Generate and save plots and CSVs ===
for metric in sorted(metric_names):
    plt.figure(figsize=(10, 5))
    found = False

    # Collect data for CSV
    all_epochs = set()
    for name, metrics in all_metrics.items():
        if metric in metrics:
            all_epochs.update(metrics[metric].keys())
    all_epochs = sorted(all_epochs)

    csv_rows = []
    for epoch in all_epochs:
        row = {'epoch': epoch}
        for name in log_files.keys():
            val = all_metrics.get(name, {}).get(metric, {}).get(epoch, None)
            row[name] = val
        csv_rows.append(row)

    # Plot
    for name, metrics in all_metrics.items():
        if metric in metrics:
            found = True
            epochs = sorted(metrics[metric].keys())
            values = [metrics[metric][ep] for ep in epochs]
            plt.plot(epochs, values, label=name)

    if found:
        # Save PNG
        plt.title(f"Metric: {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"metric_{metric}.png")
        plt.savefig(plot_path)
        print(f"✅ Saved plot: {plot_path}")
        plt.close()

        # Save CSV
        csv_path = os.path.join(csvs_dir, f"metric_{metric}.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['epoch'] + list(log_files.keys()))
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        print(f"✅ Saved CSV: {csv_path}")

print("\nAll plots and CSVs saved to ./viz/train/{plots,csvs}/")
