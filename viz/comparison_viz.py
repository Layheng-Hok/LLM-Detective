import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output directory exists
output_dir = "viz/test"
os.makedirs(output_dir, exist_ok=True)

# Data from the LaTeX table
data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUROC"] * 2,
    "Setting": ["Supervised"] * 5 + ["Zero-Shot"] * 5,
    "English": [0.8539, 0.8582, 0.8475, 0.8528, 0.9297,
                0.5317, 0.5434, 0.3971, 0.4589, 0.5440],
    "Chinese": [0.7665, 0.6976, 0.9410, 0.8012, 0.8715,
                0.5559, 0.7182, 0.1840, 0.2929, 0.6502],
    "Multilingual": [0.5409, 0.5215, 0.9871, 0.6825, 0.5509,
                     0.5443, 0.6916, 0.1598, 0.2596, 0.6055],
}

# Convert to DataFrame and reshape
df = pd.DataFrame(data)
df_melted = df.melt(id_vars=["Metric", "Setting"], var_name="Language", value_name="Score")

# Use custom light color palette
custom_palette = {"Supervised": "skyblue", "Zero-Shot": "salmon"}

sns.set(style="whitegrid", font_scale=1.1)

# Create individual plots for each metric
for metric in df["Metric"].unique():
    plt.figure(figsize=(8, 6))
    subset = df_melted[df_melted["Metric"] == metric]
    sns.barplot(data=subset, x="Language", y="Score", hue="Setting", palette=custom_palette)
    plt.title(f"{metric} — Supervised vs Zero-Shot", fontsize=14)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    filename = f"{metric.lower().replace(' ', '_')}_comparison.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
