import os
import matplotlib.pyplot as plt

# Ensure output directory exists
output_dir = './slide_imgs'
os.makedirs(output_dir, exist_ok=True)

# Data before undersampling
labels_before = {'LLM-generated (1)': 24024, 'Human-written (0)': 4004}
labels_after = {'LLM-generated (1)': 4004, 'Human-written (0)': 4004}

def plot_label_distribution(data, title, filename):
    plt.figure(figsize=(5, 4))
    plt.bar(data.keys(), data.values(), color=['#FF6B6B', '#4ECDC4'])
    plt.title(title, fontsize=12)
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

# Plot and save both figures
plot_label_distribution(labels_before, 'Label Distribution Before Undersampling', 'label_dist_before.png')
plot_label_distribution(labels_after, 'Label Distribution After Undersampling', 'label_dist_after.png')
