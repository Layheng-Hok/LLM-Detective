import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure output directory exists
output_dir = './slide_imgs'
os.makedirs(output_dir, exist_ok=True)

# Define the data
data = {
    "Setting": [
        "English (Ghostbuster)",
        "Chinese (Face2)",
        "Multilingual (Essay + News)"
    ],
    "Train/Validation Domains": [
        "Essay",
        "News",
        "Essay (EN) + News (ZH)"
    ],
    "OOD Test Domains": [
        "WP",
        "Wiki",
        "WP + Wiki"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a plot
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis('off')

# Create the table
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)

# Style adjustments
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Save the table as an image
output_path = os.path.join(output_dir, 'dataset_setup_table.png')
plt.savefig(output_path, bbox_inches='tight')

print(f"Table saved to: {output_path}")
