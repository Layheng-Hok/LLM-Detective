import subprocess
import os
import sys

# Check for user input
if len(sys.argv) != 2 or sys.argv[1] not in {"eng", "chn", "mul"}:
    print("Usage: python all_nll.py [eng|chn|mul]")
    sys.exit(1)

mode = sys.argv[1]

# Directories 
input_dir = "./preprocess_outputs"
output_dir = "./nll_outputs"

# File groups by mode
file_groups = {
    "eng": [
        ("train_human_eng.txt", "train_human_eng.nll.txt"),
        ("train_model_eng.txt", "train_model_eng.nll.txt"),
        ("val_human_eng.txt", "val_human_eng.nll.txt"),
        ("val_model_eng.txt", "val_model_eng.nll.txt"),
        ("test_eng.txt", "test_eng.nll.txt"),
    ],
    "chn": [
        ("train_human_chn.txt", "train_human_chn.nll.txt"),
        ("train_model_chn.txt", "train_model_chn.nll.txt"),
        ("val_human_chn.txt", "val_human_chn.nll.txt"),
        ("val_model_chn.txt", "val_model_chn.nll.txt"),
        ("test_chn.txt", "test_chn.nll.txt"),
    ],
    "mul": [
        ("train_human_mul.txt", "train_human_mul.nll.txt"),
        ("train_model_mul.txt", "train_model_mul.nll.txt"),
        ("val_human_mul.txt", "val_human_mul.nll.txt"),
        ("val_model_mul.txt", "val_model_mul.nll.txt"),
        ("test_mul.txt", "test_mul.nll.txt"),
    ]
}

# Model paths
model_paths = {
    "eng": "./pretrained/mistral-7b",
    "chn": "./pretrained/qwen-7b",
    "mul": "./pretrained/qwen-7b",
}

# Get selected files and model path
file_pairs = file_groups[mode]
model_path = model_paths[mode]

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Run the NLL script
for input_file, output_file in file_pairs:
    input_path = os.path.join(input_dir, input_file)
    output_path = os.path.join(output_dir, output_file)

    cmd = [
        "python", "run_nll.py",
        "-i", input_path,
        "-o", output_path,
        "--model_path", model_path,
        "--model", "custom"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[ERROR] Failed on: {input_file}")
    else:
        print(f"[DONE] {output_file} generated.")

print(f"All NLL computations completed for {mode.upper()} dataset.")
