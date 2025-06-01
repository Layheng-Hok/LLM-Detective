import subprocess
import os

# Directories
input_dir = "./nll_outputs"
output_dir = "./fft_outputs"

# Input and output file pairs
file_pairs = [
    # English
    "train_human_eng.nll.txt",
    "train_model_eng.nll.txt",
    "val_human_eng.nll.txt",
    "val_model_eng.nll.txt",
    "test_eng.nll.txt",

    # Chinese
    "train_human_chn.nll.txt",
    "train_model_chn.nll.txt",
    "val_human_chn.nll.txt",
    "val_model_chn.nll.txt",
    "test_chn.nll.txt",

    # Multilingual
    "train_human_mul.nll.txt",
    "train_model_mul.nll.txt",
    "val_human_mul.nll.txt",
    "val_model_mul.nll.txt",
    "test_mul.nll.txt",
]

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Run the FFT script for each input file
for input_file in file_pairs:
    input_path = os.path.join(input_dir, input_file)
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}.nllzs.fftnorm.txt"
    output_path = os.path.join(output_dir, output_file)

    cmd = [
        "python", "run_fft.py",
        "-i", input_path,
        "-o", output_path,
        "-p", "zscore",
        "--value", "norm"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[ERROR] Failed on: {input_file}")
    else:
        print(f"[DONE] {output_file} generated.")

print("All FFT computations completed.")
