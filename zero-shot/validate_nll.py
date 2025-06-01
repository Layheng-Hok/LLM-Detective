import os

def count_txt_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def validate_nll(dataset_name, input_dir, nll_dir, file_mapping):
    print(f"\nValidating NLL outputs for {dataset_name} dataset:")
    all_passed = True

    for original_txt, nll_txt in file_mapping:
        original_path = os.path.join(input_dir, original_txt)
        nll_path = os.path.join(nll_dir, nll_txt)

        original_count = count_txt_lines(original_path)
        nll_count = count_txt_lines(nll_path)

        status = "✅ OK" if original_count == nll_count else "❌ MISMATCH"
        if status == "❌ MISMATCH":
            all_passed = False
        print(f"  {nll_txt:30}  ORIG: {original_count:<5} | NLL: {nll_count:<5} => {status}")
    
    if all_passed:
        print(f"✅ All NLL counts matched for {dataset_name}.")
    else:
        print(f"⚠️  Some mismatches found in {dataset_name}.")

# Directories
input_dir = "./preprocess_outputs"
nll_dir = "./nll_outputs"

# Config for all modes
file_groups = {
    "English": [
        ("train_human_eng.txt", "train_human_eng.nll.txt"),
        ("train_model_eng.txt", "train_model_eng.nll.txt"),
        ("val_human_eng.txt", "val_human_eng.nll.txt"),
        ("val_model_eng.txt", "val_model_eng.nll.txt"),
        ("test_eng.txt", "test_eng.nll.txt"),
    ],
    "Chinese": [
        ("train_human_chn.txt", "train_human_chn.nll.txt"),
        ("train_model_chn.txt", "train_model_chn.nll.txt"),
        ("val_human_chn.txt", "val_human_chn.nll.txt"),
        ("val_model_chn.txt", "val_model_chn.nll.txt"),
        ("test_chn.txt", "test_chn.nll.txt"),
    ],
    "Multilingual": [
        ("train_human_mul.txt", "train_human_mul.nll.txt"),
        ("train_model_mul.txt", "train_model_mul.nll.txt"),
        ("val_human_mul.txt", "val_human_mul.nll.txt"),
        ("val_model_mul.txt", "val_model_mul.nll.txt"),
        ("test_mul.txt", "test_mul.nll.txt"),
    ]
}

# Run validation for each dataset
for name, file_map in file_groups.items():
    validate_nll(name, input_dir, nll_dir, file_map)
