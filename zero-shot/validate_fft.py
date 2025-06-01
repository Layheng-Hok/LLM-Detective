import os

def count_txt_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f if _.strip())  # skip blank lines

def count_fft_samples(path):
    with open(path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        sample_ids = set()
        for line in f:
            if line.strip():  # skip empty lines
                sid = line.split(',')[0]
                sample_ids.add(sid)
        return len(sample_ids)

def validate_fft(dataset_name, nll_dir, fft_dir, file_mapping):
    print(f"\nValidating FFT outputs for {dataset_name} dataset:")
    all_passed = True

    for nll_txt, fft_txt in file_mapping:
        nll_path = os.path.join(nll_dir, nll_txt)
        fft_path = os.path.join(fft_dir, fft_txt)

        if not os.path.exists(nll_path):
            print(f"  ❌ Missing NLL file: {nll_txt}")
            all_passed = False
            continue
        if not os.path.exists(fft_path):
            print(f"  ❌ Missing FFT file: {fft_txt}")
            all_passed = False
            continue

        nll_count = count_txt_lines(nll_path)
        fft_count = count_fft_samples(fft_path)

        status = "✅ OK" if nll_count == fft_count else "❌ MISMATCH"
        if status == "❌ MISMATCH":
            all_passed = False
        print(f"  {fft_txt:35}  NLL: {nll_count:<5} | FFT sids: {fft_count:<5} => {status}")
    
    if all_passed:
        print(f"✅ All FFT counts matched for {dataset_name}.")
    else:
        print(f"⚠️  Some mismatches found in {dataset_name}.")

# Directories
nll_dir = "./nll_outputs"
fft_dir = "./fft_outputs"

# Config for all datasets
file_groups = {
    "English": [
        ("train_human_eng.nll.txt", "train_human_eng.nll.nllzs.fftnorm.txt"),
        ("train_model_eng.nll.txt", "train_model_eng.nll.nllzs.fftnorm.txt"),
        ("val_human_eng.nll.txt", "val_human_eng.nll.nllzs.fftnorm.txt"),
        ("val_model_eng.nll.txt", "val_model_eng.nll.nllzs.fftnorm.txt"),
        ("test_eng.nll.txt", "test_eng.nll.nllzs.fftnorm.txt"),
    ],
    "Chinese": [
        ("train_human_chn.nll.txt", "train_human_chn.nll.nllzs.fftnorm.txt"),
        ("train_model_chn.nll.txt", "train_model_chn.nll.nllzs.fftnorm.txt"),
        ("val_human_chn.nll.txt", "val_human_chn.nll.nllzs.fftnorm.txt"),
        ("val_model_chn.nll.txt", "val_model_chn.nll.nllzs.fftnorm.txt"),
        ("test_chn.nll.txt", "test_chn.nll.nllzs.fftnorm.txt"),
    ],
    "Multilingual": [
        ("train_human_mul.nll.txt", "train_human_mul.nll.nllzs.fftnorm.txt"),
        ("train_model_mul.nll.txt", "train_model_mul.nll.nllzs.fftnorm.txt"),
        ("val_human_mul.nll.txt", "val_human_mul.nll.nllzs.fftnorm.txt"),
        ("val_model_mul.nll.txt", "val_model_mul.nll.nllzs.fftnorm.txt"),
        ("test_mul.nll.txt", "test_mul.nll.nllzs.fftnorm.txt"),
    ]
}

# Run validation for each dataset
for name, file_map in file_groups.items():
    validate_fft(name, nll_dir, fft_dir, file_map)
