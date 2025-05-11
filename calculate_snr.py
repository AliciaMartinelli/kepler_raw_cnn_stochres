import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

if len(sys.argv) < 2:
    print("Usage: python calculate_snr.py <sigma>")
    sys.exit(1)

try:
    sigma = float(sys.argv[1])
except ValueError:
    print("Error: sigma must be a float.")
    sys.exit(1)

original_dataset = "dataset"
augmented_dataset = os.path.join("aug_dataset", f"sigma_{sigma:.5f}")
output_csv = os.path.join("results", "snr", f"snr_sigma_{sigma:.5f}.csv")
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

def compute_snr(original, noisy):
    noise = noisy - original
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)

snr_records = []

for label in ['0', '1']:
    for view in ['global', 'local']:
        original_path = os.path.join(original_dataset, label, view)
        noisy_path = os.path.join(augmented_dataset, label, view)

        files = sorted(os.listdir(original_path))
        for file in tqdm(files, desc=f"Class {label} - {view}"):
            original_file_path = os.path.join(original_path, file)
            base_filename = os.path.splitext(file)[0]
            noisy_file = f"{base_filename}_augmented.npy"
            noisy_file_path = os.path.join(noisy_path, noisy_file)

            if not os.path.exists(noisy_file_path):
                print(f"Warning: Noisy file not found for {file}")
                continue

            original = np.load(original_file_path)
            noisy = np.load(noisy_file_path)
            snr = compute_snr(original, noisy)

            snr_records.append({
                "filename": file,
                "class": label,
                "view": view,
                "snr": snr
            })

df = pd.DataFrame(snr_records)
df.to_csv(output_csv, index=False)
print(f"\nSNR results saved to {output_csv}")
