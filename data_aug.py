import os
import sys
import numpy as np
import pathlib
from tqdm import tqdm

def add_gaussian_noise(signal, sigma):
    noise = sigma * np.random.normal(0, 1, size=signal.shape)
    return signal + noise

def load_npy_files(base_path):
    npy_files = sorted(pathlib.Path(base_path).rglob("*.npy"))
    classes = [str(f.parts[-3]) for f in npy_files]
    views = [str(f.parts[-2]) for f in npy_files]
    return npy_files, classes, views

def augment_lightcurves(npy_files, classes, views, sigma, output_root="aug_dataset"):
    save_path = os.path.join(output_root, f"sigma_{sigma:.5f}")
    for i, npy_file in tqdm(enumerate(npy_files), total=len(npy_files)):
        signal = np.load(npy_file)
        noisy_signal = add_gaussian_noise(signal, sigma)

        label = classes[i]
        view = views[i]
        filename = npy_file.stem + "_augmented.npy"

        save_dir = os.path.join(save_path, label, view)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, filename), noisy_signal)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_aug_cnn.py <sigma>")
        sys.exit(1)

    try:
        sigma = float(sys.argv[1])
    except ValueError:
        print("Error: sigma must be a float.")
        sys.exit(1)

    np.random.seed(42)
    input_path = "./dataset"
    output_path = "./aug_dataset"
    npy_files, classes, views = load_npy_files(input_path)
    augment_lightcurves(npy_files, classes, views, sigma=sigma, output_root=output_path)