import os
import pathlib
import numpy as np

def save_npy(array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)

def process_lightcurves(raw_folder, output_folder):
    npy_files = sorted(pathlib.Path(raw_folder).rglob("*.npy"))

    for npy_file in npy_files:
        data = np.load(npy_file, allow_pickle=True).item()
        lightcurve = data["lightcurve"]
        label = data["label"]

        global_view = lightcurve[:2001]
        local_view = lightcurve[2001:]

        file_stem = npy_file.stem

        global_path = os.path.join(output_folder, str(label), "global", f"{file_stem}_global.npy")
        local_path = os.path.join(output_folder, str(label), "local", f"{file_stem}_local.npy")

        save_npy(global_view, global_path)
        save_npy(local_view, local_path)

        print(f"Saved global: {global_path}")
        print(f"Saved local: {local_path}")

raw_folder = './raw'
output_folder = './dataset'

process_lightcurves(raw_folder, output_folder)
