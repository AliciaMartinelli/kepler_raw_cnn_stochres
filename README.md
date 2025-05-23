# Kepler - Stochastic Resonance - raw light curves + CNN Classification

This repository contains an experiment in which Gaussian noise is injected into the Kepler light curves directly. The impact of this noise on classification performance is evaluated using a CNN classifier.

This experiment is part of the Bachelor's thesis **"Machine Learning for Exoplanet Detection: Investigating Feature Engineering Approaches and Stochastic Resonance Effects"** by Alicia Martinelli (2025).

## Folder Structure

```
kepler_raw_cnn_stochres/
├── convert.py                 # This saves the global and local view separatly into the dataset folder
├── data_aug.py                # Adds Gaussian noise to the light curves from the dataset folder and saves them into the aug_dataset folder
├── calculate_snr.py           # Calculates the SNR
├── train_CNN.py               # CNN training
├── run_pipeline.sh            # Run the pipeline to add noise and train the CNN for each noise intensity parameter sigma
└── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## Preprocessed Kepler dataset
The preprocessed Kepler dataset used in this project is based on the public release from Shallue & Vanderburg (2018) and is available via the AstroNet GitHub repository (Google Drive) [https://drive.google.com/drive/folders/1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE](https://drive.google.com/drive/folders/1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE)

Download the TFRecords from the Google Drive, convert them into .npy files and save them in the raw folder.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/AliciaMartinelli/kepler_raw_cnn_stochres.git
    cd kepler_raw_cnn_stochres
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    You may need to install `scikit-learn`, `tsfresh`, `matplotlib`, `numpy`, and `tensorflow` (and more).

## Usage

1. Prepare the light curves:
```bash
python convert.py
```
Split the light curves from the raw folder into the global and local view and save them into the dataset folder

2. Run the pipeline:
```bash
./run_pipeline.sh
```
Start the pipeline to add noise to the light curves from the dataset folder and train the CNN models per sigma. The results will be saved into the results folder.

3. Plot the AUC vs noise intensity parameter sigma:
```bash
python visualize_results.py
```
This will visualize the results in a plot with AUC vs. noise intensity parameter sigma

## Thesis Context

This repository corresponds to the experiment described in:
- **Section 6.3**: Light curve noise injection: Evaluation using CNN

**Author**: Alicia Martinelli  
**Email**: alicia.martinelli@stud.unibas.ch  
**Year**: 2025