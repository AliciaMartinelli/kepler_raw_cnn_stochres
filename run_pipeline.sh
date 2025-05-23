#!/bin/bash

sigmas=(0.00001 0.0001 0.001 0.003 0.005 0.007 0.01 0.015 0.02 0.03 0.1 0.3 1)

for sigma in "${sigmas[@]}"
do
    echo "=========================================="
    echo "Running CNN pipeline for sigma = $sigma"
    echo "=========================================="

    SECONDS=0

    echo "[1/3] data augmentation"
    python data_aug.py $sigma

    echo "[2/3] train CNN"
    python train_CNN.py $sigma

    echo "[3/3] snr calculation"
    python calculate_snr.py $sigma

    duration=$SECONDS
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    echo "Pipeline execution completed for sigma = $sigma"
    echo "Elapsed time: ${minutes}m ${seconds}s"
    echo ""
done

echo "All sigma values processed successfully."