import os
import sys
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

if len(sys.argv) < 2:
    print("Usage: python train_CNN.py <sigma>")
    sys.exit(1)

try:
    sigma = float(sys.argv[1])
except ValueError:
    print("Error: sigma must be a float.")
    sys.exit(1)

DATASET_PATH = os.path.join("aug_dataset", f"sigma_{sigma:.5f}")
RESULT_PATH = os.path.join("results", "cnn", f"results_cnn_sigma_{sigma:.5f}.csv")
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

def load_data():
    X_global, X_local, y = [], [], []
    for label in ['0', '1']:
        global_folder = os.path.join(DATASET_PATH, label, 'global')
        local_folder = os.path.join(DATASET_PATH, label, 'local')
        
        global_files = sorted([f for f in os.listdir(global_folder) if f.endswith('.npy')])

        for gfile in global_files:
            gpath = os.path.join(global_folder, gfile)
            lfile = gfile.replace('global', 'local')
            lpath = os.path.join(local_folder, lfile)

            if not os.path.exists(lpath):
                print(f"Not found: {gfile}")
                continue

            X_global.append(np.load(gpath))
            X_local.append(np.load(lpath))
            y.append(int(label))

    X_global = np.array(X_global)[..., np.newaxis]
    X_local = np.array(X_local)[..., np.newaxis]
    return (X_global, X_local), np.array(y)


def build_model(global_shape=(2001, 1), local_shape=(201, 1)):
    global_input = Input(shape=global_shape, name='global_input')
    x = Conv1D(16, kernel_size=5, activation='relu')(global_input)
    x = Conv1D(16, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(32, kernel_size=5, activation='relu')(x)
    x = Conv1D(32, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(64, kernel_size=5, activation='relu')(x)
    x = Conv1D(64, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=5, activation='relu')(x)
    x = Conv1D(128, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(256, kernel_size=5, activation='relu')(x)
    x = Conv1D(256, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)

    local_input = Input(shape=local_shape, name='local_input')
    y = Conv1D(16, kernel_size=5, activation='relu')(local_input)
    y = Conv1D(16, kernel_size=5, activation='relu')(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Conv1D(32, kernel_size=5, activation='relu')(y)
    y = Conv1D(32, kernel_size=5, activation='relu')(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Flatten()(y)

    combined = Concatenate()([x, y])
    z = Dense(512, activation='relu')(combined)
    z = Dense(512, activation='relu')(z)
    z = Dense(512, activation='relu')(z)
    z = Dense(512, activation='relu')(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[global_input, local_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    return model

results = []
print(f"Training CNN for sigma = {sigma:.5f} ...")

for run in range(50):
    (X_global, X_local), y = load_data()
    Xg_temp, Xg_test, Xl_temp, Xl_test, y_temp, y_test = train_test_split(
        X_global, X_local, y, test_size=0.1, stratify=y, random_state=SEED + run)
    Xg_train, Xg_val, Xl_train, Xl_val, y_train, y_val = train_test_split(
        Xg_temp, Xl_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=SEED + run)

    model = build_model()

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    model.fit({'global_input': Xg_train, 'local_input': Xl_train}, y_train,
              validation_data=({'global_input': Xg_val, 'local_input': Xl_val}, y_val),
              epochs=100, batch_size=32, callbacks=[es], verbose=0)

    y_val_proba = model.predict({'global_input': Xg_val, 'local_input': Xl_val}).flatten()
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    optimal_idx = np.argmax(f1_vals)
    optimal_threshold = thresholds[optimal_idx] if thresholds.size > 0 else 0.5

    y_test_proba = model.predict({'global_input': Xg_test, 'local_input': Xl_test}).flatten()
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

    auc = roc_auc_score(y_test, y_test_proba)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    p_misclass = cm[1, 0] / cm[1].sum() * 100 if cm[1].sum() != 0 else 0
    n_misclass = cm[0, 1] / cm[0].sum() * 100 if cm[0].sum() != 0 else 0

    results.append({
        "run": run + 1,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "p_misclass": p_misclass,
        "n_misclass": n_misclass
    })

    print(f"Run {run+1}/50 - AUC: {auc:.4f}, F1: {f1:.4f}")

df = pd.DataFrame(results)
df.to_csv(RESULT_PATH, index=False)
print(f"\nAll results saved to {RESULT_PATH}")
