import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

MODEL_SAVE_PATH = "sign_language_model.keras"
LABEL_ENCODER_SAVE_PATH = "label_encoder.npy"
DATASET_PATH = "data/dataset"  # Update this to the specific folder you want to evaluate

def load_data(dataset_dir):
    all_data, all_labels = [], []
    for sign_name in os.listdir(dataset_dir):
        sign_path = os.path.join(dataset_dir, sign_name)
        if not os.path.isdir(sign_path): continue
        for file_name in os.listdir(sign_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(sign_path, file_name)
                df = pd.read_csv(file_path)
                df.drop(columns=["frame"], inplace=True)
                all_data.extend(df.values)
                all_labels.extend([sign_name] * len(df))
    return np.array(all_data, dtype=np.float32), np.array(all_labels)

def print_confusion_matrix(y_true, y_pred, all_possible_labels, report=True):
    unique_y_true = np.unique(y_true)
    labels_present = [all_possible_labels[i] for i in unique_y_true]
    cmx_data = confusion_matrix(y_true, y_pred, labels=unique_y_true)
    df_cmx = pd.DataFrame(cmx_data, index=labels_present, columns=labels_present)
    sns.heatmap(df_cmx, annot=True, fmt='g')
    plt.title("Confusion Matrix")
    plt.show()

    if report:
        print("Classification Report:")
        print(classification_report(y_true, y_pred, labels=unique_y_true, target_names=labels_present))

    tpr_tnr_per_class(y_true, y_pred, unique_y_true, labels_present)

def tpr_tnr_per_class(y_true, y_pred, encoded_labels, decoded_labels):
    cm = confusion_matrix(y_true, y_pred, labels=encoded_labels)
    for i, label_encoded in enumerate(encoded_labels):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"Class: {decoded_labels[i]}")
        print(f"  TPR: {tpr:.4f}")
        print(f"  TNR: {tnr:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(LABEL_ENCODER_SAVE_PATH):
        print("Model or Label Encoder not found.")
        exit()

    model = load_model(MODEL_SAVE_PATH)
    label_classes = np.load(LABEL_ENCODER_SAVE_PATH, allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_classes

    X, y = load_data(DATASET_PATH)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_encoded = label_encoder.transform(y)

    predictions = model.predict(X_scaled)
    y_pred = np.argmax(predictions, axis=1)

    print_confusion_matrix(y_encoded, y_pred, label_encoder.classes_, report=True)
