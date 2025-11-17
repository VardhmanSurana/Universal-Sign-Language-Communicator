import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Constants
ROOT_DATASET_DIR = "data"
DATASET_SUB_DIR = "dataset"
MODEL_SAVE_PATH = "sign_language_model.keras"
LABEL_ENCODER_SAVE_PATH = "label_encoder.npy"

def get_dataset_folders(root_dir, sub_dir):
    dataset_path = os.path.join(root_dir, sub_dir)
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset subdirectory '{dataset_path}' not found.")
        return []
    return [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

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

def preprocess_data(X, y, existing_label_encoder=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    label_encoder = existing_label_encoder or LabelEncoder()
    y_encoded = label_encoder.fit_transform(y) if existing_label_encoder is None else label_encoder.transform(y)
    return X_scaled, y_encoded, label_encoder

def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    dataset_base_path = os.path.join(ROOT_DATASET_DIR, DATASET_SUB_DIR)
    dataset_folders = get_dataset_folders(ROOT_DATASET_DIR, DATASET_SUB_DIR)
    if not dataset_folders:
        print(f"No dataset folders found inside '{dataset_base_path}'.")
        exit()

    model = None
    all_classes = set()

    for folder_name in dataset_folders:
        dataset_path = os.path.join(dataset_base_path, folder_name)
        X_temp, y_temp = load_data(dataset_path)
        all_classes.update(y_temp)

    label_encoder = LabelEncoder()
    label_encoder.fit(list(all_classes))

    for folder_name in dataset_folders:
        dataset_path = os.path.join(dataset_base_path, folder_name)
        print(f"\n--- Training on dataset: {folder_name} ---")
        X, y = load_data(dataset_path)
        if X.shape[0] == 0:
            print(f"Warning: No data in '{dataset_path}'. Skipping.")
            continue

        X_processed, y_encoded, _ = preprocess_data(X, y, label_encoder)
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)
        input_shape = X_train.shape[1]
        num_classes = len(label_encoder.classes_)

        if os.path.exists(MODEL_SAVE_PATH) and model is None:
            model = load_model(MODEL_SAVE_PATH)
            if model.input_shape[1] != input_shape or model.output_shape[1] != num_classes:
                print("Model shape mismatch. Reinitializing.")
                model = create_model(input_shape, num_classes)
        elif model is None:
            model = create_model(input_shape, num_classes)

        model.fit(X_train, y_train, epochs=120, batch_size=1024, validation_data=(X_test, y_test), verbose=1)
        model.save(MODEL_SAVE_PATH)
        print(f"Model saved to '{MODEL_SAVE_PATH}'.")

        if not os.path.exists(LABEL_ENCODER_SAVE_PATH) or folder_name == dataset_folders[-1]:
            np.save(LABEL_ENCODER_SAVE_PATH, label_encoder.classes_)
            print(f"Label encoder saved to '{LABEL_ENCODER_SAVE_PATH}'.")

    print("\n--- Training completed on all datasets ---")
