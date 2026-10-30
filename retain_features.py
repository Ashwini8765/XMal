import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Conv2D, GlobalAveragePooling1D, GlobalAveragePooling2D, Dense

# ------------------
# 0. Enable GPU memory growth
# ------------------
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled.")
        except RuntimeError as e:
            print("❌ Error setting GPU memory growth:", e)
    else:
        print("⚠️ No GPU found. Using CPU.")

# ------------------
# 1. Load and Merge All CSV Files in Folder
# ------------------
def load_csv_data(folder_path):
    all_data = []
    class_names = []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            class_name = os.path.splitext(file)[0]
            class_names.append(class_name)

            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            df['Label'] = class_name  # Assign class label
            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Keep only numeric columns (drop strings like hashes)
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_names = [col for col in numeric_cols if col != 'Label']

    X = combined_df[feature_names].values.astype(np.float32)
    y = combined_df['Label'].values

    return X, y, feature_names

# ------------------
# 2. Convert to Image Format (1xN)
# ------------------
def convert_to_image_format(X):
    return X.reshape((X.shape[0], 1, X.shape[1], 1))

# ------------------
# 3. CNN Models
# ------------------
def build_csv_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu', padding='same', name='conv1d_1')(inputs)
    x = Conv1D(64, 3, activation='relu', padding='same', name='conv1d_2')(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(14, activation='softmax')(x)
    return Model(inputs, outputs)

def build_image_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (1, 3), activation='relu', padding='same', name='conv2d_1')(inputs)
    x = Conv2D(64, (1, 3), activation='relu', padding='same', name='conv2d_2')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(14, activation='softmax')(x)
    return Model(inputs, outputs)

# ------------------
# 4. Grad-CAM Functions
# ------------------
def grad_cam_1d(model, sample, layer_name='conv1d_1'):
    grad_model = Model([model.input], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(sample)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=0)
    heatmap = tf.reduce_mean(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    return heatmap / np.max(heatmap + 1e-8)

def grad_cam_2d(model, sample, layer_name='conv2d_1'):
    grad_model = Model([model.input], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(sample)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    weighted_output = tf.multiply(conv_outputs[0], pooled_grads)
    heatmap = tf.reduce_mean(weighted_output, axis=-1).numpy().flatten()
    heatmap = np.maximum(heatmap, 0)
    return heatmap / np.max(heatmap + 1e-8)

# ------------------
# 5. Feature Importance Plotting
# ------------------
def compute_classwise_gradcam_average(model, X_test, y_test, gradcam_fn, class_label):
    class_indices = np.where(y_test == class_label)[0]
    heatmaps = []
    for idx in class_indices:
        sample = X_test[idx:idx+1]
        heatmap = gradcam_fn(model, sample)
        heatmaps.append(heatmap)
    return np.mean(heatmaps, axis=0)

def plot_top_features(heatmap, feature_names, title, save_path):
    top_idx = np.argsort(heatmap)[-15:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_scores = heatmap[top_idx]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_scores, y=top_features, palette='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ------------------
# 6. Main Flow
# ------------------
def main():
    setup_gpu()
    csv_dir = r"/home/user/Documents/journal_1/cicAndMal2020_CSVs"  # change to your path

    X, y, feature_names = load_csv_data(csv_dir)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    num_classes = len(class_names)

    skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
    acc_csv, acc_img = [], []
    classwise_csv_heatmaps = [[] for _ in range(num_classes)]
    classwise_img_heatmaps = [[] for _ in range(num_classes)]

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_encoded), 1):
        print(f"\n🔁 Fold {fold}")

        # 1D CSV model
        X_csv = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        X_train_c, X_test_c = X_csv[train_idx], X_csv[test_idx]
        y_train_c, y_test_c = y_encoded[train_idx], y_encoded[test_idx]
        csv_model = build_csv_model((X_scaled.shape[1], 1))
        csv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy, recall, precision'])
        csv_model.fit(X_train_c, y_train_c, epochs=10, batch_size=64, verbose=0)
        y_pred_c = np.argmax(csv_model.predict(X_test_c), axis=1)
        acc_csv.append(accuracy_score(y_test_c, y_pred_c))

        for class_id in range(num_classes):
            classwise_csv_heatmaps[class_id].append(
                compute_classwise_gradcam_average(csv_model, X_test_c, y_test_c, grad_cam_1d, class_label=class_id)
            )

        # 2D Image model
        X_img = convert_to_image_format(X_scaled)
        X_train_i, X_test_i = X_img[train_idx], X_img[test_idx]
        y_train_i, y_test_i = y_encoded[train_idx], y_encoded[test_idx]
        img_model = build_image_model((1, X_scaled.shape[1], 1))
        img_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy, recall, precision'])
        img_model.fit(X_train_i, y_train_i, epochs=10, batch_size=64, verbose=0)
        y_pred_i = np.argmax(img_model.predict(X_test_i), axis=1)
        acc_img.append(accuracy_score(y_test_i, y_pred_i))

        for class_id in range(num_classes):
            classwise_img_heatmaps[class_id].append(
                compute_classwise_gradcam_average(img_model, X_test_i, y_test_i, grad_cam_2d, class_label=class_id)
            )

    # Save plots
    os.makedirs("output_heatmaps", exist_ok=True)
    for class_id in range(num_classes):
        class_name = class_names[class_id]
        avg_csv = np.mean(classwise_csv_heatmaps[class_id], axis=0)
        avg_img = np.mean(classwise_img_heatmaps[class_id], axis=0)

        plot_top_features(avg_csv, feature_names, f"CSV - {class_name} (Avg Acc: {np.mean(acc_csv):.2f})", f"output_heatmaps/csv_{class_name}.png")
        plot_top_features(avg_img, feature_names, f"Image - {class_name} (Avg Acc: {np.mean(acc_img):.2f})", f"output_heatmaps/img_{class_name}.png")

    print("\n✅ Final Average Accuracy:")
    print("CSV CNN:", np.mean(acc_csv))
    print("Image CNN:", np.mean(acc_img))

if __name__ == '__main__':
    main()

