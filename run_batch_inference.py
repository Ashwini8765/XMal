import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# ===============================
# Paths (EDIT THESE)
# ===============================
artifact_dir = "trained_artifacts"
CSV_DIR = "/home/user/Documents/journal_1/new_synthetic_CSVs"
IMAGE_DIR = "/home/user/Documents/journal_1/cicAndMal2020_heatmap_images"
OUTPUT_CSV = "batch_predictions.csv"
OUTPUT_FIG = "prediction_distribution.png"
PATTERN_TSNE_FIG = "pattern_clusters_tsne.png"
PATTERN_CLUSTER_CSV = "pattern_clusters.csv"

# ===============================
# Load Artifacts
# ===============================
print("[Load] Loading trained model and preprocessing artifacts...")
clf = joblib.load(os.path.join(artifact_dir, "fused_random_forest.pkl"))
le = joblib.load(os.path.join(artifact_dir, "label_encoder.pkl"))
scaler = joblib.load(os.path.join(artifact_dir, "scaler.pkl"))
csv_feats = np.load(os.path.join(artifact_dir, "selected_features_csv.npy"), allow_pickle=True)
fused_feats = np.load(os.path.join(artifact_dir, "selected_features_fused.npy"), allow_pickle=True)
print(f"[Load] Model, encoder, and scaler loaded. CSV features: {len(csv_feats)}, Fused: {len(fused_feats)}")


# ===============================
# Extract ResNet Embeddings
# ===============================
def extract_resnet_embeddings(image_dir, batch_size=32, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Image] Extracting embeddings from {image_dir} using ResNet50 ({device.upper()})")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_files = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_files.append(os.path.join(root, f))

    embeddings = {}
    if not image_files:
        print("[Warn] No images found — generating zero embeddings.")
        return embeddings

    with torch.no_grad():
        for i in tqdm(range(0, len(image_files), batch_size), desc="[Image] Batching"):
            batch_files = image_files[i:i + batch_size]
            batch_imgs, valid_files = [], []
            for path in batch_files:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_imgs.append(transform(img))
                    valid_files.append(os.path.basename(path))
                except Exception as e:
                    print(f"[Warn] Skipping {path}: {e}")
            if not batch_imgs:
                continue

            batch_tensor = torch.stack(batch_imgs).to(device)
            feats = model(batch_tensor).squeeze(-1).squeeze(-1).cpu().numpy()
            for fname, feat in zip(valid_files, feats):
                embeddings[fname] = feat

    print(f"[Image] Extracted {len(embeddings)} image embeddings.")
    return embeddings


# ===============================
# CSV Preprocessing
# ===============================
def preprocess_csv(df, scaler, csv_feats):
    df = df.drop(columns=['FileName', 'SHA256', 'Hash', 'AppName'], errors='ignore')

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    X_proc = df[num_cols].copy()
    if cat_cols:
        for col in cat_cols:
            X_proc[col] = pd.Categorical(df[col]).codes

    for c in csv_feats:
        if c not in X_proc.columns:
            X_proc[c] = 0
    X_proc = X_proc[[c for c in csv_feats]]

    print(f"[Preprocess] Aligned CSV columns: {X_proc.shape[1]} / Expected: {len(csv_feats)}")

    from sklearn.preprocessing import MinMaxScaler
    try:
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != X_proc.shape[1]:
            print(f"[Warn] Scaler mismatch ({scaler.n_features_in_} vs {X_proc.shape[1]}). Re-fitting temporary scaler.")
            tmp_scaler = MinMaxScaler()
            X_scaled = tmp_scaler.fit_transform(np.nan_to_num(X_proc))
        else:
            X_scaled = scaler.transform(np.nan_to_num(X_proc))
    except Exception as e:
        print(f"[Warn] Scaling failed: {e}. Using unscaled features.")
        X_scaled = np.nan_to_num(X_proc)

    return X_scaled, X_proc.columns.tolist()


# ===============================
# Visualization Function
# ===============================
def plot_prediction_distribution(df, column="Predicted_Category", save_path=OUTPUT_FIG):
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid", font_scale=1.1)
    sns.countplot(data=df, y=column, order=df[column].value_counts().index, palette="viridis")
    plt.title("Prediction Distribution per Malware Family", fontsize=14)
    plt.xlabel("Count")
    plt.ylabel("Predicted Malware Family")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved prediction distribution chart to {save_path}")


# ===============================
# Pattern Recognition Module (FIXED)
# ===============================
def analyze_patterns(X_fused, data):
    print("\n[Pattern] Starting t-SNE pattern visualization and clustering...")

    # Avoid fragmented frame warning
    data = data.copy()

    # === t-SNE visualization ===
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    X_tsne = tsne.fit_transform(X_fused)
    data["tsne_x"] = X_tsne[:, 0]
    data["tsne_y"] = X_tsne[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x="tsne_x",
        y="tsne_y",
        hue="Predicted_Category",
        palette="tab10",
        s=15,
        alpha=0.8
    )
    plt.title("t-SNE Visualization of Malware Pattern Clusters", fontsize=14)
    plt.tight_layout()
    plt.savefig(PATTERN_TSNE_FIG, dpi=300)
    plt.close()
    print(f"[Pattern] Saved t-SNE cluster visualization to {PATTERN_TSNE_FIG}")

    # === K-Means clustering ===
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data["Pattern_Cluster"] = kmeans.fit_predict(X_fused)

    cluster_summary = (
        data.groupby("Pattern_Cluster")["Predicted_Category"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    data["Cluster_Label"] = data["Pattern_Cluster"].map(cluster_summary)

    data.to_csv(PATTERN_CLUSTER_CSV, index=False)
    print(f"[Pattern] Saved clustering results to {PATTERN_CLUSTER_CSV} ({n_clusters} clusters).")


# ===============================
# Main Inference Routine
# ===============================
def run_batch_inference():
    print("[Data] Loading new CSV files...")
    dfs = [pd.read_csv(os.path.join(CSV_DIR, f)) for f in os.listdir(CSV_DIR) if f.endswith(".csv")]
    if not dfs:
        raise ValueError("No CSV files found in the folder.")
    data = pd.concat(dfs, ignore_index=True)
    print(f"[Data] Loaded {data.shape[0]} samples, {data.shape[1]} columns.")

    image_embeddings = extract_resnet_embeddings(IMAGE_DIR)
    X_csv_scaled, csv_columns = preprocess_csv(data, scaler, csv_feats)
    print(f"[Data] CSV feature shape after preprocessing: {X_csv_scaled.shape}")

    X_image = []
    for i in range(len(data)):
        fname = f"{i}.png"
        X_image.append(image_embeddings.get(fname, np.zeros(2048)))
    X_image = np.array(X_image)

    print(f"[Fusion] Fusing CSV + Image embeddings: {X_image.shape[0]} samples")

    # Combine into DataFrame to preserve names
    csv_df = pd.DataFrame(X_csv_scaled, columns=csv_columns)
    image_df = pd.DataFrame(X_image, columns=[f"img_feat_{i}" for i in range(X_image.shape[1])])
    fused_df = pd.concat([csv_df, image_df], axis=1)

    # === FIX: select feature subset by name ===
    print("[Predict] Running inference...")
    available_cols = set(fused_df.columns)
    selected_cols = [f for f in fused_feats if f in available_cols]

    if not selected_cols:
        raise ValueError("[Error] None of the trained fused feature names matched current features!")

    print(f"[Fix] Selecting {len(selected_cols)} trained fused features out of {fused_df.shape[1]} total.")
    X_fused = fused_df[selected_cols].to_numpy()

    y_pred = clf.predict(X_fused)
    y_labels = le.inverse_transform(y_pred)
    data["Predicted_Category"] = y_labels
    data.to_csv(OUTPUT_CSV, index=False)
    print(f"[Done] Predictions saved to {OUTPUT_CSV}")

    plot_prediction_distribution(data)
    analyze_patterns(X_fused, data)


# ===============================
# Run
# ===============================
if __name__ == "__main__":
    run_batch_inference()

