#!/usr/bin/env python3
import os
import gc
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.inspection import permutation_importance

import torch
import joblib

# ===============================
# Hybrid Feature Selection (Memory-Optimized)
# ===============================
def hybrid_feature_selection(X, y, feature_names,
                             initial_filter="mi",
                             initial_k=100,
                             final_k=40,
                             n_splits=2,
                             n_runs=1,
                             stability_threshold=0.5,
                             random_state=42):
    """
    Memory-optimized hybrid feature selection:
      1. Univariate filter (MI or chi2)
      2. Lightweight RandomForest permutation importance
      3. Stability selection
    """
    X = np.nan_to_num(X)

    # --- Initial filter ---
    if initial_filter == "mi":
        selector = SelectKBest(score_func=mutual_info_classif,
                               k=min(initial_k, X.shape[1]))
        X_reduced = selector.fit_transform(X, y)
        reduced_names = [feature_names[i] for i in selector.get_support(indices=True)]
    elif initial_filter == "chi2":
        scaler_local = MinMaxScaler()
        X_scaled = scaler_local.fit_transform(X)
        selector = SelectKBest(score_func=chi2,
                               k=min(initial_k, X.shape[1]))
        X_reduced = selector.fit_transform(X_scaled, y)
        reduced_names = [feature_names[i] for i in selector.get_support(indices=True)]
    else:
        raise ValueError("initial_filter must be 'mi' or 'chi2'")

    print(f"[Hybrid] Initial filter kept {len(reduced_names)} features")

    # --- RandomForest refinement ---
    feature_counts = Counter()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rf_params = dict(
        n_estimators=50,
        max_depth=10,
        n_jobs=1,
        random_state=random_state
    )

    for run in range(n_runs):
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_reduced, y)):
            # Garbage collection between folds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            X_train, X_val = X_reduced[train_idx], X_reduced[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            rf = RandomForestClassifier(**rf_params)
            rf.fit(X_train, y_train)

            perm = permutation_importance(
                rf, X_val, y_val,
                n_repeats=2,
                random_state=random_state + run + fold,
                n_jobs=1
            )

            importances = perm.importances_mean
            top_idx = np.argsort(importances)[-final_k:]
            for i in top_idx:
                feature_counts[reduced_names[i]] += 1

    # --- Stability filtering ---
    min_appear = int(np.ceil(n_runs * stability_threshold))
    stable_feats = [f for f, c in feature_counts.items() if c >= min_appear]
    if len(stable_feats) < final_k:
        stable_feats = [f for f, _ in feature_counts.most_common(final_k)]

    print(f"[Hybrid] Selected {len(stable_feats)} stable features")
    sel_idx = [reduced_names.index(f) for f in stable_feats]
    return X_reduced[:, sel_idx], stable_feats


# ===============================
# Data Loading & Safe Preprocessing
# ===============================
def load_csv_data(csv_dir, label_col="Category"):
    dfs = []
    for f in os.listdir(csv_dir):
        if f.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(csv_dir, f)))
    if not dfs:
        raise ValueError("No CSV files found")
    data = pd.concat(dfs, ignore_index=True)
    if label_col not in data.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV")
    return data


def preprocess_csv(df, label_col="Category"):
    """Safe preprocessing that avoids huge one-hot expansions."""
    y = df[label_col].values
    X = df.drop(columns=[label_col])

    # Drop ID-like or unique categorical columns to prevent blowup (adjust names as needed)
    X = X.drop(columns=['FileName', 'SHA256', 'Hash', 'AppName'], errors='ignore')

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"[Preprocess] Numeric={len(num_cols)}, Categorical={len(cat_cols)}")

    X_proc = X[num_cols].copy()

    if cat_cols:
        safe_cat_cols = []
        for col in cat_cols:
            n_unique = X[col].nunique()
            if n_unique < 20:
                safe_cat_cols.append(col)
            else:
                print(f"[Skip] '{col}' has {n_unique} unique values — skipped one-hot encoding.")
                # fallback to integer codes (no wide expansion)
                X[col] = pd.Categorical(X[col]).codes
                X_proc[col] = X[col]

        if safe_cat_cols:
            X_cats = pd.get_dummies(X[safe_cat_cols], drop_first=True)
            X_proc = pd.concat([X_proc, X_cats], axis=1)

    feature_names = X_proc.columns.tolist()
    print(f"[Preprocess] Final feature count: {len(feature_names)}")
    return X_proc.values, y, feature_names


# ===============================
# Main Multimodal Pipeline (Resume)
# ===============================
if __name__ == "__main__":
    CSV_DIR = "/home/user/Documents/journal_1/cicAndMal2020_CSVs"
    EMB_PATH = "/home/user/Documents/journal_1/resnet_image_embeddings.npy"

    # --- Load image embeddings ---
    print("[Resume] Loading precomputed image embeddings...")
    image_embeddings_dict = np.load(EMB_PATH, allow_pickle=True).item()
    print(f"[Resume] Loaded {len(image_embeddings_dict)} embeddings.")

    # --- Load CSV data ---
    csv_data = load_csv_data(CSV_DIR, label_col="Category")
    print(f"[Data] Loaded CSV shape: {csv_data.shape}")

    # Optional subset mode for debugging (uncomment to use)
    # csv_data = csv_data.sample(n=20000, random_state=42)

    # --- Preprocess ---
    X_numeric, y, feature_names = preprocess_csv(csv_data, label_col="Category")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # --- Stage 1: CSV Feature Selection ---
    print("[Stage 1] Running memory-optimized CSV feature selection...")
    X_csv_sel, csv_selected_features = hybrid_feature_selection(
        X_scaled, y_encoded, feature_names,
        initial_filter="mi",
        initial_k=100,
        final_k=40,
        n_splits=2,
        n_runs=1,
        stability_threshold=0.5,
        random_state=42
    )

    # --- Align image embeddings ---
    X_image = []
    for i in range(len(csv_data)):
        fname = f"{i}.png"
        X_image.append(image_embeddings_dict.get(fname, np.zeros(2048)))
    X_image = np.array(X_image)

    print(f"[Fusion] Image array shape: {X_image.shape}")
    print(f"[Fusion] CSV-selected features shape: {X_csv_sel.shape}")

    # --- Fusion ---
    X_fused = np.hstack([X_csv_sel, X_image])
    fused_feature_names = csv_selected_features + [f"img_{i}" for i in range(X_image.shape[1])]

    # --- Stage 2: Fused Feature Selection ---
    print("[Stage 2] Running memory-optimized fused feature selection...")
    X_fused_sel, fused_selected_features = hybrid_feature_selection(
        X_fused, y_encoded, fused_feature_names,
        initial_filter="mi",
        initial_k=200,
        final_k=40,
        n_splits=2,
        n_runs=1,
        stability_threshold=0.5,
        random_state=42
    )

    # --- Train/Test Split ---
    print("[Train] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_fused_sel, y_encoded, test_size=0.2,
        random_state=42, stratify=y_encoded
    )

    # --- Train RandomForest ---
    print("[Train] Training RandomForest classifier (optimized)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=1
    )
    clf.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = clf.predict(X_test)
    print("\n[Results] Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ===============================
    # Save Trained Model and Artifacts
    # ===============================
    output_dir = "trained_artifacts"
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(clf, os.path.join(output_dir, "fused_random_forest.pkl"))
    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    np.save(os.path.join(output_dir, "selected_features_csv.npy"), np.array(csv_selected_features))
    np.save(os.path.join(output_dir, "selected_features_fused.npy"), np.array(fused_selected_features))

    print(f"[Saved] Model and artifacts stored in '{output_dir}/'")

    # ===============================
    # Quick Verification (loads saved artifacts and prints a short summary)
    # ===============================
    print("[Verify] Loading saved artifacts to verify...")
    clf_loaded = joblib.load(os.path.join(output_dir, "fused_random_forest.pkl"))
    le_loaded = joblib.load(os.path.join(output_dir, "label_encoder.pkl"))
    scaler_loaded = joblib.load(os.path.join(output_dir, "scaler.pkl"))
    csv_feats_loaded = np.load(os.path.join(output_dir, "selected_features_csv.npy"), allow_pickle=True)
    fused_feats_loaded = np.load(os.path.join(output_dir, "selected_features_fused.npy"), allow_pickle=True)

    print(f"[Verify] Classifier loaded. #trees: {len(clf_loaded.estimators_)}")
    print(f"[Verify] Label encoder classes: {le_loaded.classes_}")
    print(f"[Verify] CSV-selected features count: {len(csv_feats_loaded)}")
    print(f"[Verify] Fused-selected features count: {len(fused_feats_loaded)}")

    print("\n✅ [Done] Training completed and artifacts saved/verified.")

