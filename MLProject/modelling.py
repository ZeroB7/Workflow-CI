"""
modelling.py
============
Training model clustering menggunakan MLflow Project.
Kelas Membangun Sistem Machine Learning
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
import os

# ── Argument Parser ───────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_clusters",   type=int, default=5)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# ── Konfigurasi ───────────────────────────────────────────────────────
DATASET_PATH = "mall_customers_preprocessing.csv"
OPTIMAL_K    = args.n_clusters
RANDOM_STATE = args.random_state


def load_data(path: str) -> pd.DataFrame:
    """Load dataset preprocessing."""
    df = pd.read_csv(path)
    print(f"[load_data] Dataset dimuat: {df.shape}")
    return df


def train_model(df: pd.DataFrame):
    """Training K-Means dengan MLflow logging."""

    # Log parameter
    mlflow.log_param("n_clusters",   OPTIMAL_K)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("dataset",      DATASET_PATH)

    # Training model
    model = KMeans(n_clusters=OPTIMAL_K, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(df)

    # Hitung metrik
    sil_score = silhouette_score(df, labels)
    inertia   = model.inertia_

    # Log metrik
    mlflow.log_metric("silhouette_score", sil_score)
    mlflow.log_metric("inertia",          inertia)

    print(f"[train] n_clusters     : {OPTIMAL_K}")
    print(f"[train] Inertia        : {inertia:.2f}")
    print(f"[train] Silhouette     : {sil_score:.4f}")

    # Buat visualisasi
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_result = pca.fit_transform(df)

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, OPTIMAL_K))
    for i in range(OPTIMAL_K):
        mask = labels == i
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1],
                    color=colors[i], label=f"Cluster {i}", s=60, alpha=0.8)
    plt.title(f"K-Means Clustering (k={OPTIMAL_K})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()

    os.makedirs("artifacts", exist_ok=True)
    plot_path = "artifacts/clustering_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Log artefak
    mlflow.log_artifact(plot_path)
    mlflow.sklearn.log_model(model, "kmeans_model")

    print("[train] Run selesai!")

    return model, labels


if __name__ == "__main__":
    df = load_data(DATASET_PATH)
    train_model(df)
    print("Training selesai!")