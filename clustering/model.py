import numpy as np
import joblib
import os
import json
from sklearn.mixture import GaussianMixture

# Paths
EMB_PATH = "./embeddings/embeddings.npy"
DOCS_PATH = "./data/processed_documents.json"
MODEL_PATH = "./models/gmm_model.pkl"
OUTPUT_DIR = "./models"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading embeddings and documents")
    X = np.load(EMB_PATH)
    with open(DOCS_PATH, "r") as f:
        docs = json.load(f)

    print("Fitting GMM with K=25")
    gmm = GaussianMixture(n_components=25, covariance_type="diag", random_state=42)
    gmm.fit(X)

    joblib.dump(gmm, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    probs = gmm.predict_proba(X)
    
    sorted_probs = np.sort(probs, axis=1)
    diffs = sorted_probs[:, -1] - sorted_probs[:, -2]
    boundary_idx = np.argmin(diffs)

    print("\n--- Part 2 Analysis ---")
    print(f"Boundary Case Index: {boundary_idx}")
    print(f"Top 2 Probabilities: {sorted_probs[boundary_idx, -1]:.4f} vs {sorted_probs[boundary_idx, -2]:.4f}")
    print(f"Text Snippet: {docs[boundary_idx]['text'][:150]}...")

if __name__ == "__main__":
    main()