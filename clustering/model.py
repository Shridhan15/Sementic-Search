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

    """
    Load embeddings and processed documents.
    Embeddings are used for clustering and documents
    are used for later semantic inspection.
    """
    X = np.load(EMB_PATH)
    with open(DOCS_PATH, "r") as f:
        docs = json.load(f)

    """
    Train Gaussian Mixture Model with chosen cluster count.
    GMM provides fuzzy clustering where each document
    belongs to multiple clusters with different probabilities.
    """

    print("Fitting GMM with K=25")
    gmm = GaussianMixture(n_components=25, covariance_type="diag", random_state=42)
    gmm.fit(X)

    """
    Saving trained clustering model.
    """
    joblib.dump(gmm, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()