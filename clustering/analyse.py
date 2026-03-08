import numpy as np
import joblib
import json
import os

EMB_PATH = "./embeddings/embeddings.npy"
DOCS_PATH = "./data/processed_documents.json"
MODEL_PATH = "./models/gmm_model.pkl"
OUTPUT_DIR = "./models"


def analyze():
    """
    Load trained clustering model, embeddings and document metadata.
    """
    gmm = joblib.load(MODEL_PATH)
    X = np.load(EMB_PATH)
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    """
    Compute fuzzy cluster probabilities.
    Each document receives a probability distribution
    across all clusters rather than a single label.
    """
    probs = gmm.predict_proba(X)
    """
    Determine the dominant cluster for each document.

    Although clustering is fuzzy, this step assigns each
    document to the cluster with the highest probability
    for analysis and statistics.
    """

    dominant_clusters = np.argmax(probs, axis=1)

    print("SEMANTIC ANALYSIS")

    """
    Cluster Size Distribution
    showwing how many documents belong to a cluster(for 5 clusters).
    """
    print("\nCluster Size Distribution")
    cluster_sizes = np.bincount(dominant_clusters)

    for cluster_id in range(5):
        print(f"Cluster {cluster_id}: {cluster_sizes[cluster_id]} documents")

    """
    Showing the most representative documents for a cluster. 
    """
    for cluster_id in range(5):

        print(f"\nCluster {cluster_id}:")

        cluster_indices = np.where(dominant_clusters == cluster_id)[0]
        """
        Select the top documents with highest probability
        for this cluster.
        """

        top_indices = cluster_indices[
            np.argsort(probs[cluster_indices, cluster_id])[-3:][::-1]
        ]

        for idx in top_indices:
            print(f"  {docs[idx]['text'][:100]}..")

    """
    Boundary Cases
    Boundary cases are documents where the model
    assigns nearly equal probability to two clusters.

    These documents sit between topics and demonstrate fuzzy cluster membership.
    """
    print("\nBoundary Cases")

    sorted_probs = np.sort(probs, axis=1)

    """
    Compute difference between top two cluster probabilities.
    Smaller difference means stronger boundary case.
    """

    diffs = sorted_probs[:, -1] - sorted_probs[:, -2]
    boundary_indices = np.argsort(diffs)[:3]
    for idx in boundary_indices:

        p = probs[idx]
        """
        Identify the two clusters with highest probability
        for the document.
        """
        top_two = np.argsort(p)[-2:][::-1]

        print(f"\nDocument Index: {idx}")

        print(
            f"Boundary between Cluster {top_two[0]} "
            f"({p[top_two[0]]:.2%}) and Cluster {top_two[1]} "
            f"({p[top_two[1]]:.2%})"
        )

        print(f"Text: {docs[idx]['text'][:150]}...")

    """
    Geniune Uncertainty Cases

    These are documents where even the highest cluster probability is relatively low.
    Here model is unsure where the document belongs, indicating ambiguous or multi topic content.
    """
    print("\nGenuine Uncertainty")

    max_probs = np.max(probs, axis=1)

    uncertain_indices = np.argsort(max_probs)[:3]
    for idx in uncertain_indices:

        print(
            f"\nDocument Index: {idx} "
            f"(Max Confidence: {max_probs[idx]:.2%})"
        )

        print(f"Text: {docs[idx]['text'][:150]}...")

if __name__ == "__main__":
    analyze()