import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Path to embeddings generated in Part 1
EMB_PATH = "./embeddings/embeddings.npy"
OUTPUT_DIR = "outputs"

"""
Try out different  cluster counts(k) to evaluate
Instead of assuming 20 clusters,
we test multiple values to discover the real semantic structure.
# """
Ks = [5, 10, 15, 16, 18, 20, 22, 23, 25,27, 30]  

def run_analysis():
    # Load embedding matrix
    X = np.load(EMB_PATH)
    print(f"Loaded embeddings with shape: {X.shape}")

    # Store evaluation results for each K
    results = {}

    print("\nStarting Model Selection")
    for k in Ks:
        print(f"For K={k}", end=" ", flush=True)
        
        """
        Train Gaussian Mixture Model
        GMM supports fuzzy clustering, so each document receives a probability distribution across clusters.
        """
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42)
        gmm.fit(X)

        """
        Compute BIC score.

        BIC measures model fit while penalizing overly complex models.
        Lower BIC means a better trade off between accuracy
        and model complexity.
        """
        bic_score = float(gmm.bic(X))
        
        """
        Silhouette score requires hard cluster assignments.
        So KMeans is temporarily used to generate labels.
        """

        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(X)

        """
        Silhouette score computation is expensive for large datasets.
        To improve efficiency, randomly sample up to 5000 documents.
        """
        
        sample_size = min(5000, X.shape[0])
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        sil_score = float(silhouette_score(X[indices], labels[indices]))
        
        results[k] = {
            'bic': bic_score,
            'silhouette': sil_score
        }
        print(f"Done (BIC: {bic_score:.2f}, Silhouette: {sil_score:.4f})")

    """
    Save model selection metrics to JSON file
    so they can be inspected later or used in reports.
    """

    json_path = os.path.join(OUTPUT_DIR, "model_selection.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    """
    Visualization of evaluation metrics.

    Two plots are generated:
    1. BIC vs number of clusters
    2. Silhouette score vs number of clusters

    These graphs help visually identify the most suitable K.
    """

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(list(results.keys()), [r['bic'] for r in results.values()], 'o-', color='tab:blue', linewidth=2)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('BIC Score')
    plt.title('GMM Model Selection (BIC)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(list(results.keys()), [r['silhouette'] for r in results.values()], 's-', color='tab:red', linewidth=2)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Clustering Quality (Silhouette)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "model_selection_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    
    plt.show()
 
if __name__ == "__main__":
    run_analysis()