import numpy as np
import joblib
import json
import os

def analyze(): 
    gmm = joblib.load("./models/gmm_model.pkl")
    X = np.load("./embeddings/embeddings.npy")
    with open("./data/processed_documents.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
 
    probs = gmm.predict_proba(X)
    dominant_clusters = np.argmax(probs, axis=1)

    print("SEMANTIC ANALYSIS ")
 
    for cluster_id in range(5): 
        print(f"\nCluster {cluster_id}:") 
        cluster_indices = np.where(dominant_clusters == cluster_id)[0] 
        top_indices = cluster_indices[np.argsort(probs[cluster_indices, cluster_id])[-3:][::-1]]
        
        for idx in top_indices:
            print(f"  {docs[idx]['text'][:100]}..")
 
    print("\n Boundary Cases  ")
    sorted_probs = np.sort(probs, axis=1)
    diffs = sorted_probs[:, -1] - sorted_probs[:, -2]
     
    boundary_indices = np.argsort(diffs)[:3]
    
    for idx in boundary_indices:
        p = probs[idx]
        top_two = np.argsort(p)[-2:][::-1]
        print(f"\nDocument Index: {idx}")
        print(f" Boundary between Cluster {top_two[0]} ({p[top_two[0]]:.2%}) and Cluster {top_two[1]} ({p[top_two[1]]:.2%})")
        print(f" Text: {docs[idx]['text'][:150]}...")
 
    print("\nGenuine Uncertainty ") 
    max_probs = np.max(probs, axis=1)
    uncertain_indices = np.argsort(max_probs)[:3]

    for idx in uncertain_indices:
        print(f"\nDocument Index: {idx} (Max Confidence: {max_probs[idx]:.2%})")
        print(f" Text: {docs[idx]['text'][:150]}...")

if __name__ == "__main__":
    analyze()