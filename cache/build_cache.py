import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:
    def __init__(self, threshold=0.85):
        """
        The data structure is a dictionary where keys are Cluster IDs.
        This ensures O(1) routing to a specific 'topic bucket'.
        """
        self.store = {}  
        self.threshold = threshold 
        self.hits = 0
        self.misses = 0

    def lookup(self, query_vector, cluster_id):
        """
        Instead of searching the whole cache, we only search the assigned cluster.
        This solves the efficiency problem for large caches.
        """
        if cluster_id not in self.store:
            self.misses += 1
            return None

        for entry in self.store[cluster_id]:
            sim = cosine_similarity(
                query_vector.reshape(1, -1), 
                entry['vector'].reshape(1, -1)
            )[0][0]
            
            if sim >= self.threshold:
                self.hits += 1
                return {
                    "response": entry['response'],
                    "similarity": round(float(sim), 4),
                    "matched_query": entry['query']
                }
        
        self.misses += 1
        return None

    def update(self, cluster_id, query_text, query_vector, response):
        """
        Stores the query and its vector in the specific cluster bucket.
        """
        if cluster_id not in self.store:
            self.store[cluster_id] = []
        
        self.store[cluster_id].append({
            "query": query_text,
            "vector": query_vector,
            "response": response
        })

    def get_stats(self):
        total = self.hits + self.misses
        return {
            "total_entries": sum(len(v) for v in self.store.values()),
            "hit_count": self.hits,
            "miss_count": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0
        }