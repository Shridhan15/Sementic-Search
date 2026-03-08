import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:
    """
    Simple semantic cache to avoid recomputing similar queries.
    Queries are grouped by cluster to reduce search space.
    Matching happens in two steps:
    Exact text match 
    Semantic similarity using cosine similarity
    """

    def __init__(self, threshold=0.80):
        """
        threshold: minimum cosine similarity required
        to consider two queries semantically the same.
        """
        # cluster_id -> list of cached entries
        self.store = {}
        # similarity threshold
        self.threshold = threshold
        # cache stats
        self.hits = 0
        self.misses = 0

        # exact query lookup for O(1) match
        self.exact_match_map = {}

    def lookup(self, query_text, query_vector, cluster_id):
        """
        Try to find a cached response.
        1 Check exact query match
        2 Search inside the predicted cluster
        3 Compare vectors using cosine similarity
        """

        # exact match check
        if query_text in self.exact_match_map:
            self.hits += 1
            return self.exact_match_map[query_text]

        # if cluster bucket doesn't exist it is a miss
        if cluster_id not in self.store:
            self.misses += 1
            return None

        # check semantic similarity inside cluster
        for entry in self.store[cluster_id]:

            sim = cosine_similarity(
                query_vector.reshape(1, -1),
                entry["vector"].reshape(1, -1)
            )[0][0]

            if sim >= self.threshold:
                self.hits += 1
                return {
                    "response": entry["response"],
                    "similarity_score": round(float(sim), 4),
                    "matched_query": entry["query"]
                }

        # nothing similar found
        self.misses += 1
        return None

    def update(self, cluster_id, query_text, query_vector, response):
        """
        Store a new query and its response in the cache.
        Queries are stored inside their cluster bucket.
        """

        if cluster_id not in self.store:
            self.store[cluster_id] = []

        cache_entry = {
            "query": query_text,
            "vector": query_vector,
            "response": response
        }

        self.store[cluster_id].append(cache_entry)

        #  store for exact match lookup
        self.exact_match_map[query_text] = {
            "response": response,
            "similarity_score": 1.0,
            "matched_query": query_text
        }

    def get_stats(self):
        """
        Return basic cache statistics.
        """

        total = self.hits + self.misses

        return {
            "total_entries": sum(len(v) for v in self.store.values()),
            "hit_count": self.hits,
            "miss_count": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0
        }

    def clear(self):
        """
        Reset the cache and stats.
        """
        self.store = {}
        self.exact_match_map = {}
        self.hits = 0
        self.misses = 0