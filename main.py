import os
import json
import joblib
import faiss
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
from cache.build_cache import SemanticCache

"""
FastAPI service for semantic search with caching.

Flow:
query -> embedding -> cluster detection -> cache lookup
If cache hit then return cached result
If miss then search FAISS, store in cache  and return result
"""
app = FastAPI(title="Trademarkia Semantic Search API")

encoder = SentenceTransformer("all-MiniLM-L6-v2")

gmm = joblib.load("models/gmm_model.pkl")

index = faiss.read_index("vectorStore/faiss.index")

with open("data/processed_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

"""
Initialize semantic cache.
threshold controls how similar queries must be
to reuse a cached result.
"""

semantic_cache = SemanticCache(threshold=0.80)
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def perform_query(request: QueryRequest):
    """
    Main search endpoint.
    Steps:
    1 Embed query
    2 Predict cluster
    3 Check semantic cache
    4 If miss then  search FAISS, store in cache  and return result
    """
    query_text = request.query

    # convert query into embedding
    query_emb = encoder.encode([query_text])[0].astype("float32")

    # determine dominant cluster
    probs = gmm.predict_proba(query_emb.reshape(1, -1))[0]
    dominant_cluster = int(np.argmax(probs))

    # try cache lookup
    cache_result = semantic_cache.lookup(query_text, query_emb, dominant_cluster)
    if cache_result:
        return {
            "query": query_text,
            "cache_hit": True,
            "matched_query": cache_result["matched_query"],
            "similarity_score": cache_result["similarity_score"],
            "result": cache_result["response"],
            "dominant_cluster": dominant_cluster
        }

    """
    Cache miss then perform FAISS vector search.
    """

    _, indices = index.search(query_emb.reshape(1, -1), k=1)

    doc_idx = int(indices[0][0])

    result_text = documents[doc_idx]["text"]

    # store result in cache
    semantic_cache.update(dominant_cluster, query_text, query_emb, result_text)

    return {
        "query": query_text,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result_text,
        "dominant_cluster": dominant_cluster
    }


@app.get("/cache/stats")
async def get_stats():
    """
    Return cache statistics.
    """
    return semantic_cache.get_stats()


@app.delete("/cache")
async def flush_cache():
    """
    Clear the cache and reset stats.
    """
    semantic_cache.clear()

    return {"message": "Cache flushed and stats reset successfully"}


@app.get("/cache")
async def view_cache():
    """
    Returns current cache contents grouped by cluster.
    Useful for debugging and inspecting cached queries.
    """
    
    cache_view = {}

    for cluster_id, entries in semantic_cache.store.items():
        cache_view[cluster_id] = [
            {
                "query": entry["query"],
                "response_preview": entry["response"][:100]
            }
            for entry in entries
        ]

    return {
        "total_clusters": len(cache_view),
        "cache_contents": cache_view
    }