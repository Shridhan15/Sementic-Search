import numpy as np
import faiss
import os



# Path to the embedding matrix
EMBEDDINGS_FILE = "embeddings/embeddings.npy"
# Directory where the vector index will be stored
INDEX_DIR = "vectorStore"
# Final FAISS index file
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")


def build_index():
    """
    Build FAISS vector index for efficient similarity search.
    FAISS is chosen because:
    Extremely fast vector similarity search
    Works well with large embedding collections
    Lightweight and easy to integrate
    """
    print("Loading embeddings")

    embeddings = np.load(EMBEDDINGS_FILE)

    print("Embedding matrix shape:", embeddings.shape)

    dimension = embeddings.shape[1]

    print("Vector dimension:", dimension)

    # Ensure vector store directory exists
    os.makedirs(INDEX_DIR, exist_ok=True)

    """
    IndexFlatL2 performs exact nearest neighbor search using
    Euclidean distance.

    This choice is suitable because:
    Dataset size (around 20k documents) is relatively small
    Exact search ensures maximum retrieval accuracy
    Simpler than approximate indexing methods
    """
    index = faiss.IndexFlatL2(dimension)

    print("Adding vectors to FAISS index")

    index.add(embeddings)

    print("Total vectors indexed:", index.ntotal)

    # Persist the index to disk so it can be reused by the API
    faiss.write_index(index, INDEX_FILE)

    print("Index saved to:", INDEX_FILE)


if __name__ == "__main__":
    build_index()
    print("\nVector database setup complete ")