import numpy as np
import faiss
import os

EMBEDDINGS_FILE = "embeddings/embeddings.npy"
INDEX_DIR = "vectorStore"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")


def build_index():
    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_FILE)

    print("Embedding matrix shape:", embeddings.shape)

    dimension = embeddings.shape[1]

    print("Vector dimension:", dimension)

    # Ensure folder exists
    os.makedirs(INDEX_DIR, exist_ok=True)

    index = faiss.IndexFlatL2(dimension)

    print("Adding vectors to FAISS index...")

    index.add(embeddings)

    print("Total vectors indexed:", index.ntotal)

    faiss.write_index(index, INDEX_FILE)

    print("Index saved to:", INDEX_FILE)


if __name__ == "__main__":
    build_index()
    print("\nVector database setup complete ")