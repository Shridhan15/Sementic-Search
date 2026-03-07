import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_FILE = "data/processed_documents.json"
OUTPUT_FILE = "embeddings/embeddings.npy"

BATCH_SIZE = 64


def load_documents():
    print("Loading processed documents...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    texts = [doc["text"] for doc in docs]

    print(f"Loaded {len(texts)} documents")

    return texts


def generate_embeddings(texts):

    # We use all-MiniLM-L6-v2 because it provides a good trade-off
    # between embedding quality and CPU speed.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = []

    print("\nGenerating embeddings...\n")

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):

        batch = texts[i:i+BATCH_SIZE]

        batch_embeddings = model.encode(batch)

        embeddings.extend(batch_embeddings)

    embeddings = np.array(embeddings)

    print("Embedding shape:", embeddings.shape)

    return embeddings


def save_embeddings(embeddings):

    np.save(OUTPUT_FILE, embeddings)

    print("Embeddings saved to:", OUTPUT_FILE)


if __name__ == "__main__":

    texts = load_documents()

    embeddings = generate_embeddings(texts)

    save_embeddings(embeddings)

    print("\nEmbedding generation complete")