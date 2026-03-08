import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Input file containing processed documents
INPUT_FILE = "data/processed_documents.json"


# Output file for the embedding matrix
OUTPUT_FILE = "embeddings/embeddings.npy"

BATCH_SIZE = 64


def load_documents():
    """
    Load processed documents and extract text fields.
    Only the text content is required for embedding generation.
    """
    print("Loading processed documents")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    texts = [doc["text"] for doc in docs]

    print(f"Loaded {len(texts)} documents")

    return texts


def generate_embeddings(texts):

    """
    Convert documents into dense vector embeddings.
    Model: all-MiniLM-L6-v2
    Reasons:
    Lightweight (around 80MB)
    Works efficiently on CPU
    Produces high-quality semantic embeddings
    384 dimensional vectors are compact but expressive
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = []

    print("\nGenerating embeddings.\n")

    # Process documents in batches to reduce memory usage
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):

        batch = texts[i:i+BATCH_SIZE]

        batch_embeddings = model.encode(batch)

        embeddings.extend(batch_embeddings)

    embeddings = np.array(embeddings)

    print("Embedding shape:", embeddings.shape)

    return embeddings


def save_embeddings(embeddings):

    """
    Save embeddings as a Numpy matrix.
    This format is efficient for numerical operations and
    can be loaded directly into FAISS without conversion.
    """

    np.save(OUTPUT_FILE, embeddings)

    print("Embeddings saved to:", OUTPUT_FILE)


if __name__ == "__main__":

    texts = load_documents()

    embeddings = generate_embeddings(texts)

    save_embeddings(embeddings)

    print("\nEmbedding generation complete")