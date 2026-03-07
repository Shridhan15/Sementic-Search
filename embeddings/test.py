from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "computer hardware repair",
    "NASA satellite mission"
]

embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)
print("First embedding vector:", embeddings[0])