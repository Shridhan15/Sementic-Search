import os
import json
import re
from tqdm import tqdm

DATA_DIR = "data/20_newsgroups"
OUTPUT_FILE = "data/processed_documents.json"


def clean_text(text):
    """
    Basic text cleaning
    """
    text = re.sub(r"\s+", " ", text)   # remove extra whitespace
    text = text.strip()
    return text


def parse_file(filepath):
    """
    Extract Subject + Body from a Usenet file
    """
    with open(filepath, "r", encoding="latin1") as f:
        content = f.read()

    # split header and body
    parts = content.split("\n\n", 1)

    headers = parts[0]
    body = parts[1] if len(parts) > 1 else ""

    subject = ""

    # extract subject
    for line in headers.split("\n"):
        if line.lower().startswith("subject:"):
            subject = line.replace("Subject:", "").strip()
            break

    text = subject + " " + body

    return clean_text(text)


def load_dataset():

    documents = []
    doc_id = 0

    categories = os.listdir(DATA_DIR)

    print(f"\nFound {len(categories)} categories\n")

    for category in categories:

        category_path = os.path.join(DATA_DIR, category)

        if not os.path.isdir(category_path):
            continue

        files = os.listdir(category_path)

        print(f"Processing category: {category} ({len(files)} files)")

        for filename in tqdm(files):

            filepath = os.path.join(category_path, filename)

            try:
                text = parse_file(filepath)

                documents.append({
                    "doc_id": doc_id,
                    "category": category,
                    "text": text
                })

                doc_id += 1

            except Exception as e:
                print("Error reading:", filepath, e)

    return documents


def save_documents(documents):

    print("\nSaving processed dataset...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f)

    print(f"Dataset saved successfully to {OUTPUT_FILE}")


if __name__ == "__main__":

    docs = load_dataset()

    print("\nTotal documents loaded:", len(docs))

    save_documents(docs)

    print("\nExample Document 1:\n")
    print("Category:", docs[0]["category"])
    print("Text:", docs[0]["text"][:500])

    print("\nExample Document 2:\n")
    print("Category:", docs[1]["category"])
    print("Text:", docs[1]["text"][:500])

    print("\nDATA LOADING COMPLETE ")