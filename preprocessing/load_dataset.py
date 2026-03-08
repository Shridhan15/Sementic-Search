import os
import json
import re
from tqdm import tqdm


# Directory containing the original 20 Newsgroups dataset
DATA_DIR = "data/20_newsgroups"
# Output file where processed documents will be stored
OUTPUT_FILE = "data/processed_documents.json"


def clean_text(text):
    """
    Basic text cleaning.

    The 20 Newsgroups dataset contains noises such as
    inconsistent spacing and line breaks. So performing minimal
    cleaning to preserve meaning while removing formatting noise.
    """
    text = re.sub(r"\s+", " ", text)   
    text = text.strip()
    return text


def parse_file(filepath):
    """
    Extract the Subject and Body from a Usenet message.

    Posts contain metadata headers (From, Date, Subject, etc.)
    followed by the actual message body. We retain only the Subject and
    Body because they contain most semantic meaning for search tasks.
    """
    with open(filepath, "r", encoding="latin1") as f:
        content = f.read()

    # Split headers from body using the blank line separator
    parts = content.split("\n\n", 1)

    headers = parts[0]
    body = parts[1] if len(parts) > 1 else ""

    subject = ""

    # Extract the subject line from the headers
    for line in headers.split("\n"):
        if line.lower().startswith("subject:"):
            subject = line.replace("Subject:", "").strip()
            break

    # Combine subject and body into a single text field
    text = subject + " " + body

    return clean_text(text)


def load_dataset():

    """
    Traverse all category folders and get documents.
    Each folder represents a topic category. Keeping the category
    label as metadata so it can be used for filtered retrieval
    or evaluation of clustering quality.
    """

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
                    "doc_id": doc_id,  # unique identifier for the document
                    "category": category,  # original newsgroup topic
                    "text": text  # cleaned text used for embedding
                })

                doc_id += 1

            except Exception as e:
                # Skip problematic files but continue processing
                print("Error reading:", filepath, e)

    return documents


def save_documents(documents):
    """
    Saving processed documents as JSON.
    JSON format is chosen because:
    It is easy to inspect and debug
    It works well with Python data pipelines
    It preserves metadata such as category labels
    """

    print("\nSaving processed dataset")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f)

    print(f"Dataset saved successfully to {OUTPUT_FILE}")


if __name__ == "__main__":

    docs = load_dataset()

    print("\nTotal documents loaded:", len(docs))

    save_documents(docs)

    # Display a few examples to verify the preprocessing
    print("\nExample Document 1:\n")
    print("Category:", docs[0]["category"])
    print("Text:", docs[0]["text"][:500])

    print("\nExample Document 2:\n")
    print("Category:", docs[1]["category"])
    print("Text:", docs[1]["text"][:500])

    print("\nDATA LOADING COMPLETE ")