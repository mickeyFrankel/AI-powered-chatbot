import os
import csv
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = Path("./chroma_db")
COLLECTION_NAME = "contacts"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

def init_chromadb(db_path: Path, collection_name: COLLECTION_NAME):
    """
    Initialize a persistent Chroma client and get/create the collection
    with a SentenceTransformer embedding function.
    """

    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_fn)

    return client, collection

def load_contacts(collection, filepath, has_header=True):
    """
    Load contacts from a CSV of: first_name, middle_name, last_name
    - Skips empty lines
    - Trims whitespace
    - Ignores duplicates by full name
    Returns number of contacts added.
    """

    fp = Path(filepath)
    if not fp.exists() or not fp.is_file():
        raise FileNotFoundError(f"File {filepath} does not exist or is not a file.")
    
    added = 0
    seen_fullnames = set()

    with fp(mode='r', encoding="utf-8") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)  # Skip header row
        
        ids: List[str] = []
        docs: List[str] = []
        metas: List[dict] = []

        for idx, row in enumerate(reader):
            if not row or all(not c.strip() for c in row):
                continue

            first_name = (row[0] if len(row) > 0 else "").strip()
            middle_name = (row[1] if len(row) > 1 else "").strip()
            last_name = (row[2] if len(row) > 2 else "").strip()

            parts = [p for p in [first_name, middle_name, last_name] if p]
            if not parts:
                continue

            full_contact = " ".join(parts)

            # dedupe by full name
            if full_contact.lower() in seen_fullnames:
                continue
            seen_fullnames.add(full_contact.lower())

            ids.append(f"{fp.stem}-{idx}")
            docs.append(full_contact)
            metas.append(
                {
                    "first_name": first_name,
                    "middle_name": middle_name,
                    "last_name": last_name,
                }
            )

        if ids:
            collection.add(documents=docs, ids=ids, metadatas=metas)
            added = len(ids)

    return added
    
def search_contacts(collection, query, n_results=5):
    """
    Semantic search over contacts.
    Returns lines with similarity and reconstructed full name.
    """
    res = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["distances", "metadatas", "documents"],
    )

    out: List[str] = []
    # Chroma returns distances (e.g., cosine). Similarity ~ (1 - distance).
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    for dist, meta in zip(dists, metas):
        sim = 1.0 - float(dist)
        fn = meta.get("first_name", "")
        mn = meta.get("middle_name", "")
        ln = meta.get("last_name", "")
        full = " ".join([p for p in [fn, mn, ln] if p])
        out.append(f"Similarity: {sim:.4f} | Contact: {full}")
    return out

def start_chat():
    """
    Minimal console chat using OpenAI Chat Completions (modern client).
    Requires OPENAI_API_KEY in environment.
    """
    if not API_KEY:
        print("Missing OPENAI_API_KEY. Put it in a .env file or environment.")
        return

    client = OpenAI(api_key=API_KEY)
    print("Welcome to my chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": user_input}],
            )
            print("Bot:", resp.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")

def main(csv_path: Optional[str] = None, interactive: bool = True):
    # Initialize vector DB
    _, collection = init_chromadb(DB_PATH, COLLECTION_NAME)

    # Optionally (re)load contacts
    if csv_path:
        count = load_contacts(collection, csv_path, has_header=True)
        print(f"Loaded {count} contacts from: {csv_path}")

    if not interactive:
        return

    # Simple REPL for contact search and chat
    print("\nCommands:")
    print("  find <query>     - semantic search over contacts")
    print("  chat             - start OpenAI chat")
    print("  exit             - quit\n")

    while True:
        cmd = input("> ").strip()
        if cmd.lower() == "exit":
            break
        if cmd.lower().startswith("find "):
            query = cmd[5:].strip()
            if not query:
                print("Please provide a query. Example: find john")
                continue
            try:
                results = search_contacts(collection, query, n_results=5)
                if results:
                    for line in results:
                        print(line)
                else:
                    print("No results.")
            except Exception as e:
                print(f"Search error: {e}")
        elif cmd.lower() == "chat":
            start_chat()
        else:
            print("Unknown command. Use: find <query> | chat | exit")

if __name__ == "__main__":
    # Example usage:
    # 1) First run to load:
    #    python app.py  (and then type: exit)
    #
    # Or modify below to hardwire your CSV path:
    #
    # On Windows, escaping backslashes is important. Prefer raw string:
    # csv_example = r"C:\Users\micke\OneDrive\Scripts\contacts.csv"
    csv_example = None  # set a path if you want to auto-load on start
    main(csv_example, interactive=True)