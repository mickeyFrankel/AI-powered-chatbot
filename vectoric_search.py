import csv
from sentence_transformers import SentenceTransformer, util
import chromadb
from chromadb.utils import embedding_functions
# Both ChromaDB and sentence_transformers support embedding Hebrew strings, provided the underlying model supports multilingual text.
# The 'all-MiniLM-L6-v2' model is trained on English and some other languages, but its performance on Hebrew may be limited.
# For better Hebrew support, consider using a multilingual model like 'paraphrase-multilingual-MiniLM-L12-v2' from sentence_transformers.
# ChromaDB's embedding_functions can use any HuggingFace-compatible model, including multilingual ones.
# Example:
# embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
# embedding = embedding_fn([hebrew_text])

def main():
    print("Enter contact for search:")
    contact = input().strip()
    # model = SentenceTransformer('all-MiniLM-L12-v2')
    
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    collection = client.get_or_create_collection(name="contacts", embedding_function=embedding_fn)
    # contact_embedding = model.encode(contact, convert_to_tensor=True)

    with open("C:\\Users\\micke\\OneDrive\\Scripts\\contacts.csv", mode='r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            first_name = row[0]
            middle_name = row[1]
            last_name = row[2]
            full_contact = f"{first_name} {middle_name} {last_name}".strip()
            # full_contact_embedding = model.encode(full_contact, convert_to_tensor=True)
            # similarity = util.pytorch_cos_sim(contact_embedding, full_contact_embedding).item()
            collection.add(documents=[full_contact], ids = [str(idx)], metadatas=[{"first_name": first_name,
            "middle_name": middle_name,
            "last_name": last_name}])

    results = collection.query(query_texts=[contact], n_results=5)
    for score, metadata in zip(results['distances'][0], results['metadatas'][0]):
        print(f"Similarity: {1 - score:.4f}, Contact: {metadata['first_name']} {metadata['middle_name']} {metadata['last_name']}")
    # results.sort(key=lambda x: x[0], reverse=True)
    # for sim, row in results[:5]:
    #     print(f"Similarity: {sim:.4f}, Contact: {row}")

if __name__ == "__main__":
    main()