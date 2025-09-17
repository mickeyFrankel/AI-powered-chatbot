import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# ---------------- Config ----------------
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = Path("./chroma_db")
DEFAULT_TOP_K = 5
DEFAULT_FMT = "json"  # or "yaml"
# ---------------------------------------

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

def init_chroma(path: Path = DB_PATH):
    client = chromadb.PersistentClient(path=str(path))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client, emb_fn

def dataframe_from_file(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if filepath.suffix.lower() == ".csv":
        return pd.read_csv(filepath, encoding="utf-8")
    if filepath.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(filepath)
    raise ValueError("Unsupported file type. Use .csv, .xlsx, or .xls")

def build_collection_for_df(
    client: chromadb.ClientAPI,
    emb_fn,
    df: pd.DataFrame,
    collection_name: str,
    id_prefix: Optional[str] = None,
    include_cols: Optional[List[str]] = None,
) -> chromadb.Collection:
    col = client.get_or_create_collection(name=collection_name, embedding_function=emb_fn)
    # Clear & rebuild to keep it simple (idempotent if same data)
    try:
        col.delete(where={})
    except Exception:
        pass

    docs: List[str] = []
    ids: List[str] = []
    metas: List[Dict[str, Any]] = []

    if include_cols is None:
        include_cols = [c for c in df.columns]

    for i, row in df.iterrows():
        text_parts = []
        meta = {}
        for c in include_cols:
            val = row.get(c, "")
            # stringify scalars; skip NaNs
            s = "" if (pd.isna(val)) else str(val)
            if s:
                text_parts.append(f"{c}: {s}")
                meta[c] = s
        doc = " | ".join(text_parts) if text_parts else ""
        if not doc:
            continue
        ids.append(f"{(id_prefix or collection_name)}-{i}")
        docs.append(doc)
        metas.append(meta)

    if docs:
        col.add(documents=docs, ids=ids, metadatas=metas)
    return col

def vector_search(col: chromadb.Collection, query: str, n_results: int) -> List[Dict[str, Any]]:
    res = col.query(
        query_texts=[query],
        n_results=n_results,
        include=["distances", "metadatas", "documents", "ids"],
    )
    out = []
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    for dist, meta, _id, doc in zip(dists, metas, ids, docs):
        sim = 1.0 - float(dist)
        out.append({"id": _id, "similarity": round(sim, 6), "meta": meta, "doc": doc})
    return out

# -------- Intent parsing (simple rules) --------
FILE_PAT = re.compile(r'(?P<path>([A-Za-z]:\\|\.\/|\.\.\/)[^\s]+?\.(csv|xlsx?|CSV|XLSX?))')
TOP_PAT = re.compile(r'\btop\s+(\d+)\b|\bn\s*=\s*(\d+)', re.IGNORECASE)
FMT_PAT = re.compile(r'\b(json|yaml|yml)\b', re.IGNORECASE)
COLS_PAT = re.compile(r'columns?\s*:\s*([A-Za-z0-9_,\s-]+)', re.IGNORECASE)

SEARCH_HINTS = ("search", "find", "lookup", "query", "match", "nearest", "similar", "vector", "top")

def parse_intent(text: str) -> Dict[str, Any]:
    intent: Dict[str, Any] = {"mode": "chat", "query": text.strip()}
    # file path?
    m_file = FILE_PAT.search(text)
    if m_file:
        intent["file"] = Path(m_file.group("path").strip('"').strip("'"))
    # top K?
    m_top = TOP_PAT.search(text)
    if m_top:
        k = m_top.group(1) or m_top.group(2)
        intent["top_k"] = int(k)
    # format?
    m_fmt = FMT_PAT.search(text)
    if m_fmt:
        fmt = m_fmt.group(1).lower()
        intent["fmt"] = "yaml" if fmt in ("yaml", "yml") else "json"
    # columns?
    m_cols = COLS_PAT.search(text)
    if m_cols:
        cols = [c.strip() for c in m_cols.group(1).split(",") if c.strip()]
        if cols:
            intent["columns"] = cols

    # decide mode: if file present or hints present → search
    if intent.get("file") or any(h in text.lower() for h in SEARCH_HINTS) or ("top_k" in intent) or ("fmt" in intent):
        intent["mode"] = "search"
    return intent

# --------------- Chat ---------------
def chat_once(prompt: str) -> str:
    if not API_KEY:
        return "Chat disabled: set OPENAI_API_KEY in .env."
    client = OpenAI(api_key=API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

# ------------- Export helpers -------------
def export_results(results: List[Dict[str, Any]], fmt: str = DEFAULT_FMT) -> Tuple[str, str]:
    """
    Returns (text, filename). Also writes the file to disk.
    """
    fmt = fmt.lower()
    if fmt not in {"json", "yaml"}:
        fmt = DEFAULT_FMT
    if fmt == "json":
        text = json.dumps(results, ensure_ascii=False, indent=2)
        fname = "export.json"
        Path(fname).write_text(text, encoding="utf-8")
        return text, fname
    else:
        text = yaml.safe_dump(results, allow_unicode=True, sort_keys=False)
        fname = "export.yaml"
        Path(fname).write_text(text, encoding="utf-8")
        return text, fname

# --------------- Main REPL ---------------
def main():
    print("Smart bot ready. Type your request (e.g., 'top 7 similar to משה in C:\\path\\contacts.csv as yaml').")
    client, emb_fn = init_chroma()

    while True:
        user = input("> ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        intent = parse_intent(user)
        mode = intent["mode"]

        if mode == "chat":
            print(chat_once(intent["query"]))
            continue

        # SEARCH mode
        file = intent.get("file")
        top_k = int(intent.get("top_k", DEFAULT_TOP_K))
        fmt = intent.get("fmt", DEFAULT_FMT)
        cols = intent.get("columns")  # optional

        if not file:
            print("I need a data source. Include a path to a CSV/Excel file in your message.")
            continue

        try:
            df = dataframe_from_file(file)
        except Exception as e:
            print(f"Failed to read file: {e}")
            continue

        collection_name = file.stem  # one collection per file
        col = build_collection_for_df(client, emb_fn, df, collection_name, id_prefix=file.stem, include_cols=cols)
        results = vector_search(col, intent["query"], n_results=top_k)
        text, out_file = export_results(results, fmt=fmt)
        print(text)
        print(f"\nSaved to {out_file}")

if __name__ == "__main__":
    main()
