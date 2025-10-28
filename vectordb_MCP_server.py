# vectordb_MCP_server.py (hardened)
import contextlib, sys, os, logging
from typing import Any, Dict, List, Optional, TypedDict
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

# Ensure stderr logging (NEVER print to stdout in stdio servers)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("vectordb-mcp")

@contextlib.contextmanager
def stdout_to_stderr():
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old

# Lazy singleton for QA system, initialized on first tool call
_qa = None

def get_chroma_dir() -> str:
    # Allow override via env; default to ./chroma_db next to this file
    here = os.path.dirname(__file__)
    return os.environ.get("CHROMA_DB_DIR", os.path.join(here, "chroma_db"))

def get_qa():
    global _qa
    if _qa is None:
        with stdout_to_stderr():
            from vectoric_search import AdvancedVectorDBQASystem  # local import to avoid import-time prints
            _qa = AdvancedVectorDBQASystem(persist_directory=get_chroma_dir())
    return _qa

# --------- Structured types ----------
class Source(BaseModel):
    source: str = Field(description="Source or file identifier")
    score: float = Field(description="Similarity score (higher is more similar)")
    chunk: Optional[str] = Field(default=None, description="Snippet of the matched text")

class QAResult(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)

class SearchResult(BaseModel):
    query: str
    results: List[Source] = Field(default_factory=list)
    filters: Optional[Dict[str, Any]] = None

mcp = FastMCP("vectordb")

@mcp.tool()
def ping() -> Dict[str, Any]:
    """Health check and environment info."""
    with stdout_to_stderr():
        qa = get_qa()
        stats = qa.get_collection_stats() if hasattr(qa, "get_collection_stats") else {}
        return {
            "ok": True,
            "cwd": os.getcwd(),
            "chroma_dir": get_chroma_dir(),
            "stats": stats
        }

@mcp.tool()
def stats() -> Dict[str, Any]:
    """Return collection statistics."""
    with stdout_to_stderr():
        return get_qa().get_collection_stats()

@mcp.tool()
def ask(question: str, top_k: int = 5) -> QAResult:
    """Answer a question using the vector DB (uses internal agent if available)."""
    with stdout_to_stderr():
        qa = get_qa()
        # Prefer agent if implemented; fallback to semantic search
        if hasattr(qa, "agent_answer"):
            try:
                answer = qa.agent_answer(question)
            except Exception as e:
                log.exception("agent_answer failed: %s", e)
                answer = None
        else:
            answer = None

        sources: List[Source] = []
        try:
            srch = qa.search(question, n_results=top_k)
            # normalize
            results = srch.get("results") or srch.get("documents") or []
            metadatas = srch.get("metadatas") or []
            distances = srch.get("distances") or []
            # Chroma sometimes returns per-query lists; flatten first item
            if results and isinstance(results[0], list):
                results = results[0]
            if metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0]
            if distances and isinstance(distances[0], list):
                distances = distances[0]
            for i, meta in enumerate(metadatas):
                score = (1.0 - float(distances[i])) if i < len(distances) else 0.0
                sources.append(Source(
                    source=str(meta.get("source") or meta.get("filename") or f"doc_{i}"),
                    score=score,
                    chunk=str(meta.get("chunk") or meta.get("text") or "")
                ))
        except Exception as e:
            log.exception("search failed: %s", e)

        if answer is None:
            # Basic answer from top source chunk(s)
            combined = "\n\n".join(s.chunk or "" for s in sources[:3]).strip() or "(no context found)"
            answer = f"Top context:\n{combined}"
        return QAResult(answer=answer, sources=sources)

@mcp.tool()
def search(query: str, n_results: int = 10, filters_json: Optional[str] = None) -> SearchResult:
    """Semantic search with optional filters (JSON as string)."""
    with stdout_to_stderr():
        qa = get_qa()
        filters = None
        if filters_json:
            try:
                import json as _json
                filters = _json.loads(filters_json)
            except Exception as e:
                log.warning("Invalid filters_json: %s", e)
        if hasattr(qa, "semantic_search_with_filters"):
            res = qa.semantic_search_with_filters(query, n_results=n_results, filters=filters)
            results = res.get("results") or []
            # normalize to Source[]
            sources: List[Source] = []
            for r in results:
                sources.append(Source(
                    source=str(r.get("source") or r.get("filename") or ""),
                    score=float(r.get("score") or 0.0),
                    chunk=str(r.get("text") or r.get("chunk") or "")
                ))
            return SearchResult(query=query, results=sources, filters=filters)
        else:
            srch = qa.search(query, n_results=n_results)
            # map to Source[]
            results = srch.get("results") or []
            metadatas = srch.get("metadatas") or []
            distances = srch.get("distances") or []
            if results and isinstance(results[0], list):
                results = results[0]
            if metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0]
            if distances and isinstance(distances[0], list):
                distances = distances[0]
            sources: List[Source] = []
            for i, meta in enumerate(metadatas):
                score = (1.0 - float(distances[i])) if i < len(distances) else 0.0
                sources.append(Source(
                    source=str(meta.get("source") or meta.get("filename") or f"doc_{i}"),
                    score=score,
                    chunk=str(meta.get("chunk") or meta.get("text") or "")
                ))
            return SearchResult(query=query, results=sources, filters=filters)

@mcp.tool()
def ingest_file(path: str) -> Dict[str, Any]:
    """Load a single file (csv/xlsx/docx/pdf/txt) and add to the DB."""
    with stdout_to_stderr():
        qa = get_qa()
        try:
            df = qa.load_file(path)
            added = qa.add_documents(df)
            return {"ok": True, "rows_added": added, "path": path}
        except Exception as e:
            log.exception("ingest_file failed: %s", e)
            return {"ok": False, "error": str(e), "path": path}

@mcp.tool()
def list_sources() -> Dict[str, int]:
    """Return per-source document counts."""
    with stdout_to_stderr():
        qa = get_qa()
        if hasattr(qa, "count_sources"):
            return qa.count_sources()
        elif hasattr(qa, "list_sources"):
            return qa.list_sources()
        return {}

@mcp.tool()
def purge_duplicates() -> Dict[str, Any]:
    """Delete duplicate rows (same basename+row_id)."""
    with stdout_to_stderr():
        qa = get_qa()
        try:
            qa.purge_duplicates()
            return {"ok": True}
        except Exception as e:
            log.exception("purge_duplicates failed: %s", e)
            return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    # Run as an MCP stdio server
    mcp.run(transport="stdio")