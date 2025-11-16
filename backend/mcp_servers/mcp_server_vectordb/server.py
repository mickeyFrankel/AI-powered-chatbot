# mcp_server_vectordb/server.py
import argparse
import asyncio
import json
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# --- Project import path (so `vectordb_system.py` at project root is importable) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Third-party and app imports ---
from mcp.server import Server
from mcp.server.stdio import stdio_server

# If your class name changes, edit here:
from vectoric_search import AdvancedVectorDBQASystem  # noqa: E402

# --- Logging (stderr only to avoid corrupting MCP stdio protocol) ---
LOG_LEVEL = os.getenv("VDB_MCP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("vectordb-mcp")

# --- Constants/Helpers ---
APP_NAME = "vectordb"
APP_VERSION = "0.3.0"

SUPPORTED_EXTS = {".csv", ".xlsx", ".xls", ".docx", ".pdf", ".txt", ".md"}

def _as_json(obj: Any) -> Dict[str, Any]:
    return {"type": "json", "json": obj}

def _ok(**kwargs) -> Dict[str, Any]:
    payload = {"ok": True, **kwargs}
    return _as_json(payload)

def _fail(msg: str, **kwargs) -> Dict[str, Any]:
    log.warning("Tool error: %s", msg)
    payload = {"ok": False, "error": msg, **kwargs}
    return _as_json(payload)

def _safe_int(value: Any, default: int, min_value: int = 1, max_value: int = 100) -> int:
    try:
        iv = int(value)
    except Exception:
        return default
    return max(min_value, min(max_value, iv))

def _resolve_path(p: str) -> Path:
    # Accept absolute or relative (relative to project root)
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


async def run_server():
    # Persist dir: env CHROMA_PATH or project_root/chroma_db
    persist_dir = os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db"))
    persist_dir = str(Path(persist_dir))

    log.info("Starting MCP server '%s' v%s", APP_NAME, APP_VERSION)
    log.info("Using CHROMA_PATH: %s", persist_dir)

    # Ensure qa is defined in all paths
    qa: Optional[AdvancedVectorDBQASystem] = None

    @contextlib.contextmanager
    def _stdout_to_stderr():
        old = sys.stdout
        sys.stdout = sys.stderr
        try:
            yield
        finally:
            sys.stdout = old

    try:
        with _stdout_to_stderr():
            qa = AdvancedVectorDBQASystem(persist_directory=persist_dir)
    except Exception as e:
        # Keep the server booting so we can report health gracefully
        log.exception("Failed to initialize AdvancedVectorDBQASystem: %s", e)
        qa = None  # explicit for clarity

    server = Server(APP_NAME)

    # ---------------- Health Check ----------------
    @server.tool(
        "ping",
        description="Lightweight health check. Returns ok, counts and environment info."
    )
    async def tool_ping():
        doc_count = None
        source_count = None
        if qa is not None:
            # Try optional counters if present
            for meth, target in (("count_documents", "doc_count"), ("count_sources", "source_count")):
                try:
                    if hasattr(qa, meth):
                        val = getattr(qa, meth)()
                        if target == "doc_count":
                            doc_count = val
                        else:
                            source_count = val
                except Exception as e:
                    log.debug("Counter %s failed: %s", meth, e)

        return _ok(
            app=APP_NAME,
            version=APP_VERSION,
            chroma_path=persist_dir,
            has_openai_key=bool(os.getenv("OPENAI_API_KEY")),
            doc_count=doc_count,
            source_count=source_count,
            initialized=qa is not None,
        )

    # ---------------- Core search ----------------
    @server.tool("search", description="Semantic search over the vector DB.")
    async def tool_search(query: str, n_results: int = 5):
        if qa is None:
            return _fail("QA system not initialized.")
        query = (query or "").strip()
        if not query:
            return _fail("Missing required argument: query")
        n = _safe_int(n_results, default=5)
        try:
            results = qa.search(query, n_results=n)
            return _as_json(results)
        except Exception as e:
            return _fail(f"search failed: {e}")

    # ---------------- Search with filters (optional) ----------------
    @server.tool(
        "search_with_filters",
        description="Semantic search with optional metadata filters. Pass filters as a JSON object."
    )
    async def tool_search_with_filters(query: str, n_results: int = 5, filters: Optional[dict] = None):
        if qa is None:
            return _fail("QA system not initialized.")
        query = (query or "").strip()
        if not query:
            return _fail("Missing required argument: query")
        n = _safe_int(n_results, default=5)
        try:
            # If your AdvancedVectorDBQASystem exposes this as `semantic_search_with_filters`,
            # call that here. Otherwise adapt accordingly.
            if hasattr(qa, "semantic_search_with_filters"):
                results = qa.semantic_search_with_filters(query, n_results=n, filters=filters or {})
            else:
                # Fallback to plain search if the method doesn't exist
                results = qa.search(query, n_results=n)
            return _as_json(results)
        except Exception as e:
            return _fail(f"search_with_filters failed: {e}", filters=filters or {})

    # ---------------- Add file ----------------
    @server.tool(
        "add_file",
        description="Load a file (csv/xlsx/docx/pdf/txt) into the DB. Returns how many chunks were added."
    )
    async def tool_add_file(path: str):
        if qa is None:
            return _fail("QA system not initialized.")
        path = (path or "").strip()
        if not path:
            return _fail("Missing required argument: path")

        fpath = _resolve_path(path)
        if not fpath.exists():
            return _fail(f"File not found: {str(fpath)}")

        ext = fpath.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            return _fail(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTS)}")

        try:
            docs = qa.load_file(str(fpath))
            qa.add_documents(docs)
            return _ok(added=len(docs), path=str(fpath))
        except Exception as e:
            return _fail(f"add_file failed: {e}", path=str(fpath))

    # ---------------- Dedupe ----------------
    @server.tool(
        "purge_duplicates",
        description="Delete duplicate rows (basename + row_id)."
    )
    async def tool_purge_dups():
        if qa is None:
            return _fail("QA system not initialized.")
        try:
            qa.purge_duplicates()
            return _ok(message="dedupe complete")
        except Exception as e:
            return _fail(f"purge_duplicates failed: {e}")

    # ---------------- Backfill names (optional util) ----------------
    @server.tool(
        "backfill_names",
        description="Optional utility to backfill missing display names/metadata."
    )
    async def tool_backfill_names():
        if qa is None:
            return _fail("QA system not initialized.")
        if not hasattr(qa, "backfill_names"):
            return _fail("backfill_names not supported by this QA system.")
        try:
            out = qa.backfill_names()
            return _ok(result=out)
        except Exception as e:
            return _fail(f"backfill_names failed: {e}")

    # ---------------- Catalog/name utilities ----------------
    @server.tool("list_by_prefix", description="List names by prefix.")
    async def tool_list_by_prefix(prefix: str, limit: int = 50):
        if qa is None:
            return _fail("QA system not initialized.")
        prefix = (prefix or "")
        lim = _safe_int(limit, default=50, min_value=1, max_value=1000)
        try:
            out = qa.list_by_prefix(prefix, limit=lim)
            return _as_json(out)
        except Exception as e:
            return _fail(f"list_by_prefix failed: {e}", prefix=prefix, limit=lim)

    @server.tool("names_by_length", description="List names with exact length.")
    async def tool_names_by_length(length: int, limit: int = 50):
        if qa is None:
            return _fail("QA system not initialized.")
        ln = _safe_int(length, default=0, min_value=0, max_value=10000)
        lim = _safe_int(limit, default=50, min_value=1, max_value=1000)
        try:
            out = qa.names_by_length(ln, limit=lim)
            return _as_json(out)
        except Exception as e:
            return _fail(f"names_by_length failed: {e}", length=ln, limit=lim)

    @server.tool("names_containing", description="List names containing the given substring.")
    async def tool_names_containing(substr: str, limit: int = 50):
        if qa is None:
            return _fail("QA system not initialized.")
        s = (substr or "")
        lim = _safe_int(limit, default=50, min_value=1, max_value=1000)
        try:
            out = qa.names_containing(s, limit=lim)
            return _as_json(out)
        except Exception as e:
            return _fail(f"names_containing failed: {e}", substr=s, limit=lim)

    @server.tool(
        "names_by_prefix_and_length",
        description="List names that start with prefix and have a given length."
    )
    async def tool_names_by_prefix_and_length(prefix: str, length: int, limit: int = 50):
        if qa is None:
            return _fail("QA system not initialized.")
        p = (prefix or "")
        ln = _safe_int(length, default=0, min_value=0, max_value=10000)
        lim = _safe_int(limit, default=50, min_value=1, max_value=1000)
        try:
            out = qa.names_by_prefix_and_length(p, ln, limit=lim)
            return _as_json(out)
        except Exception as e:
            return _fail(
                f"names_by_prefix_and_length failed: {e}",
                prefix=p, length=ln, limit=lim
            )

    @server.tool("letter_histogram", description="Histogram of first letters.")
    async def tool_letter_histogram():
        if qa is None:
            return _fail("QA system not initialized.")
        try:
            out = qa.letter_histogram()
            return _as_json(out)
        except Exception as e:
            return _fail(f"letter_histogram failed: {e}")

    @server.tool("length_histogram", description="Histogram of name lengths.")
    async def tool_length_histogram():
        if qa is None:
            return _fail("QA system not initialized.")
        try:
            out = qa.length_histogram()
            return _as_json(out)
        except Exception as e:
            return _fail(f"length_histogram failed: {e}")

    @server.tool("list_sources", description="List unique source identifiers/URIs.")
    async def tool_list_sources():
        if qa is None:
            return _fail("QA system not initialized.")
        try:
            out = qa.list_sources()
            return _as_json(out)
        except Exception as e:
            return _fail(f"list_sources failed: {e}")

    @server.tool("count_sources", description="Return number of unique sources.")
    async def tool_count_sources():
        if qa is None:
            return _fail("QA system not initialized.")
        try:
            out = qa.count_sources()
            return _as_json({"count": int(out)})
        except Exception as e:
            return _fail(f"count_sources failed: {e}")

    # --- MCP stdio bridge ---
    async with stdio_server() as (read, write):
        await server.run(read, write)


def cli():
    parser = argparse.ArgumentParser("mcp_server_vectordb")
    parser.add_argument("command", nargs="?", default="serve", choices=["serve"])
    args = parser.parse_args()
    if args.command == "serve":
        asyncio.run(run_server())


if __name__ == "__main__":
    cli()
