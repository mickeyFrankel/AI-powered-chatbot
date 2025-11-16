# Quick Reference: Your Chatbot Architecture

## 🎯 Simple Answer

**Q: Does the chatbot have distinct MCPs?**
**A: YES! You have 2 separate MCP servers:**

1. **VectorDB MCP** → Talks to ChromaDB (semantic search)
2. **PostgreSQL MCP** → Talks to PostgreSQL (SQL queries)

---

## 🤔 ChromaDB = VectorDB?

**YES!** ChromaDB **IS** a vector database (not a wrapper).

```
┌─────────────────────────────────────┐
│     "Vector Database" Category      │
├─────────────────────────────────────┤
│  • ChromaDB     (what you use)      │
│  • Pinecone     (cloud service)     │
│  • Weaviate     (GraphQL API)       │
│  • Milvus       (high performance)  │
│  • Qdrant       (Rust-based)        │
│  • FAISS        (Facebook AI)       │
└─────────────────────────────────────┘
```

All of these are **different vector database products**.
ChromaDB is just one implementation.

---

## 📊 Your Data Flow

```
User Query: "אמא של אשתי"
        ↓
┌───────────────────────┐
│  chatbot.py           │
│  (LangChain Agent)    │
└───────────────────────┘
        ↓
   Decides which tool to use
        ↓
    ┌───────┴───────┐
    ↓               ↓
[VectorDB MCP]  [PostgreSQL MCP]
    ↓               ↓
[ChromaDB]      [PostgreSQL]
    ↓               ↓
"חמותי היקרה"   (no match)
```

**Semantic queries** → VectorDB MCP → ChromaDB
**Structured queries** → PostgreSQL MCP → PostgreSQL

---

## 🔧 Your MCP Servers

### VectorDB MCP
- **Files:** `vectordb_MCP_server.py`, `mcp_server_vectordb/`
- **Database:** ChromaDB (`./chroma_db/`)
- **Best for:** Hebrew search, typos, semantic similarity
- **Returns:** Top 5 most similar results (default)

### PostgreSQL MCP  
- **File:** `postgres_mcp_server.py`
- **Database:** PostgreSQL (`chatbot_db`)
- **Best for:** Exact matches, SQL filters, structured data
- **Returns:** Exact query results

---

## 💡 Key Insight

You're NOT using a generic "vectordb" with ChromaDB as a wrapper.

You're using **ChromaDB** (a specific vector database) through an **MCP server** (a protocol interface).

The MCP server is the wrapper, not ChromaDB!

```
MCP Server (wrapper/interface)
    ↓
ChromaDB (the actual vector database)
    ↓
Your data (1,917 embedded contacts)
```

---

## 📝 Summary

✅ **2 MCP servers** (VectorDB + PostgreSQL)
✅ **ChromaDB is a vector database** (not a wrapper)
✅ **MCP is the protocol** (standardized interface)
✅ **Top 5 results by default** for fuzzy search

You're all set! 🚀
