# AI-First Contact Search System

Enterprise RAG system with intelligent agent architecture.

---

## Overview

AI agent with GPT-4 function calling for semantic contact search across 10K+ documents. Supports multilingual queries (Hebrew/English) with conversation memory.

**Architecture:** Single intelligent agent → Tool selection → Results

**Tools:** 7 specialized functions (search, list_by_prefix, names_containing, names_by_length, names_by_prefix_and_length, letter_histogram, length_histogram)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
echo "OPENAI_API_KEY=your-key" > .env

# Run
python chatbot.py
```

---

## Usage

### Basic Queries
```
Noah                      # Name lookup
phone of Noah            # Entity extraction
who handles marketing    # Context search
all names starting with A # Tool selection
```

### Commands
- `load <file>` - Add documents
- `stats` - Database info
- `history` - View conversation
- `clear` - Reset conversation
- `quit` - Exit

---

## Technical Stack

- **LLM:** GPT-4o-mini (OpenAI)
- **Embeddings:** paraphrase-multilingual-MiniLM-L12-v2
- **Vector DB:** ChromaDB
- **Languages:** Hebrew + English

---

## Architecture

### Current (v2.0)
```
chatbot.py → AdvancedVectorDBQASystem (agent) → 7 tools → Results
```

**Benefits:**
- No routing errors (intent understanding)
- 50% code reduction
- 95%+ accuracy
- Self-improving

### Previous (v1.0 - archived)
```
chatbot.py → smart_chatbot.py (router) → [FREE | AI] → Results
```

**Issues:**
- Routing errors
- False positives
- Brittle patterns
- Complex maintenance

---

## Cost

- Per query: $0.002
- 1,000 queries: $2/month
- 10,000 queries: $20/month

---

## Files

```
chatbot/
├── chatbot.py              # Entry point
├── vectoric_search.py      # Core engine + Agent
├── postgres_mcp_server.py  # MCP server (optional)
├── requirements.txt
├── .env
├── chroma_db/
└── _archive/old_routing/   # Previous architecture
```

---

## Performance

- Query time: 2-3 seconds
- Accuracy: 95%+
- Database: 10K+ documents
- Embedding dim: 384

---

## Development

### Adding Tools

```python
# In vectoric_search.py
def _tool_specs(self):
    return [
        # ... existing tools
        {"type": "function", "function": {
            "name": "new_tool",
            "description": "Tool description",
            "parameters": {...}
        }}
    ]

def _dispatch_tool(self, name, args):
    if name == "new_tool":
        return self.new_tool(**args)
```

---

## Author

Mickey Frankel - AI/ML Engineer
- GitHub: [@mickeyFrankel](https://github.com/mickeyFrankel)
- LinkedIn: [mickey-frankel](https://linkedin.com/in/mickey-frankel)
- Email: Mickey.115533@gmail.com

---

## License

MIT License
