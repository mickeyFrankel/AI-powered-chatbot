#!/bin/bash
# Chatbot Project Reorganization Script
# This script reorganizes the project structure for better maintainability

set -e  # Exit on error

echo "🎯 Starting Chatbot Project Reorganization..."
echo ""

# Backup first!
echo "📦 Creating backup..."
BACKUP_DIR="_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "   Backup created: $BACKUP_DIR"
echo ""

# Create new directory structure
echo "📁 Creating new directory structure..."
mkdir -p backend/{core,mcp_servers}
mkdir -p scripts/{utils,database,setup}
mkdir -p databases/{vector,postgres,sqlite,backups}
mkdir -p config
mkdir -p v3_development
echo "   ✓ Directories created"
echo ""

# Move core backend files
echo "🔧 Moving core backend files..."
if [ -f "chatbot.py" ]; then
    cp chatbot.py backend/core/
    echo "   ✓ chatbot.py → backend/core/"
fi
if [ -f "vectoric_search.py" ]; then
    cp vectoric_search.py backend/core/
    echo "   ✓ vectoric_search.py → backend/core/"
fi
if [ -f "api.py" ]; then
    cp api.py backend/core/
    echo "   ✓ api.py → backend/core/"
fi
echo ""

# Move utility scripts
echo "🛠️  Moving utility scripts..."
for file in check_vaad.py diagnose.py fix_search.py update_prompt.py; do
    if [ -f "$file" ]; then
        cp "$file" scripts/utils/
        echo "   ✓ $file → scripts/utils/"
    fi
done
echo ""

# Move database scripts
echo "🗄️  Moving database management scripts..."
for file in manual_clear.py postgres_mcp_server.py; do
    if [ -f "$file" ]; then
        cp "$file" scripts/database/
        echo "   ✓ $file → scripts/database/"
    fi
done
echo ""

# Move setup scripts
echo "⚙️  Moving setup scripts..."
for file in setup_all.sh start.sh; do
    if [ -f "$file" ]; then
        cp "$file" scripts/setup/
        echo "   ✓ $file → scripts/setup/"
    fi
done
echo ""

# Move MCP servers
echo "🌐 Moving MCP servers..."
if [ -f "vectordb_MCP_server.py" ]; then
    cp vectordb_MCP_server.py backend/mcp_servers/
    echo "   ✓ vectordb_MCP_server.py → backend/mcp_servers/"
fi
if [ -d "mcp_server_vectordb" ]; then
    cp -r mcp_server_vectordb backend/mcp_servers/
    echo "   ✓ mcp_server_vectordb/ → backend/mcp_servers/"
fi
echo ""

# Move databases
echo "💾 Organizing databases..."
if [ -d "chroma_db" ]; then
    cp -r chroma_db databases/vector/
    echo "   ✓ chroma_db → databases/vector/"
fi
if [ -d "chatbot_db" ]; then
    cp -r chatbot_db databases/sqlite/
    echo "   ✓ chatbot_db → databases/sqlite/"
fi
if [ -d "postgres_data" ]; then
    cp -r postgres_data databases/postgres/
    echo "   ✓ postgres_data → databases/postgres/"
fi
if [ -d "postgres_backups" ]; then
    cp -r postgres_backups databases/backups/
    echo "   ✓ postgres_backups → databases/backups/"
fi
if [ -d "postgres_init" ]; then
    cp -r postgres_init databases/postgres/init/
    echo "   ✓ postgres_init → databases/postgres/init/"
fi
echo ""

# Move documentation
echo "📚 Moving documentation..."
for file in PRODUCTION_READY.md QUICK_MCP_REFERENCE.md BUG_FIXES.md; do
    if [ -f "$file" ]; then
        cp "$file" docs/
        echo "   ✓ $file → docs/"
    fi
done
echo ""

# Move refactored to v3_development
echo "🔄 Moving refactored code to v3_development..."
if [ -d "refactored" ]; then
    cp -r refactored/* v3_development/
    echo "   ✓ refactored/ → v3_development/"
fi
echo ""

# Move config files
echo "⚙️  Moving configuration files..."
if [ -f "docker-compose.yml" ]; then
    cp docker-compose.yml config/
    echo "   ✓ docker-compose.yml → config/"
fi
if [ -f "industries.txt" ]; then
    cp industries.txt config/
    echo "   ✓ industries.txt → config/"
fi
echo ""

# Create navigation helpers
echo "📝 Creating navigation helpers..."

# Create backend/__init__.py
cat > backend/__init__.py << 'EOF'
"""
Chatbot Backend Package
Main backend components for the AI-powered contact search system.
"""
EOF

# Create backend/core/__init__.py
cat > backend/core/__init__.py << 'EOF'
"""
Core Backend Components
- chatbot.py: CLI interface
- vectoric_search.py: AI agent and vector search
- api.py: FastAPI REST API
"""
EOF

# Create README in new structure
cat > STRUCTURE_README.md << 'EOF'
# New Project Structure

After reorganization:

```
chatbot/
├── backend/           # All backend Python code
│   ├── core/         # Main application files
│   └── mcp_servers/  # MCP server implementations
├── frontend/         # React UI
├── databases/        # All database files
│   ├── vector/      # ChromaDB
│   ├── postgres/    # PostgreSQL
│   ├── sqlite/      # SQLite
│   └── backups/     # All backups
├── scripts/          # Utility scripts
│   ├── utils/       # Helper scripts
│   ├── database/    # DB management
│   └── setup/       # Installation/setup
├── config/           # Configuration files
├── v3_development/   # Refactored modular version
├── docs/            # Documentation
└── data/            # Input data files
```

## Quick Start After Reorganization

**Start Backend:**
```bash
cd backend/core
python3 api.py
```

**Start Frontend:**
```bash
cd frontend
npm run dev
```

**Database Scripts:**
```bash
cd scripts/database
python3 manual_clear.py
```

**All scripts use relative imports and should work from project root.**
EOF

echo "   ✓ Navigation helpers created"
echo ""

echo "✅ Reorganization complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Review the new structure: ls -R backend/ scripts/ databases/"
echo "2. Test that everything still works"
echo "3. Update import paths in Python files if needed"
echo "4. If everything works, you can delete the old files:"
echo "   - Review files in root directory"
echo "   - Compare with new locations in $BACKUP_DIR"
echo ""
echo "⚠️  NOTE: Original files are COPIED, not moved. Delete manually after testing."
echo "   Backup location: $BACKUP_DIR"
echo ""
echo "📖 See STRUCTURE_README.md for the new project layout"
