#!/bin/bash

# Master Installation Script for PostgreSQL + MCP
# Run: chmod +x install.sh && ./install.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}Creating PostgreSQL + MCP Setup in current directory...${NC}"
echo ""

# Create directories
echo -e "${YELLOW}[1/9] Creating directories...${NC}"
mkdir -p postgres_data
mkdir -p postgres_init
mkdir -p postgres_backups
echo -e "${GREEN}✓ Directories created${NC}"

# Create docker-compose.yml
echo -e "${YELLOW}[2/9] Creating Docker configuration...${NC}"
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: chatbot_postgres
    environment:
      POSTGRES_USER: chatbot_user
      POSTGRES_PASSWORD: chatbot_password
      POSTGRES_DB: chatbot_db
    ports:
      - "5432:5432"
    volumes:
      - ./postgres_data:/var/lib/postgresql/data
      - ./postgres_init:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U chatbot_user -d chatbot_db"]
      interval: 10s
      timeout: 5s
      retries: 5
EOF
echo -e "${GREEN}✓ Docker config created${NC}"

# Create init SQL
echo -e "${YELLOW}[3/9] Creating database schema...${NC}"
cat > postgres_init/01-init.sql << 'EOF'
CREATE SCHEMA IF NOT EXISTS chatbot;
GRANT ALL PRIVILEGES ON SCHEMA chatbot TO chatbot_user;

CREATE TABLE IF NOT EXISTS chatbot.contacts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    company VARCHAR(255),
    industry VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chatbot.data_load_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(255),
    file_path TEXT,
    status VARCHAR(50),
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
EOF
echo -e "${GREEN}✓ Schema created${NC}"

# Create data loader
echo -e "${YELLOW}[4/9] Creating data loader...${NC}"
cat > postgres_data_loader.py << 'EOF'
#!/usr/bin/env python3
import psycopg2
import pandas as pd
import json
import sys
from pathlib import Path

def load_file(file_path, table_name='data', schema='chatbot'):
    conn_params = {
        'host': 'localhost',
        'port': 5432,
        'dbname': 'chatbot_db',
        'user': 'chatbot_user',
        'password': 'chatbot_password'
    }
    
    path = Path(file_path)
    if not path.exists():
        print(f"✗ File not found: {file_path}")
        return False
    
    if path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif path.suffix == '.json':
        with open(file_path) as f:
            data = json.load(f)
        df = pd.DataFrame(data if isinstance(data, list) else [data])
    else:
        print(f"✗ Unsupported file type")
        return False
    
    print(f"Read {len(df)} rows from {file_path}")
    
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()
    
    type_map = {'int64': 'INTEGER', 'float64': 'FLOAT', 'object': 'TEXT', 'bool': 'BOOLEAN'}
    columns = [f'"{col}" {type_map.get(str(dtype), "TEXT")}' for col, dtype in df.dtypes.items()]
    
    create_stmt = f"CREATE TABLE IF NOT EXISTS {schema}.{table_name} (id SERIAL PRIMARY KEY, {', '.join(columns)});"
    cursor.execute(create_stmt)
    
    cols = ', '.join([f'"{col}"' for col in df.columns])
    placeholders = ', '.join(['%s'] * len(df.columns))
    insert_stmt = f'INSERT INTO {schema}.{table_name} ({cols}) VALUES ({placeholders})'
    
    for _, row in df.iterrows():
        cursor.execute(insert_stmt, tuple(row))
    
    conn.commit()
    print(f"✓ Loaded {len(df)} rows into {schema}.{table_name}")
    
    cursor.close()
    conn.close()
    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--table', default=None)
    args = parser.parse_args()
    
    table_name = args.table or Path(args.file).stem.lower().replace(' ', '_')
    load_file(args.file, table_name)
EOF
chmod +x postgres_data_loader.py
echo -e "${GREEN}✓ Data loader created${NC}"

# Create MCP server for PostgreSQL
echo -e "${YELLOW}[5/9] Creating PostgreSQL MCP server...${NC}"
cat > postgres_mcp_server.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

conn_params = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'chatbot_db',
    'user': 'chatbot_user',
    'password': 'chatbot_password'
}

server = Server("postgresql-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_database",
            description="Execute SQL SELECT query on PostgreSQL",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        ),
        Tool(
            name="list_tables",
            description="List all tables in the database",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="describe_table",
            description="Show table structure and sample data",
            inputSchema={
                "type": "object",
                "properties": {"table_name": {"type": "string"}},
                "required": ["table_name"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        if name == "query_database":
            cursor.execute(arguments["query"])
            results = [dict(row) for row in cursor.fetchall()]
            return [TextContent(type="text", text=json.dumps({"success": True, "data": results}, default=str, indent=2))]
        
        elif name == "list_tables":
            cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname='chatbot' ORDER BY tablename;")
            tables = [dict(row) for row in cursor.fetchall()]
            return [TextContent(type="text", text=json.dumps({"success": True, "tables": tables}, indent=2))]
        
        elif name == "describe_table":
            table = arguments["table_name"]
            cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema='chatbot' AND table_name='{table}'
                ORDER BY ordinal_position;
            """)
            columns = [dict(row) for row in cursor.fetchall()]
            
            cursor.execute(f"SELECT * FROM chatbot.{table} LIMIT 5;")
            sample = [dict(row) for row in cursor.fetchall()]
            
            return [TextContent(type="text", text=json.dumps({
                "success": True, 
                "columns": columns, 
                "sample_data": sample
            }, default=str, indent=2))]
    
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}, indent=2))]
    finally:
        cursor.close()
        conn.close()

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
EOF
chmod +x postgres_mcp_server.py
echo -e "${GREEN}✓ PostgreSQL MCP server created${NC}"

# Update requirements
echo -e "${YELLOW}[6/9] Adding PostgreSQL dependencies...${NC}"
if ! grep -q "psycopg2-binary" requirements.txt 2>/dev/null; then
    echo "psycopg2-binary==2.9.9" >> requirements.txt
fi
echo -e "${GREEN}✓ Dependencies added${NC}"

# Install dependencies
echo -e "${YELLOW}[7/9] Installing dependencies...${NC}"
pip3 install psycopg2-binary --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Start Docker (try both docker compose and docker-compose)
echo -e "${YELLOW}[8/9] Starting PostgreSQL...${NC}"
if command -v docker &> /dev/null; then
    docker compose up -d 2>/dev/null || docker-compose up -d 2>/dev/null || {
        echo -e "${YELLOW}⚠ Please install Docker Desktop from https://www.docker.com/products/docker-desktop${NC}"
        exit 1
    }
    sleep 5

    for i in {1..20}; do
        if docker exec chatbot_postgres pg_isready -U chatbot_user -d chatbot_db &>/dev/null; then
            echo -e "${GREEN}✓ PostgreSQL ready!${NC}"
            break
        fi
        echo -n "."
        sleep 1
    done
    echo ""
else
    echo -e "${YELLOW}⚠ Docker not found. Please install Docker Desktop.${NC}"
    exit 1
fi

# Load existing contacts.csv if it exists
echo -e "${YELLOW}[9/9] Loading existing data...${NC}"
if [ -f "contacts.csv" ]; then
    python3 postgres_data_loader.py contacts.csv --table contacts
fi
echo ""

MCP_PATH="$(pwd)/postgres_mcp_server.py"

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}         PostgreSQL + MCP Setup Complete!              ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}Add 'postgresql' to your Claude Desktop config:${NC}"
echo ""
echo -e "${YELLOW}Edit: ~/Library/Application Support/Claude/claude_desktop_config.json${NC}"
echo ""
cat << MCPEOF
{
  "mcpServers": {
    "vectordb": {
      "command": "python3",
      "args": ["$(pwd)/vectordb_MCP_server.py"]
    },
    "postgresql": {
      "command": "python3",
      "args": ["$MCP_PATH"]
    }
  }
}
MCPEOF
echo ""
echo -e "${YELLOW}Then restart Claude Desktop (Cmd+Q and reopen)${NC}"
echo ""
echo -e "${CYAN}Quick commands:${NC}"
echo "  docker compose up -d     # Start PostgreSQL"
echo "  docker compose stop      # Stop PostgreSQL"
echo "  python3 postgres_data_loader.py <file>  # Load data"
echo ""
