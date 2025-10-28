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
