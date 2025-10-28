# import os, json
# from mcp.server.fastmcp import FastMCP
# from vectoric_search import AdvancedVectorDBQASystem

# app = FastMCP("vectordb")

# # Instantiate your system
# persist_dir = os.getenv("CHROMA_PATH", "./chroma_db")
# system = AdvancedVectorDBQASystem(persist_directory=persist_dir)

# @app.tool()
# def search(query: str, n_results: int = 5) -> dict:
#     return system.search(query, n_results=n_results)

# @app.tool()
# def list_by_prefix(letter: str, n: int = 999) -> dict:
#     return {"rows": system.first_n_by_prefix(letter, n=n)}

# @app.tool()
# def names_by_length(length: int, limit: int = 200) -> dict:
#     return {"rows": system.names_by_length(length, limit=limit)}

# @app.tool()
# def names_containing(substring: str, limit: int = 200) -> dict:
#     return {"rows": system.names_containing(substring, limit=limit)}

# @app.tool()
# def names_by_prefix_and_length(prefix: str, length: int, limit: int = 200) -> dict:
#     return {"rows": system.names_by_prefix_and_length(prefix, length, limit=limit)}

# @app.tool()
# def list_sources() -> dict:
#     return {"sources": system.list_sources()}

# @app.tool()
# def purge_duplicates() -> dict:
#     system.purge_duplicates()
#     return {"ok": True}

# if __name__ == "__main__":
#     app.run()  # stdio server for Claude Desktop
__version__ = "0.1.0"
