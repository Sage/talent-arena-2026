#!/usr/bin/env python3
"""
MCP Server Runner - Starts the MCP server via stdio transport.

Run this in a terminal:
    python3 run_mcp_server.py

The server will communicate via stdin/stdout using the MCP protocol.
Keep this running while using the workshop notebook.
"""

import asyncio
import sys
import os

# Ensure we're in the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import and run the combined MCP server
from mcp_servers import CombinedMCPServer

async def main():
    """Start the MCP server with stdio transport."""
    # Print startup info to stderr (stdout is for MCP protocol)
    print("🚀 Starting MCP Server (stdio transport)...", file=sys.stderr)
    print("📦 Tools: query_products, query_sales, get_analytics, read_file, list_directory, generate_report, etc.", file=sys.stderr)
    print("⏳ Server running. Waiting for client connection...", file=sys.stderr)
    print("   (Press Ctrl+C to stop)", file=sys.stderr)
    
    server = CombinedMCPServer()
    await server.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Server stopped.", file=sys.stderr)
