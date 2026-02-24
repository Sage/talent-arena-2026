#!/usr/bin/env python3
"""
MCP Server Runner - Starts the MCP server via stdio transport.

Run this in a terminal:
    python3 run_mcp_server.py
    
Or with specific tool packs:
    python3 run_mcp_server.py --packs filesystem,database,actions

The server will communicate via stdin/stdout using the MCP protocol.
"""

import argparse
import asyncio
import sys
import os

# Ensure we're in the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import the MCP server
# from mcp_servers import CombinedMCPServer
from mcp_servers_refactored import CombinedMCPServer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MCP Server Runner")
    parser.add_argument(
        "--packs",
        type=str,
        default="filesystem,database,actions,aggregator,grapher",
        help="Comma-separated list of tool packs: filesystem, database, actions, aggregator, grapher"
    )
    return parser.parse_args()


async def main(tool_packs: list[str]):
    """Start the MCP server with stdio transport."""
    # Print startup info to stderr (stdout is for MCP protocol)
    print("🚀 Starting MCP Server (stdio transport)...", file=sys.stderr)
    print(f"📦 Tool packs: {', '.join(tool_packs)}", file=sys.stderr)
    print("⏳ Server running. Waiting for client connection...", file=sys.stderr)
    print("   (Press Ctrl+C to stop)", file=sys.stderr)
    
    # Configure which tool packs to enable
    enable_filesystem = "filesystem" in tool_packs
    enable_database = "database" in tool_packs
    enable_actions = "actions" in tool_packs
    enable_aggregator = "aggregator" in tool_packs
    enable_grapher = "grapher" in tool_packs
    
    server = CombinedMCPServer(
        enable_filesystem=enable_filesystem,
        enable_database=enable_database,
        enable_actions=enable_actions,
        enable_aggregator=enable_aggregator,
        enable_grapher=enable_grapher
    )
    await server.run()

if __name__ == "__main__":
    args = parse_args()
    tool_packs = [p.strip() for p in args.packs.split(",")]
    
    try:
        asyncio.run(main(tool_packs))
    except KeyboardInterrupt:
        print("\n👋 Server stopped.", file=sys.stderr)
