"""
Stdio MCP Client - Connects to MCP server via subprocess.

This is the REAL MCP client that uses stdio transport to communicate
with an MCP server running as a separate process.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class MCPTool:
    """Represents a tool discovered from an MCP server."""
    name: str
    description: str
    input_schema: dict[str, Any]

    def format_for_llm(self) -> str:
        """Format tool info for LLM system prompt."""
        params = self.input_schema.get("properties", {})
        required = self.input_schema.get("required", [])
        
        param_strs = []
        for name, schema in params.items():
            req = "(required)" if name in required else "(optional)"
            desc = schema.get("description", schema.get("type", ""))
            param_strs.append(f"    - {name} {req}: {desc}")
        
        params_doc = "\n".join(param_strs) if param_strs else "    (no parameters)"
        return f"- {self.name}: {self.description}\n  Parameters:\n{params_doc}"


class StdioMCPClient:
    """
    MCP Client that connects to a server via stdio (subprocess).
    
    This is the production-ready approach for connecting to MCP servers.
    The server runs as a separate process, and we communicate via stdin/stdout.
    """
    
    def __init__(self):
        self.session: ClientSession | None = None
        self.tools: dict[str, MCPTool] = {}
        self._read_stream = None
        self._write_stream = None
        self._stdio_context = None
    
    async def connect(
        self,
        command: str = "python3",
        args: list[str] | None = None,
        server_script: str = "run_mcp_server.py"
    ) -> list[MCPTool]:
        """
        Connect to an MCP server via stdio subprocess.
        
        Args:
            command: Python interpreter to use
            args: Arguments to pass (defaults to [server_script])
            server_script: Path to the server script
            
        Returns:
            List of discovered tools
        """
        if args is None:
            args = [server_script]
        
        server_params = StdioServerParameters(
            command=command,
            args=args,
        )
        
        # Start the subprocess and get streams
        self._stdio_context = stdio_client(server_params)
        streams = await self._stdio_context.__aenter__()
        self._read_stream, self._write_stream = streams
        
        # Create and initialize session
        self.session = ClientSession(self._read_stream, self._write_stream)
        await self.session.__aenter__()
        await self.session.initialize()
        
        # Discover tools
        tools_response = await self.session.list_tools()
        discovered = []
        
        for tool in tools_response.tools:
            mcp_tool = MCPTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {}
            )
            self.tools[tool.name] = mcp_tool
            discovered.append(mcp_tool)
        
        return discovered
    
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            Tool result as string
        """
        if not self.session:
            return "Error: Not connected to server"
        
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"
        
        try:
            result = await self.session.call_tool(tool_name, arguments)
            
            # Extract text content from result
            if hasattr(result, 'content') and result.content:
                texts = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        texts.append(content.text)
                return "\n".join(texts)
            return str(result)
        except Exception as e:
            return f"Error calling '{tool_name}': {str(e)}"
    
    def get_tools_list(self) -> list[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
    
    def get_tools_documentation(self) -> str:
        """Get formatted documentation for LLM prompt."""
        return "\n".join(tool.format_for_llm() for tool in self.tools.values())
    
    async def close(self):
        """Close the connection."""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)
        self.session = None
        self.tools.clear()


@asynccontextmanager
async def connect_to_mcp_server(server_script: str = "run_mcp_server.py"):
    """
    Context manager for connecting to MCP server.
    
    Usage:
        async with connect_to_mcp_server() as client:
            result = await client.call_tool("query_products", {"category": "Electronics"})
    """
    client = StdioMCPClient()
    try:
        await client.connect(server_script=server_script)
        yield client
    finally:
        await client.close()


# ============================================================
# SYNCHRONOUS WRAPPER FOR NOTEBOOKS
# ============================================================


class SyncMCPClient:
    """
    Synchronous wrapper for StdioMCPClient.
    Makes it easier to use in notebooks and with the ReAct agent.
    """
    
    def __init__(self, async_client: StdioMCPClient):
        self._client = async_client
        self._loop = asyncio.get_event_loop()
    
    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool synchronously."""
        return self._loop.run_until_complete(
            self._client.call_tool(tool_name, arguments)
        )
    
    def get_tools_list(self) -> list[str]:
        """Get list of tool names."""
        return self._client.get_tools_list()
    
    def get_tools_documentation(self) -> str:
        """Get tool documentation."""
        return self._client.get_tools_documentation()
    
    @property
    def tools(self) -> dict[str, MCPTool]:
        """Access discovered tools."""
        return self._client.tools


# ============================================================
# TEST / DEMO
# ============================================================


async def demo():
    """Demo showing how to connect and use tools."""
    print("Connecting to MCP server...")
    
    async with connect_to_mcp_server("run_mcp_server.py") as client:
        print(f"✅ Connected! Discovered {len(client.tools)} tools:\n")
        
        for tool in client.tools.values():
            print(f"  • {tool.name}: {tool.description}")
        
        print("\n--- Testing query_products ---")
        result = await client.call_tool("query_products", {"category": "Electronics"})
        print(result)
        
        print("\n--- Testing get_analytics ---")
        result = await client.call_tool("get_analytics", {"metric": "revenue"})
        print(result)
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo())
