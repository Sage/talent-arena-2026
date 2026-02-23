"""
Stdio MCP Client - Connects to MCP server via subprocess.

This is the REAL MCP client that uses stdio transport to communicate
with an MCP server running as a separate process.

Key features:
- Subprocess management (server runs as child process)
- Connection health checking
- Graceful shutdown
- Error handling with retries
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Setup logging
logger = logging.getLogger(__name__)


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
    
    Features:
    - Automatic subprocess lifecycle management
    - Health checking via ping
    - Graceful error handling
    - Connection state tracking
    """
    
    def __init__(self):
        self.session: ClientSession | None = None
        self.tools: dict[str, MCPTool] = {}
        self._read_stream = None
        self._write_stream = None
        self._stdio_context = None
        self._connected = False
        self._server_params = None
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected to server."""
        return self._connected and self.session is not None
    
    async def connect(
        self,
        command: str = "python3",
        args: list[str] | None = None,
        server_script: str = "run_mcp_server.py",
        timeout: float = 30.0
    ) -> list[MCPTool]:
        """
        Connect to an MCP server via stdio subprocess.
        
        Args:
            command: Python interpreter to use
            args: Arguments to pass (defaults to [server_script])
            server_script: Path to the server script
            timeout: Connection timeout in seconds
            
        Returns:
            List of discovered tools
            
        Raises:
            ConnectionError: If connection fails
            TimeoutError: If connection times out
        """
        if self._connected:
            logger.warning("Already connected, closing existing connection first")
            await self.close()
        
        if args is None:
            args = [server_script]
        
        self._server_params = StdioServerParameters(
            command=command,
            args=args,
        )
        
        try:
            # Start the subprocess and get streams with timeout
            self._stdio_context = stdio_client(self._server_params)
            streams = await asyncio.wait_for(
                self._stdio_context.__aenter__(),
                timeout=timeout
            )
            self._read_stream, self._write_stream = streams
            
            # Create and initialize session
            self.session = ClientSession(self._read_stream, self._write_stream)
            await self.session.__aenter__()
            await asyncio.wait_for(self.session.initialize(), timeout=timeout)
            
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
            
            self._connected = True
            logger.info(f"Connected to MCP server, discovered {len(discovered)} tools")
            return discovered
            
        except asyncio.TimeoutError:
            await self._cleanup_on_error()
            raise TimeoutError(f"Connection timed out after {timeout}s. Is the server script correct?")
        except Exception as e:
            await self._cleanup_on_error()
            raise ConnectionError(f"Failed to connect to MCP server: {e}") from e
    
    async def _cleanup_on_error(self):
        """Clean up resources after a connection error."""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
        except:
            pass
        try:
            if self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)
        except:
            pass
        self.session = None
        self._stdio_context = None
        self._connected = False
    
    async def call_tool(self, tool_name: str, arguments: dict[str, Any], timeout: float = 60.0) -> str:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            timeout: Timeout in seconds for the tool call
            
        Returns:
            Tool result as string
        """
        if not self.is_connected:
            return "Error: Not connected to server. Call connect() first."
        
        if tool_name not in self.tools:
            available = ", ".join(self.tools.keys())
            return f"Error: Tool '{tool_name}' not found. Available tools: {available}"
        
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(tool_name, arguments),
                timeout=timeout
            )
            
            # Extract text content from result
            if hasattr(result, 'content') and result.content:
                texts = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        texts.append(content.text)
                return "\n".join(texts)
            return str(result)
        except asyncio.TimeoutError:
            return f"Error: Tool '{tool_name}' timed out after {timeout}s"
        except Exception as e:
            logger.exception(f"Error calling tool '{tool_name}'")
            return f"Error calling '{tool_name}': {str(e)}"
    
    def get_tools_list(self) -> list[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
    
    def get_tools_documentation(self) -> str:
        """Get formatted documentation for LLM prompt."""
        return "\n".join(tool.format_for_llm() for tool in self.tools.values())
    
    async def close(self):
        """Close the connection and cleanup resources."""
        logger.info("Closing MCP client connection")
        self._connected = False
        
        errors = []
        
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                errors.append(f"Session cleanup: {e}")
            self.session = None
        
        if self._stdio_context:
            try:
                await self._stdio_context.__aexit__(None, None, None)
            except Exception as e:
                errors.append(f"Stdio cleanup: {e}")
            self._stdio_context = None
        
        self.tools.clear()
        
        if errors:
            logger.warning(f"Cleanup completed with warnings: {errors}")
        else:
            logger.info("MCP client closed successfully")
    
    async def ping(self, timeout: float = 5.0) -> bool:
        """
        Check if the server is still responsive.
        
        Returns:
            True if server responds, False otherwise
        """
        if not self.is_connected:
            return False
        
        try:
            # Try listing tools as a health check
            await asyncio.wait_for(self.session.list_tools(), timeout=timeout)
            return True
        except:
            return False


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
    
    Note: This uses nest_asyncio in notebooks to handle nested event loops.
    """
    
    def __init__(self, async_client: StdioMCPClient):
        self._client = async_client
    
    def _run_async(self, coro):
        """Run an async coroutine in the current event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # If we're in a running loop (notebook), use nest_asyncio pattern
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.ensure_future(coro)
        
        return loop.run_until_complete(coro)
    
    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool synchronously."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
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
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._client.is_connected


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
