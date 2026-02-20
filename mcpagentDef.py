from lab import FreeFlowLLM, ReActEngine
from stdio_mcp_client import SyncMCPClient
import json
class MCPAgent:
    """
    AI Agent that uses MCP tools via stdio transport.
    Shows step-by-step thinking (Thought → Action → Observation → Answer).
    """
    
    def __init__(self, mcp_client: SyncMCPClient):
        self.client = mcp_client
        self.llm = FreeFlowLLM()
        
        # Create tools wrapper for ReAct engine
        self.tools = self._create_tools_wrapper()
        self.engine = ReActEngine(self.llm, self.tools, max_iterations=10)
    
    def _create_tools_wrapper(self):
        """Create tools object that wraps MCP client calls."""
        client = self.client
        
        class ToolsWrapper:
            """Wrapper exposing MCP tools as simple methods."""
            
            def query_products(self, args_str):
                """Query products database. Input: JSON with category, min_price, max_price, or search."""
                args = self._parse_args(args_str)
                return client.call_tool("query_products", args)
            
            def query_sales(self, args_str):
                """Query sales data. Input: JSON with region, product_id, start_date, end_date."""
                args = self._parse_args(args_str)
                return client.call_tool("query_sales", args)
            
            def get_analytics(self, args_str):
                """Get analytics metrics. Input: JSON with metric (revenue, top_products, sales_by_region, inventory_value)."""
                args = self._parse_args(args_str)
                return client.call_tool("get_analytics", args)

            def read_file(self, args_str):
                """Read file contents. Input: JSON with path."""
                args = self._parse_args(args_str)
                return client.call_tool("read_file", args)
            
            def list_directory(self, args_str):
                """List directory contents. Input: JSON with path."""
                args = self._parse_args(args_str)
                return client.call_tool("list_directory", args)
            
            def generate_report(self, args_str):
                """Generate markdown report. Input: JSON with title and content."""
                args = self._parse_args(args_str)
                return client.call_tool("generate_report", args)
            
            def send_notification(self, args_str):
                """Send notification. Input: JSON with channel, recipient, message."""
                args = self._parse_args(args_str)
                return client.call_tool("send_notification", args)
            
            def create_task(self, args_str):
                """Create task. Input: JSON with title, description, priority."""
                args = self._parse_args(args_str)
                return client.call_tool("create_task", args)

            def _parse_args(self, args_str):
                """Parse string arguments into dict."""
                if isinstance(args_str, dict):
                    return args_str
                if isinstance(args_str, str):
                    args_str = args_str.strip()
                    if args_str.startswith("{"):
                        try:
                            return json.loads(args_str)
                        except json.JSONDecodeError:
                            pass
                    # Handle simple string inputs
                    return {"query": args_str}
                return {}
            
            def get_tools_documentation(self):
                """Return documentation for all tools."""
                return client.get_tools_documentation()
        
        return ToolsWrapper()
    
    def run(self, question, verbose=True):
        """
        Run the agent on a question.
        Shows step-by-step thinking if verbose=True.
        """
        system_prompt = """You are an AI assistant with access to MCP tools.

When you need to use a tool, respond with this EXACT format:

Thought: [your reasoning about what to do]
Action: [tool name - one of: query_products, query_sales, get_analytics, read_file, list_directory, generate_report, send_notification, create_task]
Action Input: [JSON object with the parameters]

After receiving the Observation (tool result), either:
- Use another tool if needed
- OR provide your Final Answer:

Final Answer: [your answer to the user]

IMPORTANT:
- Always use valid JSON for Action Input
- Wait for Observation before continuing
- When you have enough information, give a Final Answer

Available tools and their parameters:
""" + self.client.get_tools_documentation()

        print("🤖 Agent starting...\n")
        print("=" * 70)
        
        for step in self.engine.run(system_prompt, question, stream_final=False):
            if verbose:
                print(step)
        
        print("=" * 70)
        print("\n✅ Agent finished!")

