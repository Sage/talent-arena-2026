"""
MCP Servers for Session 2 Workshop: Giving AI Agents Superpowers

This module implements MCP (Model Context Protocol) servers that expose
real-world tools to AI agents. These servers demonstrate:
- Filesystem operations (read, list, summarize)
- Database queries (structured data access)
- Controlled actions (reports, notifications)

Each server can be run independently or combined.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ============================================================
# SAMPLE DATA - In-memory database for workshop demos
# ============================================================

SAMPLE_PRODUCTS = [
    {"id": 1, "name": "Widget Pro", "category": "Electronics", "price": 299.99, "stock": 150},
    {"id": 2, "name": "Gadget Plus", "category": "Electronics", "price": 199.99, "stock": 75},
    {"id": 3, "name": "Super Tool", "category": "Hardware", "price": 49.99, "stock": 500},
    {"id": 4, "name": "Smart Sensor", "category": "IoT", "price": 89.99, "stock": 200},
    {"id": 5, "name": "Cloud Connect", "category": "Software", "price": 149.99, "stock": 999},
]

SAMPLE_SALES = [
    {"id": 1, "product_id": 1, "quantity": 10, "date": "2026-02-01", "region": "EMEA"},
    {"id": 2, "product_id": 2, "quantity": 25, "date": "2026-02-05", "region": "Americas"},
    {"id": 3, "product_id": 1, "quantity": 5, "date": "2026-02-10", "region": "APAC"},
    {"id": 4, "product_id": 3, "quantity": 100, "date": "2026-02-12", "region": "EMEA"},
    {"id": 5, "product_id": 4, "quantity": 50, "date": "2026-02-15", "region": "Americas"},
    {"id": 6, "product_id": 5, "quantity": 30, "date": "2026-02-18", "region": "EMEA"},
]


def init_demo_database() -> sqlite3.Connection:
    """Initialize an in-memory SQLite database with sample data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            date TEXT NOT NULL,
            region TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    # Insert sample data
    for product in SAMPLE_PRODUCTS:
        cursor.execute(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?)",
            (product["id"], product["name"], product["category"], product["price"], product["stock"])
        )

    for sale in SAMPLE_SALES:
        cursor.execute(
            "INSERT INTO sales VALUES (?, ?, ?, ?, ?)",
            (sale["id"], sale["product_id"], sale["quantity"], sale["date"], sale["region"])
        )

    conn.commit()
    return conn


# ============================================================
# FILESYSTEM MCP SERVER
# ============================================================


class FileSystemMCPServer:
    """
    MCP Server for filesystem operations.
    
    Tools:
    - read_file: Read contents of a file
    - list_directory: List files in a directory
    - get_file_info: Get metadata about a file
    """

    def __init__(self, allowed_paths: list[str] | None = None):
        """
        Initialize the filesystem server.
        
        Args:
            allowed_paths: List of paths the server is allowed to access.
                          If None, defaults to current directory only.
        """
        self.server = Server("filesystem-server")
        self.allowed_paths = allowed_paths or [os.getcwd()]
        self._setup_handlers()

    #def _get_allowed_paths(self, path: str) -> str | None:
        

    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is within allowed directories."""
        abs_path = os.path.abspath(path)
        return any(
            abs_path.startswith(os.path.abspath(allowed))
            for allowed in self.allowed_paths
        )

    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="read_file",
                    description="Read the contents of a file. Returns the text content.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            },
                            "max_lines": {
                                "type": "integer",
                                "description": "Maximum number of lines to read (default: all)",
                                "default": None
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="list_directory",
                    description="List files and folders in a directory.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the directory to list"
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Optional glob pattern to filter files (e.g., '*.py')",
                                "default": "*"
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="get_file_info",
                    description="Get metadata about a file (size, modification time, etc).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file"
                            }
                        },
                        "required": ["path"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            if name == "read_file":
                return await self._read_file(arguments)
            elif name == "list_directory":
                return await self._list_directory(arguments)
            elif name == "get_file_info":
                return await self._get_file_info(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _read_file(self, args: dict) -> list[TextContent]:
        path = args["path"]
        max_lines = args.get("max_lines")

        if not self._is_path_allowed(path):
            return [TextContent(type="text", text=f"Access denied: {path} is outside allowed directories")]

        try:
            with open(path, "r") as f:
                if max_lines:
                    lines = f.readlines()[:max_lines]
                    content = "".join(lines)
                else:
                    content = f.read()
            return [TextContent(type="text", text=content)]
        except FileNotFoundError:
            return [TextContent(type="text", text=f"File not found: {path}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error reading file: {str(e)}")]

    async def _list_directory(self, args: dict) -> list[TextContent]:
        path = args["path"]
        pattern = args.get("pattern", "*")

        if not self._is_path_allowed(path):
            return [TextContent(type="text", text=f"Access denied: {path} is outside allowed directories")]

        try:
            dir_path = Path(path)
            if not dir_path.is_dir():
                return [TextContent(type="text", text=f"Not a directory: {path}")]

            entries = []
            for entry in dir_path.glob(pattern):
                entry_type = "dir" if entry.is_dir() else "file"
                entries.append({"name": entry.name, "type": entry_type})

            return [TextContent(type="text", text=json.dumps(entries, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error listing directory: {str(e)}")]

    async def _get_file_info(self, args: dict) -> list[TextContent]:
        path = args["path"]

        if not self._is_path_allowed(path):
            return [TextContent(type="text", text=f"Access denied: {path} is outside allowed directories")]

        try:
            file_path = Path(path)
            if not file_path.exists():
                return [TextContent(type="text", text=f"File not found: {path}")]

            stat = file_path.stat()
            info = {
                "name": file_path.name,
                "path": str(file_path.absolute()),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_directory": file_path.is_dir(),
                "extension": file_path.suffix if file_path.is_file() else None
            }
            return [TextContent(type="text", text=json.dumps(info, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting file info: {str(e)}")]

    async def run(self):
        """Run the MCP server via stdio."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


# ============================================================
# DATABASE MCP SERVER
# ============================================================


class DatabaseMCPServer:
    """
    MCP Server for database operations.
    
    Tools:
    - query_products: Search and filter products
    - query_sales: Get sales data with filters
    - get_analytics: Get aggregated analytics
    """

    def __init__(self, db_connection: sqlite3.Connection | None = None):
        """
        Initialize the database server.
        
        Args:
            db_connection: SQLite connection. If None, uses demo database.
        """
        self.server = Server("database-server")
        self.conn = db_connection or init_demo_database()
        self._setup_handlers()

    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="query_products",
                    description="Query the products database. Can filter by category, price range, or search by name.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Filter by category (Electronics, Hardware, IoT, Software)"
                            },
                            "min_price": {
                                "type": "number",
                                "description": "Minimum price filter"
                            },
                            "max_price": {
                                "type": "number",
                                "description": "Maximum price filter"
                            },
                            "search": {
                                "type": "string",
                                "description": "Search term for product name"
                            }
                        }
                    }
                ),
                Tool(
                    name="query_sales",
                    description="Query sales data. Can filter by region, date range, or product.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "region": {
                                "type": "string",
                                "description": "Filter by region (EMEA, Americas, APAC)"
                            },
                            "product_id": {
                                "type": "integer",
                                "description": "Filter by product ID"
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date (YYYY-MM-DD)"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date (YYYY-MM-DD)"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_analytics",
                    description="Get aggregated analytics: total revenue, top products, sales by region.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "metric": {
                                "type": "string",
                                "enum": ["revenue", "top_products", "sales_by_region", "inventory_value"],
                                "description": "The analytics metric to retrieve"
                            }
                        },
                        "required": ["metric"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            if name == "query_products":
                return await self._query_products(arguments)
            elif name == "query_sales":
                return await self._query_sales(arguments)
            elif name == "get_analytics":
                return await self._get_analytics(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _query_products(self, args: dict) -> list[TextContent]:
        cursor = self.conn.cursor()
        query = "SELECT * FROM products WHERE 1=1"
        params = []

        if args.get("category"):
            query += " AND category = ?"
            params.append(args["category"])
        if args.get("min_price"):
            query += " AND price >= ?"
            params.append(args["min_price"])
        if args.get("max_price"):
            query += " AND price <= ?"
            params.append(args["max_price"])
        if args.get("search"):
            query += " AND name LIKE ?"
            params.append(f"%{args['search']}%")

        cursor.execute(query, params)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    async def _query_sales(self, args: dict) -> list[TextContent]:
        cursor = self.conn.cursor()
        query = """
            SELECT s.*, p.name as product_name, p.price,
                   (s.quantity * p.price) as total_value
            FROM sales s
            JOIN products p ON s.product_id = p.id
            WHERE 1=1
        """
        params = []

        if args.get("region"):
            query += " AND s.region = ?"
            params.append(args["region"])
        if args.get("product_id"):
            query += " AND s.product_id = ?"
            params.append(args["product_id"])
        if args.get("start_date"):
            query += " AND s.date >= ?"
            params.append(args["start_date"])
        if args.get("end_date"):
            query += " AND s.date <= ?"
            params.append(args["end_date"])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    async def _get_analytics(self, args: dict) -> list[TextContent]:
        cursor = self.conn.cursor()
        metric = args["metric"]

        if metric == "revenue":
            cursor.execute("""
                SELECT SUM(s.quantity * p.price) as total_revenue,
                       COUNT(*) as total_transactions,
                       SUM(s.quantity) as total_units_sold
                FROM sales s
                JOIN products p ON s.product_id = p.id
            """)
            row = cursor.fetchone()
            result = dict(row)

        elif metric == "top_products":
            cursor.execute("""
                SELECT p.name, p.category,
                       SUM(s.quantity) as units_sold,
                       SUM(s.quantity * p.price) as revenue
                FROM sales s
                JOIN products p ON s.product_id = p.id
                GROUP BY p.id
                ORDER BY revenue DESC
                LIMIT 5
            """)
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]

        elif metric == "sales_by_region":
            cursor.execute("""
                SELECT s.region,
                       COUNT(*) as transactions,
                       SUM(s.quantity) as units_sold,
                       SUM(s.quantity * p.price) as revenue
                FROM sales s
                JOIN products p ON s.product_id = p.id
                GROUP BY s.region
                ORDER BY revenue DESC
            """)
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]

        elif metric == "inventory_value":
            cursor.execute("""
                SELECT SUM(price * stock) as total_inventory_value,
                       SUM(stock) as total_units,
                       COUNT(*) as product_count
                FROM products
            """)
            row = cursor.fetchone()
            result = dict(row)

        else:
            result = {"error": f"Unknown metric: {metric}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def run(self):
        """Run the MCP server via stdio."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


# ============================================================
# ACTIONS MCP SERVER
# ============================================================


class ActionsMCPServer:
    """
    MCP Server for controlled actions.
    
    Tools:
    - generate_report: Generate a formatted report
    - send_notification: Send a notification (simulated)
    - create_task: Create a task/todo item (simulated)
    """

    def __init__(self, output_dir: str | None = None):
        """
        Initialize the actions server.
        
        Args:
            output_dir: Directory for generated files. Defaults to ./output
        """
        self.server = Server("actions-server")
        self.output_dir = Path(output_dir or "./output")
        self.output_dir.mkdir(exist_ok=True)
        self.notifications_log: list[dict] = []
        self.tasks_log: list[dict] = []
        self._setup_handlers()

    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="generate_report",
                    description="Generate a formatted markdown report and save it to a file.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Report title"
                            },
                            "content": {
                                "type": "string",
                                "description": "Report content (can include markdown)"
                            },
                            "filename": {
                                "type": "string",
                                "description": "Output filename (without extension)"
                            }
                        },
                        "required": ["title", "content"]
                    }
                ),
                Tool(
                    name="send_notification",
                    description="Send a notification to a channel (simulated - logs the notification).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "channel": {
                                "type": "string",
                                "enum": ["slack", "email", "teams"],
                                "description": "Notification channel"
                            },
                            "recipient": {
                                "type": "string",
                                "description": "Recipient (email or channel name)"
                            },
                            "message": {
                                "type": "string",
                                "description": "Notification message"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "default": "medium"
                            }
                        },
                        "required": ["channel", "recipient", "message"]
                    }
                ),
                Tool(
                    name="create_task",
                    description="Create a task/todo item (simulated - logs the task).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Task title"
                            },
                            "description": {
                                "type": "string",
                                "description": "Task description"
                            },
                            "assignee": {
                                "type": "string",
                                "description": "Person assigned to the task"
                            },
                            "due_date": {
                                "type": "string",
                                "description": "Due date (YYYY-MM-DD)"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"],
                                "default": "medium"
                            }
                        },
                        "required": ["title"]
                    }
                ),
                Tool(
                    name="get_action_log",
                    description="Get the log of all actions taken (notifications sent, tasks created).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "action_type": {
                                "type": "string",
                                "enum": ["notifications", "tasks", "all"],
                                "default": "all"
                            }
                        }
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            if name == "generate_report":
                return await self._generate_report(arguments)
            elif name == "send_notification":
                return await self._send_notification(arguments)
            elif name == "create_task":
                return await self._create_task(arguments)
            elif name == "get_action_log":
                return await self._get_action_log(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _generate_report(self, args: dict) -> list[TextContent]:
        title = args["title"]
        content = args["content"]
        filename = args.get("filename", f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        report = f"""# {title}

Generated: {datetime.now().isoformat()}

---

{content}

---
*Report generated by MCP Actions Server*
"""

        output_path = self.output_dir / f"{filename}.md"
        with open(output_path, "w") as f:
            f.write(report)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "path": str(output_path.absolute()),
                "title": title,
                "size_bytes": len(report)
            }, indent=2)
        )]

    async def _send_notification(self, args: dict) -> list[TextContent]:
        notification = {
            "id": len(self.notifications_log) + 1,
            "timestamp": datetime.now().isoformat(),
            "channel": args["channel"],
            "recipient": args["recipient"],
            "message": args["message"],
            "priority": args.get("priority", "medium"),
            "status": "sent"  # Simulated
        }
        self.notifications_log.append(notification)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "notification_id": notification["id"],
                "message": f"Notification sent to {args['recipient']} via {args['channel']}"
            }, indent=2)
        )]

    async def _create_task(self, args: dict) -> list[TextContent]:
        task = {
            "id": len(self.tasks_log) + 1,
            "created_at": datetime.now().isoformat(),
            "title": args["title"],
            "description": args.get("description", ""),
            "assignee": args.get("assignee", "unassigned"),
            "due_date": args.get("due_date"),
            "priority": args.get("priority", "medium"),
            "status": "open"
        }
        self.tasks_log.append(task)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "task_id": task["id"],
                "message": f"Task '{args['title']}' created successfully"
            }, indent=2)
        )]

    async def _get_action_log(self, args: dict) -> list[TextContent]:
        action_type = args.get("action_type", "all")

        if action_type == "notifications":
            result = {"notifications": self.notifications_log}
        elif action_type == "tasks":
            result = {"tasks": self.tasks_log}
        else:
            result = {
                "notifications": self.notifications_log,
                "tasks": self.tasks_log
            }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def run(self):
        """Run the MCP server via stdio."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


# ============================================================
# COMBINED MCP SERVER (for simplicity in demos)
# ============================================================


class CombinedMCPServer:
    """
    Combined MCP Server with all tools from filesystem, database, and actions.
    This is the main server used in the workshop for simplicity.
    """

    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        db_connection: sqlite3.Connection | None = None,
        output_dir: str | None = None
    ):
        self.server = Server("workshop-mcp-server")
        
        # Initialize sub-components
        self.allowed_paths = allowed_paths or [os.getcwd()]
        self.conn = db_connection or init_demo_database()
        self.output_dir = Path(output_dir or "./output")
        self.output_dir.mkdir(exist_ok=True)
        self.notifications_log: list[dict] = []
        self.tasks_log: list[dict] = []
        
        # Create instances for delegation
        self.fs = FileSystemMCPServer(self.allowed_paths)
        self.db = DatabaseMCPServer(self.conn)
        self.actions = ActionsMCPServer(str(self.output_dir))
        
        self._setup_handlers()

    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            # Combine tools from all servers
            fs_tools = [
                Tool(
                    name="read_file",
                    description="Read the contents of a file. Returns the text content.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to read"},
                            "max_lines": {"type": "integer", "description": "Maximum lines to read"}
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="list_directory",
                    description="List files and folders in a directory.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to directory"},
                            "pattern": {"type": "string", "description": "Glob pattern", "default": "*"}
                        },
                        "required": ["path"]
                    }
                ),
            ]
            
            db_tools = [
                Tool(
                    name="query_products",
                    description="Query products database. Filter by category, price range, or search name.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "min_price": {"type": "number"},
                            "max_price": {"type": "number"},
                            "search": {"type": "string"}
                        }
                    }
                ),
                Tool(
                    name="query_sales",
                    description="Query sales data by region, date range, or product.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "region": {"type": "string"},
                            "product_id": {"type": "integer"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"}
                        }
                    }
                ),
                Tool(
                    name="get_analytics",
                    description="Get analytics: revenue, top_products, sales_by_region, inventory_value.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "metric": {"type": "string", "enum": ["revenue", "top_products", "sales_by_region", "inventory_value"]}
                        },
                        "required": ["metric"]
                    }
                ),
            ]
            
            action_tools = [
                Tool(
                    name="generate_report",
                    description="Generate and save a markdown report.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "filename": {"type": "string"}
                        },
                        "required": ["title", "content"]
                    }
                ),
                Tool(
                    name="send_notification",
                    description="Send notification to slack/email/teams (simulated).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "channel": {"type": "string", "enum": ["slack", "email", "teams"]},
                            "recipient": {"type": "string"},
                            "message": {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                        },
                        "required": ["channel", "recipient", "message"]
                    }
                ),
                Tool(
                    name="create_task",
                    description="Create a task/todo item.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "assignee": {"type": "string"},
                            "due_date": {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                        },
                        "required": ["title"]
                    }
                ),
            ]
            
            return fs_tools + db_tools + action_tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            # Route to appropriate handler
            if name in ["read_file", "list_directory", "get_file_info"]:
                return await self._handle_fs(name, arguments)
            elif name in ["query_products", "query_sales", "get_analytics"]:
                return await self._handle_db(name, arguments)
            elif name in ["generate_report", "send_notification", "create_task", "get_action_log"]:
                return await self._handle_actions(name, arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _handle_fs(self, name: str, args: dict) -> list[TextContent]:
        if name == "read_file":
            return await self.fs._read_file(args)
        elif name == "list_directory":
            return await self.fs._list_directory(args)
        elif name == "get_file_info":
            return await self.fs._get_file_info(args)
        return [TextContent(type="text", text=f"Unknown fs tool: {name}")]

    async def _handle_db(self, name: str, args: dict) -> list[TextContent]:
        if name == "query_products":
            return await self.db._query_products(args)
        elif name == "query_sales":
            return await self.db._query_sales(args)
        elif name == "get_analytics":
            return await self.db._get_analytics(args)
        return [TextContent(type="text", text=f"Unknown db tool: {name}")]

    async def _handle_actions(self, name: str, args: dict) -> list[TextContent]:
        if name == "generate_report":
            return await self.actions._generate_report(args)
        elif name == "send_notification":
            return await self.actions._send_notification(args)
        elif name == "create_task":
            return await self.actions._create_task(args)
        elif name == "get_action_log":
            return await self.actions._get_action_log(args)
        return [TextContent(type="text", text=f"Unknown action tool: {name}")]

    async def run(self):
        """Run the combined MCP server via stdio."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


# ============================================================
# MAIN ENTRY POINTS
# ============================================================


async def run_filesystem_server():
    """Entry point for filesystem MCP server."""
    server = FileSystemMCPServer()
    await server.run()


async def run_database_server():
    """Entry point for database MCP server."""
    server = DatabaseMCPServer()
    await server.run()


async def run_actions_server():
    """Entry point for actions MCP server."""
    server = ActionsMCPServer()
    await server.run()


async def run_combined_server():
    """Entry point for combined MCP server (recommended for workshop)."""
    server = CombinedMCPServer()
    await server.run()


if __name__ == "__main__":
    import asyncio
    import sys

    server_type = sys.argv[1] if len(sys.argv) > 1 else "combined"
    
    if server_type == "filesystem":
        asyncio.run(run_filesystem_server())
    elif server_type == "database":
        asyncio.run(run_database_server())
    elif server_type == "actions":
        asyncio.run(run_actions_server())
    else:
        asyncio.run(run_combined_server())
