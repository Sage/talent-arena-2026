"""
MCP Servers for Session 2 Workshop: Giving AI Agents Superpowers

REFACTORED ARCHITECTURE:
- Implementation classes contain business logic without MCP dependencies
- Tool definitions are created by helper functions (defined once)
- MCP servers are thin wrappers that delegate to implementations
- CombinedMCPServer reuses implementations without code duplication

This eliminates ~800+ lines of redundant code while maintaining full MCP functionality.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

# Matplotlib setup for non-interactive chart generation
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving charts
import matplotlib.pyplot as plt
import seaborn as sns

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
    {"id": 1, "product_id": 1, "quantity": 33, "date": "2026-02-01", "region": "EMEA"},
    {"id": 2, "product_id": 2, "quantity": 59, "date": "2026-02-05", "region": "Americas"},
    {"id": 3, "product_id": 1, "quantity": 6, "date": "2026-02-10", "region": "APAC"},
    {"id": 4, "product_id": 3, "quantity": 114, "date": "2026-02-12", "region": "EMEA"},
    {"id": 5, "product_id": 4, "quantity": 23, "date": "2026-02-15", "region": "Americas"},
    {"id": 6, "product_id": 5, "quantity": 57, "date": "2026-02-18", "region": "EMEA"},
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
# IMPLEMENTATION CLASSES (No MCP dependencies)
# ============================================================


class FileSystemManager:
    """Pure filesystem operations without MCP coupling."""
    
    def __init__(self, allowed_paths: list[str]):
        self.allowed_paths = [Path(p).resolve() for p in allowed_paths]
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is within allowed directories."""
        resolved = Path(path).resolve()
        return any(resolved.is_relative_to(allowed) for allowed in self.allowed_paths)
    
    def read_file(self, path: str, max_lines: int | None = None) -> dict:
        """Read file contents."""
        if not self._is_path_allowed(path):
            return {"error": f"Access denied: {path} is outside allowed directories"}
        
        try:
            with open(path, "r") as f:
                if max_lines:
                    lines = f.readlines()[:max_lines]
                    content = "".join(lines)
                else:
                    content = f.read()
            return {"content": content}
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}
        except Exception as e:
            return {"error": f"Error reading file: {str(e)}"}
    
    def list_directory(self, path: str, pattern: str = "*") -> dict:
        """List directory contents."""
        if not self._is_path_allowed(path):
            return {"error": f"Access denied: {path} is outside allowed directories"}
        
        try:
            dir_path = Path(path)
            if not dir_path.is_dir():
                return {"error": f"Not a directory: {path}"}
            
            entries = []
            for entry in dir_path.glob(pattern):
                entry_type = "dir" if entry.is_dir() else "file"
                entries.append({"name": entry.name, "type": entry_type})
            
            return {"entries": entries}
        except Exception as e:
            return {"error": f"Error listing directory: {str(e)}"}
    
    def get_file_info(self, path: str) -> dict:
        """Get file metadata."""
        if not self._is_path_allowed(path):
            return {"error": f"Access denied: {path} is outside allowed directories"}
        
        try:
            file_path = Path(path)
            if not file_path.exists():
                return {"error": f"File not found: {path}"}
            
            stat = file_path.stat()
            return {
                "name": file_path.name,
                "path": str(file_path.absolute()),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_directory": file_path.is_dir(),
                "extension": file_path.suffix if file_path.is_file() else None
            }
        except Exception as e:
            return {"error": f"Error getting file info: {str(e)}"}


class DatabaseManager:
    """Pure database operations without MCP coupling."""
    
    def __init__(self, db_connection: sqlite3.Connection):
        self.conn = db_connection
    
    def query_products(self, category: str | None = None, min_price: float | None = None,
                      max_price: float | None = None, search: str | None = None) -> list[dict]:
        """Query products with filters."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM products WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        if min_price is not None:
            query += " AND price >= ?"
            params.append(min_price)
        if max_price is not None:
            query += " AND price <= ?"
            params.append(max_price)
        if search:
            query += " AND name LIKE ?"
            params.append(f"%{search}%")
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def query_sales(self, region: str | None = None, product_id: int | None = None,
                   start_date: str | None = None, end_date: str | None = None) -> list[dict]:
        """Query sales with filters."""
        cursor = self.conn.cursor()
        query = """
            SELECT s.*, p.name as product_name, p.price,
                   (s.quantity * p.price) as total_value
            FROM sales s
            JOIN products p ON s.product_id = p.id
            WHERE 1=1
        """
        params = []
        
        if region:
            query += " AND s.region = ?"
            params.append(region)
        if product_id is not None:
            query += " AND s.product_id = ?"
            params.append(product_id)
        if start_date:
            query += " AND s.date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND s.date <= ?"
            params.append(end_date)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_analytics(self, metric: str) -> dict | list[dict]:
        """Get analytics metrics."""
        cursor = self.conn.cursor()
        
        if metric == "revenue":
            cursor.execute("""
                SELECT SUM(s.quantity * p.price) as total_revenue,
                       COUNT(*) as total_transactions,
                       SUM(s.quantity) as total_units_sold
                FROM sales s
                JOIN products p ON s.product_id = p.id
            """)
            row = cursor.fetchone()
            return dict(row)
        
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
            return [dict(row) for row in rows]
        
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
            return [dict(row) for row in rows]
        
        elif metric == "inventory_value":
            cursor.execute("""
                SELECT SUM(price * stock) as total_inventory_value,
                       SUM(stock) as total_units,
                       COUNT(*) as product_count
                FROM products
            """)
            row = cursor.fetchone()
            return dict(row)
        
        else:
            return {"error": f"Unknown metric: {metric}"}


class ActionsManager:
    """Pure actions operations without MCP coupling."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.notifications_log: list[dict] = []
        self.tasks_log: list[dict] = []
    
    def generate_report(self, title: str, content: str, filename: str | None = None) -> dict:
        """Generate a markdown report."""
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
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
        
        return {
            "status": "success",
            "path": str(output_path.absolute()),
            "title": title,
            "size_bytes": len(report)
        }
    
    def send_notification(self, channel: str, recipient: str, message: str,
                         priority: str = "medium") -> dict:
        """Send a notification (simulated)."""
        notification = {
            "id": len(self.notifications_log) + 1,
            "timestamp": datetime.now().isoformat(),
            "channel": channel,
            "recipient": recipient,
            "message": message,
            "priority": priority,
            "status": "sent"
        }
        self.notifications_log.append(notification)
        
        return {
            "status": "success",
            "notification_id": notification["id"],
            "message": f"Notification sent to {recipient} via {channel}"
        }
    
    def create_task(self, title: str, description: str | None = None,
                   assignee: str | None = None, due_date: str | None = None,
                   priority: str = "medium") -> dict:
        """Create a task (simulated)."""
        task = {
            "id": len(self.tasks_log) + 1,
            "created_at": datetime.now().isoformat(),
            "title": title,
            "description": description or "",
            "assignee": assignee or "unassigned",
            "due_date": due_date,
            "priority": priority,
            "status": "open"
        }
        self.tasks_log.append(task)
        
        return {
            "status": "success",
            "task_id": task["id"],
            "message": f"Task '{title}' created successfully"
        }
    
    def get_action_log(self, action_type: str = "all") -> dict:
        """Get action logs."""
        if action_type == "notifications":
            return {"notifications": self.notifications_log}
        elif action_type == "tasks":
            return {"tasks": self.tasks_log}
        else:
            return {
                "notifications": self.notifications_log,
                "tasks": self.tasks_log
            }


class AggregatorManager:
    """Pure aggregation operations without MCP coupling."""
    
    def aggregate_for_chart(self, data: list[dict], group_by: str,
                           value_field: str, aggregation: str = "sum") -> dict:
        """Aggregate raw data into chart-ready format."""
        if not isinstance(data, list) or len(data) == 0:
            return {"error": "Input must be a non-empty list of records"}
        
        # Validate fields exist
        first_record = data[0]
        if group_by not in first_record:
            available_fields = list(first_record.keys())
            return {"error": f"Field '{group_by}' not found. Available: {available_fields}"}
        if value_field not in first_record:
            available_fields = list(first_record.keys())
            return {"error": f"Field '{value_field}' not found. Available: {available_fields}"}
        
        # Group and aggregate
        groups = {}
        for record in data:
            key = record[group_by]
            value = float(record[value_field]) if record[value_field] is not None else 0
            
            if key not in groups:
                groups[key] = []
            groups[key].append(value)
        
        # Apply aggregation
        result = {}
        for key, values in groups.items():
            if aggregation == "sum":
                result[key] = sum(values)
            elif aggregation == "count":
                result[key] = len(values)
            elif aggregation == "average":
                result[key] = sum(values) / len(values) if values else 0
            elif aggregation == "max":
                result[key] = max(values)
            elif aggregation == "min":
                result[key] = min(values)
            else:
                result[key] = sum(values)
        
        # Format for charting
        return {
            "labels": list(result.keys()),
            "values": list(result.values()),
            "group_by": group_by,
            "value_field": value_field,
            "aggregation": aggregation,
            "chart_ready": True
        }


class GrapherManager:
    """Pure charting operations without MCP coupling."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def create_chart(self, chart_data: dict, chart_type: str = "bar",
                    title: str | None = None, filename: str | None = None) -> dict:
        """Create a visual chart from aggregated data."""
        # Validate chart-ready format
        if not chart_data.get("chart_ready"):
            return {
                "error": "Data is not chart-ready! Use aggregate_for_chart first.",
                "hint": "You must call aggregate_for_chart and use its output as input to create_chart. Manual formatting will not work."
            }
        
        if "labels" not in chart_data or "values" not in chart_data:
            return {
                "error": "Invalid chart data format. Must have 'labels' and 'values' arrays.",
                "hint": "You must call aggregate_for_chart and use its output as input to create_chart. Manual formatting will not work."
            }
        
        labels = chart_data["labels"]
        values = chart_data["values"]
        
        if len(labels) == 0 or len(values) == 0:
            return {"error": "No data to chart"}
        
        # Generate title if not provided
        if title is None:
            group_by = chart_data.get("group_by", "Category")
            value_field = chart_data.get("value_field", "Value")
            aggregation = chart_data.get("aggregation", "sum")
            title = f"{value_field.replace('_', ' ').title()} by {group_by.title()} ({aggregation})"
        
        # Generate filename if not provided
        if filename is None:
            filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set seaborn style
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        
        # Create chart based on type
        palette = sns.color_palette("husl", len(labels))
        
        if chart_type == "bar":
            bars = plt.bar(labels, values, color=palette)
            plt.xlabel(chart_data.get("group_by", "Category"))
            plt.ylabel(chart_data.get("value_field", "Value"))
            # Add value labels on bars
            for bar, val in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:,.0f}', ha='center', va='bottom', fontsize=9)
        
        elif chart_type == "horizontal_bar":
            bars = plt.barh(labels, values, color=palette)
            plt.xlabel(chart_data.get("value_field", "Value"))
            plt.ylabel(chart_data.get("group_by", "Category"))
        
        elif chart_type == "pie":
            plt.pie(values, labels=labels, autopct='%1.1f%%', colors=palette)
        
        elif chart_type == "line":
            plt.plot(labels, values, marker='o', linewidth=2, markersize=8, color=palette[0])
            plt.xlabel(chart_data.get("group_by", "Category"))
            plt.ylabel(chart_data.get("value_field", "Value"))
            plt.fill_between(labels, values, alpha=0.3, color=palette[0])
        
        else:
            return {"error": f"Unknown chart type: {chart_type}. Use 'bar', 'pie', 'line', or 'horizontal_bar'"}
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save to output directory
        output_path = self.output_dir / f"{filename}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            "status": "success",
            "path": str(output_path.absolute()),
            "title": title,
            "chart_type": chart_type,
            "data_points": len(labels),
            "message": f"Chart created: {filename}.png"
        }


# ============================================================
# TOOL DEFINITION HELPERS (Define schemas once)
# ============================================================


def get_filesystem_tools() -> list[Tool]:
    """Get filesystem tool definitions."""
    return [
        Tool(
            name="read_file",
            description="Read the contents of a file. IMPORTANT: Only files inside the 'files/' directory are accessible.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file (e.g., 'files/q4_report.txt'). Must start with 'files/'."},
                    "max_lines": {"type": "integer", "description": "Maximum lines to read"}
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="list_directory",
            description="List files in a directory. IMPORTANT: Only the 'files/' directory is accessible.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path. Use 'files/' to list available files."},
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., '*.txt')", "default": "*"}
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="get_file_info",
            description="Get metadata about a file. IMPORTANT: Only files inside 'files/' directory are accessible.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file (e.g., 'files/report.txt')"}
                },
                "required": ["path"]
            }
        ),
    ]


def get_database_tools() -> list[Tool]:
    """Get database tool definitions."""
    return [
        Tool(
            name="query_products",
            description="Query the products table. Products have: id (1-5), name, category, price, stock.",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Filter by category", "enum": ["Electronics", "Hardware", "IoT", "Software"]},
                    "min_price": {"type": "number", "description": "Minimum price"},
                    "max_price": {"type": "number", "description": "Maximum price"},
                    "search": {"type": "string", "description": "Search term for product name"}
                }
            }
        ),
        Tool(
            name="query_sales",
            description="Query the sales table. Sales have: id, product_id, quantity, date, region.",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Filter by region", "enum": ["EMEA", "Americas", "APAC"]},
                    "product_id": {"type": "integer", "description": "Product ID (1-5)"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
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


def get_actions_tools() -> list[Tool]:
    """Get actions tool definitions."""
    return [
        Tool(
            name="generate_report",
            description="Generate and save a markdown report.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Report title"},
                    "content": {"type": "string", "description": "Report content (can include markdown)"},
                    "filename": {"type": "string", "description": "Output filename (without extension)"}
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
                    "recipient": {"type": "string", "description": "Recipient (email or channel name)"},
                    "message": {"type": "string", "description": "Notification message"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"}
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
                    "title": {"type": "string", "description": "Task title"},
                    "description": {"type": "string", "description": "Task description"},
                    "assignee": {"type": "string", "description": "Person assigned"},
                    "due_date": {"type": "string", "description": "Due date (YYYY-MM-DD)"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"], "default": "medium"}
                },
                "required": ["title"]
            }
        ),
        Tool(
            name="get_action_log",
            description="Get the log of all actions taken.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "enum": ["notifications", "tasks", "all"], "default": "all"}
                }
            }
        ),
    ]


def get_aggregator_tools() -> list[Tool]:
    """Get aggregator tool definitions."""
    return [
        Tool(
            name="aggregate_for_chart",
            description="Aggregate raw data into chart-ready format. Required before creating charts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_json": {"type": "string", "description": "JSON string of raw data"},
                    "group_by": {"type": "string", "description": "Field to group by (e.g., 'region', 'category')"},
                    "value_field": {"type": "string", "description": "Field to aggregate (e.g., 'total_value', 'quantity')"},
                    "aggregation": {"type": "string", "enum": ["sum", "count", "average", "max", "min"], "default": "sum"}
                },
                "required": ["data_json", "group_by", "value_field"]
            }
        ),
    ]


def get_grapher_tools() -> list[Tool]:
    """Get grapher tool definitions."""
    return [
        Tool(
            name="create_chart",
            description="Create a visual chart from aggregated data. IMPORTANT: Input must be from aggregate_for_chart.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chart_data_json": {"type": "string", "description": "JSON from aggregate_for_chart"},
                    "chart_type": {"type": "string", "enum": ["bar", "pie", "line", "horizontal_bar"], "default": "bar"},
                    "title": {"type": "string", "description": "Chart title"},
                    "filename": {"type": "string", "description": "Output filename without extension"}
                },
                "required": ["chart_data_json"]
            }
        ),
    ]

# ============================================================
# COMBINED MCP SERVER 
# ============================================================


class CombinedMCPServer:
    """
    Combined MCP Server with all tools.
    
    CLEAN ARCHITECTURE:
    - Reuses implementation managers (no code duplication)
    - Reuses tool definitions (no schema duplication)
    - Individual servers can still run independently
    """
    
    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        db_connection: sqlite3.Connection | None = None,
        output_dir: str | None = None,
        enable_filesystem: bool = True,
        enable_database: bool = True,
        enable_actions: bool = True,
        enable_aggregator: bool = True,
        enable_grapher: bool = True
    ):
        self.server = Server("workshop-mcp-server")
        
        # Initialize managers (not full MCP servers)
        if allowed_paths is None:
            allowed_paths = [str(Path("files").resolve())]
        output_path = Path(output_dir or "./output")
        
        self.fs_manager = FileSystemManager(allowed_paths) if enable_filesystem else None
        self.db_manager = DatabaseManager(db_connection or init_demo_database()) if enable_database else None
        self.actions_manager = ActionsManager(output_path) if enable_actions else None
        self.aggregator_manager = AggregatorManager() if enable_aggregator else None
        self.grapher_manager = GrapherManager(output_path) if enable_grapher else None
        
        # Track enabled packs
        self.enable_filesystem = enable_filesystem
        self.enable_database = enable_database
        self.enable_actions = enable_actions
        self.enable_aggregator = enable_aggregator
        self.enable_grapher = enable_grapher
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            all_tools = []
            
            if self.enable_filesystem:
                all_tools.extend(get_filesystem_tools())
            if self.enable_database:
                all_tools.extend(get_database_tools())
            if self.enable_actions:
                all_tools.extend(get_actions_tools())
            if self.enable_aggregator:
                all_tools.extend(get_aggregator_tools())
            if self.enable_grapher:
                all_tools.extend(get_grapher_tools())
            
            return all_tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            # Filesystem tools
            if name in ["read_file", "list_directory", "get_file_info"]:
                if not self.enable_filesystem:
                    return [TextContent(type="text", text=f"Tool '{name}' not available")]
                
                if name == "read_file":
                    result = self.fs_manager.read_file(arguments["path"], arguments.get("max_lines"))
                elif name == "list_directory":
                    result = self.fs_manager.list_directory(arguments["path"], arguments.get("pattern", "*"))
                else:  # get_file_info
                    result = self.fs_manager.get_file_info(arguments["path"])
                
                # Handle different return formats
                if "error" in result:
                    text = result["error"]
                elif "content" in result:
                    text = result["content"]
                elif "entries" in result:
                    text = json.dumps(result["entries"], indent=2)
                else:
                    text = json.dumps(result, indent=2)
                
                return [TextContent(type="text", text=text)]
            
            # Database tools
            elif name in ["query_products", "query_sales", "get_analytics"]:
                if not self.enable_database:
                    return [TextContent(type="text", text=f"Tool '{name}' not available")]
                
                if name == "query_products":
                    result = self.db_manager.query_products(**arguments)
                elif name == "query_sales":
                    result = self.db_manager.query_sales(**arguments)
                else:  # get_analytics
                    result = self.db_manager.get_analytics(arguments["metric"])
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            # Actions tools
            elif name in ["generate_report", "send_notification", "create_task", "get_action_log"]:
                if not self.enable_actions:
                    return [TextContent(type="text", text=f"Tool '{name}' not available")]
                
                if name == "generate_report":
                    result = self.actions_manager.generate_report(**arguments)
                elif name == "send_notification":
                    result = self.actions_manager.send_notification(**arguments)
                elif name == "create_task":
                    result = self.actions_manager.create_task(**arguments)
                else:  # get_action_log
                    result = self.actions_manager.get_action_log(arguments.get("action_type", "all"))
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            # Aggregator tools
            elif name == "aggregate_for_chart":
                if not self.enable_aggregator:
                    return [TextContent(type="text", text=f"Tool '{name}' not available")]
                
                data_json = arguments["data_json"]
                if isinstance(data_json, str):
                    try:
                        data = json.loads(data_json)
                    except json.JSONDecodeError as e:
                        return [TextContent(type="text", text=json.dumps({"error": f"Invalid JSON: {str(e)}"}))]
                else:
                    data = data_json
                
                result = self.aggregator_manager.aggregate_for_chart(
                    data,
                    arguments["group_by"],
                    arguments["value_field"],
                    arguments.get("aggregation", "sum")
                )
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            # Grapher tools
            elif name == "create_chart":
                if not self.enable_grapher:
                    return [TextContent(type="text", text=f"Tool '{name}' not available")]
                
                chart_data_json = arguments["chart_data_json"]
                if isinstance(chart_data_json, str):
                    try:
                        chart_data = json.loads(chart_data_json)
                    except json.JSONDecodeError as e:
                        return [TextContent(type="text", text=json.dumps({"error": f"Invalid JSON: {str(e)}"}))]
                else:
                    chart_data = chart_data_json
                
                result = self.grapher_manager.create_chart(
                    chart_data,
                    arguments.get("chart_type", "bar"),
                    arguments.get("title"),
                    arguments.get("filename")
                )
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    async def run(self):
        """Run the combined MCP server via stdio."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())