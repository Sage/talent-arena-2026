"""
MCP Workshop Lab - AI Agents & Model Context Protocol

This module provides:
1. LLM Wrappers (BedrockBridge, FreeFlowLLM)
2. ReAct Agent Engine
3. Traditional coupled tools (Tools class)
4. AI Agent that can work with coupled tools OR MCP servers
5. MCPServerBuilder for easy MCP server creation
6. Pre-defined tool packs (filesystem, database, actions)
"""

import asyncio
import inspect
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from dotenv import load_dotenv
from freeflow_llm import FreeFlowClient

load_dotenv()

# Apply nest_asyncio for notebook compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# ============================================================
# LLM WRAPPER
# ============================================================


class BedrockBridge:
    """
    Bridge to connect to Sage Bedrock via API Gateway with IAM authentication.
    """

    def __init__(self, api_url=None, profile=None, region=None):
        """
        Initialize BedrockBridge.
        
        Args:
            api_url: API Gateway URL (defaults to BEDROCK_API_URL env var)
            profile: AWS profile name (optional)
            region: AWS region (optional, will be inferred from URL or env)
        """
        self.api_url = api_url or os.getenv("BEDROCK_API_URL")
        if not self.api_url:
            raise ValueError("BEDROCK_API_URL must be provided or set in environment")
        
        # Get API password from environment
        self.api_password = os.getenv("API_PASSWORD")
        if not self.api_password:
            raise ValueError("API_PASSWORD must be set in environment")
        
        # Setup AWS session and credentials
        self.session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.credentials = self.session.get_credentials()
        if not self.credentials:
            raise ValueError("No AWS credentials found. Run 'aws configure' or 'aws sso login'")
        
        # Determine region
        if not region:
            if "eu-west-1" in self.api_url:
                region = "eu-west-1"
            elif "us-east-1" in self.api_url:
                region = "us-east-1"
            else:
                region = os.getenv("AWS_REGION", "us-east-1")
        self.region = region
        
        self.endpoint = f"{self.api_url.rstrip('/')}/v1/query"

    def _sign_request(self, payload):
        """Sign the request with AWS SigV4."""
        headers = {
            "Content-Type": "application/json",
            "X-API-Password": self.api_password
        }
        
        request = AWSRequest(
            method="POST",
            url=self.endpoint,
            data=json.dumps(payload),
            headers=headers
        )
        SigV4Auth(self.credentials, "execute-api", self.region).add_auth(request)
        return dict(request.headers), request.body

    def _convert_messages_to_prompt(self, messages):
        """Convert messages format to a single prompt string."""
        # Combine system and user messages into a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts)

    def __call__(self, messages):
        """Call the API with messages and return the response."""
        prompt = self._convert_messages_to_prompt(messages)
        payload = {"prompt": prompt}
        
        headers, body = self._sign_request(payload)
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            data=body
        )
        
        if not response.ok:
            raise Exception(f"API Error {response.status_code}: {response.text}")
        
        data = response.json()
        return data.get("response", "")

    def stream(self, messages):
        """Stream the API response (note: this may not support streaming)."""
        # Most Lambda APIs don't support streaming, so we'll return the full response
        response = self(messages)
        # Simulate streaming by yielding the full response
        yield response


class FreeFlowLLM:
    """
    Wrapper around FreeFlowClient supporting normal and streaming calls.
    
    This class provides a unified interface for interacting with various LLM providers
    (Groq, Gemini, OpenAI, Anthropic, etc.) through the FreeFlow client.
    
    Attributes:
        None (stateless wrapper)
    
    Examples:
        >>> llm = FreeFlowLLM()
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = llm(messages)
        >>> print(response)
    """

    def __call__(self, messages):
        """
        Execute a synchronous chat completion request.
        
        Args:
            messages (list[dict]): List of message dictionaries with 'role' and 'content' keys.
                                   Example: [{"role": "user", "content": "Hello"}]
        
        Returns:
            str: The LLM's text response content.
        
        Examples:
            >>> llm = FreeFlowLLM()
            >>> response = llm([{"role": "user", "content": "What is 2+2?"}])
            >>> print(response)
            '4'
        """
        try:
            with FreeFlowClient() as client:
                response = client.chat(messages=messages)
                return response.content
        except Exception as e:
            error_msg = f"LLM API Error: {str(e)}"
            print(f"⚠️  {error_msg}")
            raise RuntimeError(error_msg) from e

    def stream(self, messages):
        """
        Execute a streaming chat completion request.
        
        Yields response chunks as they become available from the LLM, enabling
        real-time display of the response to users.
        
        Args:
            messages (list[dict]): List of message dictionaries with 'role' and 'content' keys.
        
        Yields:
            str: Chunks of the LLM's response content as they arrive.
        
        Examples:
            >>> llm = FreeFlowLLM()
            >>> for chunk in llm.stream([{"role": "user", "content": "Tell me a story"}]):
            ...     print(chunk, end="", flush=True)
        """
        try:
            with FreeFlowClient() as client:
                stream = client.chat_stream(messages=messages)
                for chunk in stream:
                    if chunk.content:
                        yield chunk.content
        except Exception as e:
            error_msg = f"LLM Streaming Error: {str(e)}"
            print(f"⚠️  {error_msg}")
            raise RuntimeError(error_msg) from e


def get_llm():
    """
    Factory function to create the appropriate LLM based on environment configuration.
    
    Set USE_BEDROCK=true in your .env file to use BedrockBridge.
    Otherwise, defaults to FreeFlowLLM.
    
    Required environment variables for BedrockBridge:
    - USE_BEDROCK=true
    - BEDROCK_API_URL=https://your-api-url.execute-api.region.amazonaws.com
    - API_PASSWORD=your-password
    - Optional: AWS_PROFILE, AWS_REGION
    """
    use_bedrock = os.getenv("USE_BEDROCK", "true").lower() in ("true", "1", "yes")
    
    if use_bedrock:
        print("🔧 Using BedrockBridge for LLM calls")
        profile = os.getenv("AWS_PROFILE")
        region = os.getenv("AWS_REGION")
        return BedrockBridge(profile=profile, region=region)
    else:
        print("🔧 Using FreeFlowLLM for LLM calls")
        return FreeFlowLLM()


# ============================================================
# REACT ENGINE
# ============================================================


class ReActEngine:
    """
    Generic ReAct (Reasoning + Acting) loop engine reusable by all agents.
    
    This engine implements the ReAct pattern where an LLM iteratively:
    1. Thinks about what to do (Reasoning)
    2. Takes an action using tools (Acting)
    3. Observes the result
    4. Repeats until reaching a final answer
    
    Attributes:
        llm (FreeFlowLLM): The language model wrapper for generating responses.
        tools (Tools): The toolbox containing available actions for the agent.
        max_iterations (int): Maximum number of reasoning loops before timeout.
    
    Examples:
        >>> llm = FreeFlowLLM()
        >>> tools = Tools()
        >>> engine = ReActEngine(llm, tools, max_iterations=5)
        >>> for output in engine.run("You are a helpful assistant", "What is the weather?"):
        ...     print(output)
    """

    def __init__(self, llm, tools, max_iterations=10):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations

    def _tool_list(self):
        """Get formatted tool list with parameters if available."""
        # Check if tools object has detailed documentation method
        if hasattr(self.tools, 'get_tools_documentation'):
            return self.tools.get_tools_documentation()
        
        # Fallback to simple list of tool names
        """
        Generate a formatted list of available tools for the LLM.
        
        Dynamically inspects the tools object to find all public methods
        (bound methods only, excluding imported classes/modules).
        
        Returns:
            str: Newline-separated list of tool names prefixed with dashes.
                 Example: "- get_stock_price\n- search_news\n- scrape_hacker_news"
        """
        return "\n".join(
            f"- {name}"
            for name in dir(self.tools)
            if not name.startswith("_") and inspect.ismethod(getattr(self.tools, name))
        )

    def _parse(self, output: str):
        """
        Parse the LLM's output to extract actions or final answers.
        
        The LLM is expected to follow the ReAct format:
        - For actions: "Action: tool_name\nAction Input: input_value"
        - For final answers: "Final Answer: the answer text"
        
        Args:
            output (str): Raw text output from the LLM.
        
        Returns:
            dict: Parsed output in one of two formats:
                  - {"type": "final", "content": str} for final answers
                  - {"type": "action", "tool": str, "input": str} for tool actions
        
        Raises:
            ValueError: If the output doesn't match the expected ReAct format.
        
        Examples:
            >>> engine._parse("Action: get_stock_price\nAction Input: AAPL")
            {"type": "action", "tool": "get_stock_price", "input": "AAPL"}
            
            >>> engine._parse("Final Answer: The price is $150")
            {"type": "final", "content": "The price is $150"}
        """
        if "Final Answer:" in output:
            return {
                "type": "final",
                "content": output.split("Final Answer:")[-1].strip(),
            }

        action_match = re.search(r"Action:\s*(.*)", output)
        input_match = re.search(r"Action Input:\s*(.*)", output)
        # Check for Action FIRST (even if Final Answer appears - LLM might include both by mistake)
        # action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", output)
        # input_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", output)

        if action_match and input_match:
            return {
                "type": "action",
                "tool": action_match.group(1).strip(),
                "input": input_match.group(1).strip(),
            }
        
        # Only check Final Answer if no Action was found
        if "Final Answer:" in output:
            return {
                "type": "final",
                "content": output.split("Final Answer:")[-1].strip(),
            }

        raise ValueError("Invalid LLM format")

    def run(self, system_prompt: str, user_input: str, stream_final=False):
        """
        Execute the ReAct reasoning loop.
        
        Iteratively prompts the LLM to think, act, and observe until a final answer
        is reached or max_iterations is exceeded. Each iteration:
        1. Sends the system prompt, question, and scratchpad to the LLM
        2. Parses the LLM's response for actions or final answer
        3. Executes tools if actions are requested
        4. Updates the scratchpad with observations
        5. Yields outputs for visibility
        
        Args:
            system_prompt (str): The agent's role and behavioral instructions.
            user_input (str): The user's question or task.
            stream_final (bool): If True, streams the final answer token by token.
                                If False, returns the final answer as a single chunk.
        
        Yields:
            str: Status updates including:
                 - LLM reasoning outputs
                 - Tool actions and observations
                 - Final answer (streamed or complete)
                 - Error messages if max iterations reached
        
        Examples:
            >>> for chunk in engine.run("You are a stock analyst", "AAPL stock price"):
            ...     print(chunk)
        """
        scratchpad = ""
        system_message = f"""
                        {system_prompt}

                        You have access to the following tools:
                        {self._tool_list()}

                        Use EXACTLY this format:

                        Thought:
                        Action:
                        Action Input:
                        Observation:
                        Final Answer:
                        """

        for iteration in range(self.max_iterations):
            try:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Question: {user_input}\n\n{scratchpad}"},
                ]
                llm_output = self.llm(messages)
                
                try:
                    parsed = self._parse(llm_output)
                except ValueError as e:
                    error_msg = f"⚠️  Parse Error: {str(e)}\nLLM Output was: {llm_output}\n"
                    yield error_msg
                    
                    # Add actionable feedback to scratchpad so LLM can self-correct
                    scratchpad += f"\nObservation: Your output format was incorrect. {str(e)}\n"
                    scratchpad += "Remember to use EXACTLY this format:\n"
                    scratchpad += "Thought: [your reasoning]\n"
                    scratchpad += "Action: [tool_name]\n"
                    scratchpad += "Action Input: [input_value]\n"
                    scratchpad += "OR if you have the final answer:\n"
                    scratchpad += "Final Answer: [your answer]\n"
                    continue

                if parsed["type"] == "final":
                    # Only yield the clean final answer, not the raw LLM output
                    yield f"Final Answer: {parsed['content']}\n"
                    if stream_final:
                        try:
                            stream_messages = [
                                {
                                    "role": "system",
                                    "content": "Return only the final answer text.",
                                },
                                {"role": "user", "content": parsed["content"]},
                            ]
                            for chunk in self.llm.stream(stream_messages):
                                yield chunk
                        except Exception as e:
                            yield f"\n⚠️  Streaming error: {str(e)}\n"
                    return

                elif parsed["type"] == "action":
                    # Yield LLM output for visibility during action steps
                    yield f"LLM Output: {llm_output}\n"
                    
                    tool_name = parsed["tool"]
                    tool_input = parsed["input"]
                    tool = getattr(self.tools, tool_name, None)
                    
                    # Verify the tool exists and is a bound method (not a class attribute)
                    if not tool or not inspect.ismethod(tool):
                        error_msg = f"Tool '{tool_name}' not found. Available tools: {self._tool_list()}"
                        yield f"⚠️  {error_msg}\n"
                        scratchpad += f"\nObservation: Error - {error_msg}\n"
                        continue
                    
                    try:
                        observation = tool(tool_input)
                        step = f"Action: {tool_name}\nAction Input: {tool_input}\nObservation: {observation}\n"
                        yield step
                        scratchpad += f"""
                                        Thought:
                                        Action: {tool_name}
                                        Action Input: {tool_input}
                                        Observation: {observation}
                                        """
                    except Exception as e:
                        error_msg = f"Tool execution failed: {str(e)}"
                        yield f"⚠️  {error_msg}\n"
                        scratchpad += f"\nObservation: Error - {error_msg}\n"
                        
            except RuntimeError as e:
                yield f"⚠️  LLM Error on iteration {iteration + 1}: {str(e)}\n"
                yield "Stopping due to LLM failure.\n"
                return
            except Exception as e:
                yield f"⚠️  Unexpected error on iteration {iteration + 1}: {str(e)}\n"
                continue
                
        yield "Max iterations reached without final answer.\n"


# ============================================================
# BASE AGENT
# ============================================================


class BaseAgent:
    """
    Base class for all specialized agents using the ReAct pattern.
    
    This abstract class provides the foundation for creating domain-specific agents.
    Subclasses only need to implement the `system_prompt` property to define their
    specialized behavior. The run and stream methods are inherited and work automatically.
    
    Attributes:
        tools (Tools): The toolbox containing available actions.
        llm (FreeFlowLLM): The language model wrapper.
        engine (ReActEngine): The ReAct reasoning loop engine.
    
    Examples:
        >>> class MyAgent(BaseAgent):
        ...     @property
        ...     def system_prompt(self):
        ...         return "You are a helpful assistant."
        >>> agent = MyAgent(Tools())
        >>> for output in agent.run("What's the weather?"):
        ...     print(output)
    """

    def __init__(self, tools):
        self.tools = tools
        self.llm = get_llm()
        self.engine = ReActEngine(self.llm, tools)

    @property
    def system_prompt(self):
        """
        Define the agent's role and behavioral instructions.
        
        This property must be implemented by all subclasses to specify:
        - The agent's role and expertise
        - Which tools to use and when
        - Output format expectations
        - Any constraints or guidelines
        
        Returns:
            str: The system prompt for this agent.
        
        Raises:
            NotImplementedError: If not implemented by subclass.
        
        Examples:
            >>> @property
            >>> def system_prompt(self):
            ...     return "You are a financial analyst. Use get_stock_price to fetch data."
        """
        raise NotImplementedError

    def run(self, user_input):
        """
        Execute the agent with non-streaming output.
        
        Runs the ReAct loop and yields intermediate steps and the final answer.
        
        Args:
            user_input (str): The user's question or task for the agent.
        
        Yields:
            str: Progress updates including reasoning, actions, and final answer.
        
        Examples:
            >>> agent = MyAgent(tools)
            >>> for chunk in agent.run("What is the stock price of AAPL?"):
            ...     print(chunk)
        """
        return self.engine.run(self.system_prompt, user_input)

    def stream(self, user_input):
        """
        Execute the agent with streaming output for the final answer.
        
        Similar to run(), but streams the final answer token-by-token for better UX.
        
        Args:
            user_input (str): The user's question or task for the agent.
        
        Yields:
            str: Progress updates and streaming final answer chunks.
        
        Examples:
            >>> agent = MyAgent(tools)
            >>> for chunk in agent.stream("Analyze AAPL stock"):
            ...     print(chunk, end="", flush=True)
        """
        return self.engine.run(self.system_prompt, user_input, stream_final=True)


# ============================================================
# TOOLS
# ============================================================


class Tools:
    """
    Toolbox containing agent capabilities and actions.
    
    This class provides a collection of tools that agents can use to interact
    with external services, APIs, and data sources. Each method is automatically
    discovered and made available to agents through the ReAct engine.
    
    Available Tools:
        - get_stock_price: Fetch real-time stock market data
        - search_news: Search for recent news articles
        - scrape_hacker_news: Get trending tech stories from Hacker News
        - get_project_summary: Web search for project/topic information
        - get_crypto_prices: Fetch cryptocurrency prices from Binance
        - generate_chart: Create QuickChart visualization URLs
    
    Note:
        New tools can be added by simply defining new methods in this class.
        They will be automatically available to all agents.
    
    Examples:
        >>> tools = Tools()
        >>> price = tools.get_stock_price("AAPL")
        >>> print(price)
    """

    import json
    import requests
    import yfinance
    from bs4 import BeautifulSoup
    from ddgs import DDGS

    def get_stock_price(self, symbol):
        """
        Fetch real-time stock market data for a given symbol.
        
        Uses yfinance to retrieve the latest trading day data including current price,
        price change, and percent change from market open.
        
        Args:
            symbol (str): Stock ticker symbol (e.g., "AAPL", "GOOGL", "TSLA").
        
        Returns:
            dict or str: If data found, returns dict with keys:
                         - symbol (str): The stock ticker
                         - price (float): Current/closing price
                         - change (float): Price change from open
                         - percent_change (float): Percentage change
                         If no data found, returns "No data found." string.
        
        Examples:
            >>> tools = Tools()
            >>> result = tools.get_stock_price("AAPL")
            >>> print(result)
            {'symbol': 'AAPL', 'price': 185.32, 'change': 2.14, 'percent_change': 1.17}
        """
        try:
            if not symbol or not isinstance(symbol, str):
                return "Error: Invalid symbol. Please provide a valid stock ticker (e.g., 'AAPL')."
            
            ticker = self.yfinance.Ticker(symbol.upper())
            data = ticker.history(period="1d")
            
            if data.empty:
                return f"No data found for '{symbol}'. The market may be closed or the symbol is invalid."

            price = data["Close"].iloc[-1]
            open_price = data["Open"].iloc[-1]
            change = price - open_price
            percent_change = (change / open_price) * 100

            return {
                "symbol": symbol.upper(),
                "price": float(price),
                "change": float(change),
                "percent_change": float(percent_change),
            }
        except Exception as e:
            return f"Error fetching stock price for '{symbol}': {str(e)}"

    def search_news(self, query):
        """
        Search for recent news articles using DuckDuckGo.
        
        Args:
            query (str): Search query for news articles.
                        Example: "Apple stock earnings", "Tesla cybertruck news"
        
        Returns:
            list[dict]: List of up to 3 news articles, each containing:
                        - title: Article headline
                        - url: Article URL
                        - source: News source name
                        - date: Publication date
                        - body: Article excerpt/summary
        
        Examples:
            >>> tools = Tools()
            >>> news = tools.search_news("AI regulation Europe")
            >>> print(news[0]['title'])
        """
        try:
            if not query or not isinstance(query, str):
                return [{"title": "Error", "body": "Invalid search query. Please provide a search term."}]
            
            results = list(self.DDGS().news(query, max_results=3))
            return results if results else [{"title": "No results", "body": f"No news found for '{query}'."}]
        except Exception as e:
            return [{"title": "Error", "body": f"News search failed: {str(e)}"}]

    def scrape_hacker_news(self, _unused=None):
        """
        Scrape the top trending stories from Hacker News.
        
        Fetches the current front page of Hacker News and extracts the top 3 headlines
        with their corresponding URLs.
        
        Args:
            _unused: Ignored parameter for ReActEngine compatibility (can be any value).
        
        Returns:
            list[tuple]: List of (title, url) tuples for the top 3 stories.
        
        Examples:
            >>> tools = Tools()
            >>> stories = tools.scrape_hacker_news()
            >>> for title, url in stories:
            ...     print(f"{title}: {url}")
        """
        try:
            url = "https://news.ycombinator.com/"
            resp = self.requests.get(url, timeout=10)
            resp.raise_for_status()
            
            soup = self.BeautifulSoup(resp.text, "html.parser")
            links = soup.select(".titleline > a")
            
            if not links:
                return [("No stories found", "https://news.ycombinator.com/")]
            
            return [(a.text, a.get("href", "")) for a in links[:3]]
        except self.requests.exceptions.Timeout:
            return [("Error: Request timeout", "Check your internet connection")]
        except self.requests.exceptions.RequestException as e:
            return [(f"Error: Network error - {str(e)}", "")]
        except Exception as e:
            return [(f"Error: Scraping failed - {str(e)}", "")]

    def get_project_summary(self, query):
        """
        Perform a web search to get information about a project or topic.
        
        Uses DuckDuckGo to search the web and return the top result, useful for
        gathering context about technical projects, frameworks, or concepts.
        
        Args:
            query (str): Search query for the project/topic.
                        Example: "Rust WebAssembly framework", "React Server Components"
        
        Returns:
            list[dict]: List containing one search result with keys:
                        - title: Page title
                        - href: URL
                        - body: Text snippet/description
        
        Examples:
            >>> tools = Tools()
            >>> result = tools.get_project_summary("Next.js 15")
            >>> print(result[0]['body'])
        """
        try:
            if not query or not isinstance(query, str):
                return [{"title": "Error", "body": "Invalid query. Please provide a search term."}]
            
            results = list(self.DDGS().text(query, max_results=1))
            return results if results else [{"title": "No results", "body": f"No information found for '{query}'."}]
        except Exception as e:
            return [{"title": "Error", "body": f"Search failed: {str(e)}"}]

    def get_crypto_prices(self, symbols_csv):
        """
        Fetch current cryptocurrency prices from Binance.
        
        Retrieves real-time prices for multiple cryptocurrencies against USDT.
        
        Args:
            symbols_csv (str or list): Comma-separated string of crypto symbols
                                       (e.g., "BTC,ETH,SOL") or list of symbols
                                       (e.g., ["BTC", "ETH", "SOL"]).
        
        Returns:
            dict: Dictionary with one of the following structures:
                  
                  Success case:
                  - Maps symbols to their current prices in USDT (float)
                  - Failed individual symbols have None as their value
                  - Example: {'BTC': 65432.10, 'ETH': 3521.45, 'INVALID': None}
                  
                  Error cases (dict with 'error' key):
                  - {'error': 'No valid symbols provided'} - Empty/invalid input
                  - {'error': 'symbols_csv must be a string or list'} - Wrong type
                  - {'error': 'Unexpected error: ...'} - Other failures
                  
                  To check for errors: use 'error' in result
        
        Examples:
            >>> tools = Tools()
            >>> prices = tools.get_crypto_prices("BTC,ETH,SOL")
            >>> if 'error' in prices:
            ...     print(f"Error: {prices['error']}")
            ... else:
            ...     print(prices)
            {'BTC': 65432.10, 'ETH': 3521.45, 'SOL': 142.89}
            
            >>> # Handle individual failures
            >>> prices = tools.get_crypto_prices("BTC,INVALID")
            >>> if 'error' not in prices:
            ...     btc = prices.get('BTC')
            ...     if btc is not None:
            ...         print(f"BTC: ${btc}")
        """
        try:
            import json
            
            # Handle JSON string input (e.g., '["BTC", "ETH"]' from LLM)
            if isinstance(symbols_csv, str) and symbols_csv.strip().startswith('['):
                try:
                    symbols_csv = json.loads(symbols_csv)
                except json.JSONDecodeError:
                    pass  # Fall through to normal string processing
            
            if isinstance(symbols_csv, str):
                symbols = [s.strip().upper() for s in symbols_csv.split(",") if s.strip()]
            elif isinstance(symbols_csv, list):
                symbols = [s.strip().upper() for s in symbols_csv if isinstance(s, str) and s.strip()]
            else:
                raise TypeError("symbols_csv must be a string or list")
            
            if not symbols:
                return {"error": "No valid symbols provided"}
            
            prices = {}

            for symbol in symbols:
                try:
                    pair = f"{symbol}USDT"
                    url = f"https://api.binance.com/api/v3/ticker/price?symbol={pair}"
                    resp = self.requests.get(url, timeout=10)
                    
                    if resp.status_code == 200:
                        prices[symbol] = float(resp.json()["price"])
                    else:
                        prices[symbol] = None
                        print(f"⚠️  Failed to fetch {symbol}: HTTP {resp.status_code}")
                except (ValueError, KeyError) as e:
                    prices[symbol] = None
                    print(f"⚠️  Error parsing {symbol}: {str(e)}")
                except self.requests.exceptions.Timeout:
                    prices[symbol] = None
                    print(f"⚠️  Timeout fetching {symbol}")
                except Exception as e:
                    prices[symbol] = None
                    print(f"⚠️  Error fetching {symbol}: {str(e)}")

            return prices
        except TypeError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def generate_chart(self, price_dict):
        """
        Generate a QuickChart bar chart URL from price data.
        
        Creates a shareable URL that renders a bar chart visualization of prices.
        Uses the QuickChart.io service to generate charts without requiring local
        chart rendering libraries.
        
        Args:
            price_dict (dict or str): Dictionary mapping labels to numeric values,
                                     or JSON string representation of a dict.
                                     Example: {"BTC": 65000, "ETH": 3500, "SOL": 140}
                                     or '{"BTC": 65000, "ETH": 3500}'
        
        Returns:
            str: Complete QuickChart URL that can be opened in a browser to view
                 the generated chart.
        
        Examples:
            >>> tools = Tools()
            >>> prices = {"BTC": 65000, "ETH": 3500, "SOL": 140}
            >>> chart_url = tools.generate_chart(prices)
            >>> print(chart_url)
            https://quickchart.io/chart?c=%7B%22type%22%3A%22bar%22...
            
            # Open in browser to see the chart
            >>> import webbrowser
            >>> webbrowser.open(chart_url)
        """
        try:
            import json
            
            # Handle JSON string input (e.g., '{"BTC": 65000}' from LLM)
            if isinstance(price_dict, str):
                try:
                    price_dict = json.loads(price_dict)
                except json.JSONDecodeError as e:
                    return f"Error: Invalid JSON string. {str(e)}"
            
            if not isinstance(price_dict, dict):
                return "Error: Input must be a dictionary (e.g., {'BTC': 65000, 'ETH': 3500})"
            
            if not price_dict:
                return "Error: Empty price dictionary provided"
            
            # Filter out None values and validate numeric data
            filtered_data = {k: v for k, v in price_dict.items() if v is not None}
            
            if not filtered_data:
                return "Error: No valid price data to chart"
            
            labels = list(filtered_data.keys())
            data = []
            
            for value in filtered_data.values():
                try:
                    data.append(float(value))
                except (ValueError, TypeError):
                    return f"Error: Non-numeric value found in price data: {value}"

            chart_config = {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{"label": "Price (USDT)", "data": data}],
                },
            }

            base_url = "https://quickchart.io/chart"
            chart_url = f"{base_url}?c={self.requests.utils.quote(json.dumps(chart_config))}"
            
            # Validate URL length (QuickChart has limits)
            if len(chart_url) > 16000:
                return "Error: Chart data too large. Try with fewer items."
            
            return chart_url
            
        except Exception as e:
            return f"Error generating chart: {str(e)}"


# ============================================================
# NOTEBOOK HELPERS
# ============================================================


def test_all_tools(verbose=True):
    """
    Test all available tools with sample inputs.
    
    Provides a quick way to verify that all tools are working and see
    what kind of data they return. Perfect for notebook demonstrations.
    
    Args:
        verbose (bool): If True, prints detailed output. If False, only shows summary.
    
    Returns:
        dict: Test results for each tool with success/failure status.
    
    Examples:
        >>> from lab import test_all_tools
        >>> 
        >>> # Run all tests with output
        >>> results = test_all_tools()
        >>> 
        >>> # Quick check without verbose output
        >>> results = test_all_tools(verbose=False)
        >>> print(f"Passed: {sum(1 for r in results.values() if r['success'])}/6")
    """
    tools = Tools()
    results = {}
    
    if verbose:
        print("🧪 Testing All Tools")
        print("=" * 70)
    
    # Test 1: Stock Price
    test_name = "get_stock_price"
    try:
        result = tools.get_stock_price("AAPL")
        success = isinstance(result, dict) and "price" in result
        results[test_name] = {"success": success, "result": result}
        if verbose:
            status = "✅" if success else "❌"
            print(f"\n{status} {test_name}('AAPL')")
            if success:
                print(f"   Price: ${result['price']:.2f} ({result['percent_change']:+.2f}%)")
    except Exception as e:
        results[test_name] = {"success": False, "error": str(e)}
        if verbose:
            print(f"\n❌ {test_name}: {e}")
    
    # Test 2: News Search
    test_name = "search_news"
    try:
        result = tools.search_news("technology")
        success = isinstance(result, list) and len(result) > 0
        results[test_name] = {"success": success, "result": result}
        if verbose:
            status = "✅" if success else "❌"
            print(f"\n{status} {test_name}('technology')")
            if success and result:
                print(f"   Found {len(result)} articles")
    except Exception as e:
        results[test_name] = {"success": False, "error": str(e)}
        if verbose:
            print(f"\n❌ {test_name}: {e}")
    
    # Test 3: Hacker News
    test_name = "scrape_hacker_news"
    try:
        result = tools.scrape_hacker_news()
        success = isinstance(result, list) and len(result) > 0
        results[test_name] = {"success": success, "result": result}
        if verbose:
            status = "✅" if success else "❌"
            print(f"\n{status} {test_name}()")
            if success:
                print(f"   Found {len(result)} trending stories")
    except Exception as e:
        results[test_name] = {"success": False, "error": str(e)}
        if verbose:
            print(f"\n❌ {test_name}: {e}")
    
    # Test 4: Project Summary
    test_name = "get_project_summary"
    try:
        result = tools.get_project_summary("Python")
        success = isinstance(result, list) and len(result) > 0
        results[test_name] = {"success": success, "result": result}
        if verbose:
            status = "✅" if success else "❌"
            print(f"\n{status} {test_name}('Python')")
    except Exception as e:
        results[test_name] = {"success": False, "error": str(e)}
        if verbose:
            print(f"\n❌ {test_name}: {e}")
    
    # Test 5: Crypto Prices
    test_name = "get_crypto_prices"
    try:
        result = tools.get_crypto_prices("BTC,ETH")
        success = isinstance(result, dict) and "BTC" in result
        results[test_name] = {"success": success, "result": result}
        if verbose:
            status = "✅" if success else "❌"
            print(f"\n{status} {test_name}('BTC,ETH')")
            if success:
                print(f"   BTC: ${result.get('BTC', 'N/A')}")
    except Exception as e:
        results[test_name] = {"success": False, "error": str(e)}
        if verbose:
            print(f"\n❌ {test_name}: {e}")
    
    # Test 6: Chart Generation
    test_name = "generate_chart"
    try:
        result = tools.generate_chart({"Test": 100, "Data": 150})
        success = isinstance(result, str) and result.startswith("https://")
        results[test_name] = {"success": success, "result": result}
        if verbose:
            status = "✅" if success else "❌"
            print(f"\n{status} {test_name}(dict)")
            if success:
                print(f"   URL: {result[:60]}...")
    except Exception as e:
        results[test_name] = {"success": False, "error": str(e)}
        if verbose:
            print(f"\n❌ {test_name}: {e}")
    
    if verbose:
        print("\n" + "=" * 70)
        passed = sum(1 for r in results.values() if r.get("success", False))
        print(f"✅ Tests passed: {passed}/{len(results)}")
        print("=" * 70)
    
    return results

# ============================================================
# AI AGENT - Works with coupled tools OR MCP servers
# ============================================================


class AIAgent:
    """
    AI Agent that can use either:
    1. Coupled tools (traditional approach - tools are tightly bound)
    2. MCP tools (decoupled approach - tools come from external servers)
    
    This demonstrates the difference between coupled and decoupled architectures.
    """
    
    def __init__(self, tools=None, name: str = "AI Agent"):
        """
        Initialize the AI Agent.
        
        Args:
            tools: A tools object with callable methods. Can be:
                   - Tools instance (coupled)
                   - MCPToolsWrapper instance (decoupled via MCP)
                   - None (no tools available)
            name: Display name for the agent
        """
        self.name = name
        self.tools = tools
        self.llm = get_llm()
        self._engine = None
        if tools:
            self._engine = ReActEngine(self.llm, tools)
    
    @property
    def system_prompt(self) -> str:
        return """You are a helpful AI assistant. Answer questions accurately and concisely.
        
When you need to use a tool, respond with this EXACT format:

Thought: [your reasoning about what to do]
Action: [tool name]
Action Input: [JSON object with the parameters]

After receiving the Observation (tool result), either:
- Use another tool if needed
- OR provide your Final Answer:

Final Answer: [your answer to the user]

IMPORTANT:
- Always use valid JSON for Action Input
- Wait for Observation before continuing
- When you have enough information, give a Final Answer
"""
    
    def get_tools_list(self) -> list[str]:
        """Get list of available tool names."""
        if not self.tools:
            return []
        
        if hasattr(self.tools, 'get_tools_list'):
            return self.tools.get_tools_list()
        
        # Fallback: introspect the tools object
        return [
            name for name in dir(self.tools)
            if not name.startswith("_") and callable(getattr(self.tools, name))
        ]
    
    def get_tools_documentation(self) -> str:
        """Get formatted documentation for all tools."""
        if not self.tools:
            return "(No tools available)"
        
        if hasattr(self.tools, 'get_tools_documentation'):
            return self.tools.get_tools_documentation()
        
        # Fallback: generate basic documentation
        tool_docs = []
        for name in self.get_tools_list():
            method = getattr(self.tools, name)
            doc = method.__doc__ or "No description"
            tool_docs.append(f"- {name}: {doc.strip().split(chr(10))[0]}")
        return "\n".join(tool_docs)
    
    def show_tools(self):
        """Display the agent's available tools."""
        tools = self.get_tools_list()
        print(f"🤖 {self.name}")
        print(f"{'=' * 50}")
        if tools:
            print(f"📦 Available Tools ({len(tools)}):")
            for tool in tools:
                print(f"   • {tool}")
            print(f"\n📋 Tool Documentation:")
            print(self.get_tools_documentation())
        else:
            print("⚠️  No tools attached to this agent!")
            print("   The agent can only answer from its training data.")
        print(f"{'=' * 50}")
    
    def attach_tools(self, tools):
        """Attach or replace the tools for this agent."""
        self.tools = tools
        if tools:
            self._engine = ReActEngine(self.llm, tools)
        else:
            self._engine = None
        print(f"✅ Tools {'attached' if tools else 'removed'} from {self.name}")
    
    def remove_tools(self):
        """Remove all tools from this agent."""
        self.attach_tools(None)
    
    def run(self, question: str, verbose: bool = True):
        """
        Run the agent on a question.
        
        Args:
            question: The user's question
            verbose: Whether to print step-by-step output
        """
        print(f"\n🤖 {self.name} starting...\n")
        print("=" * 70)
        
        if not self._engine or not self.tools:
            # No tools - just use LLM directly
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}
            ]
            response = self.llm(messages)
            if verbose:
                print(f"💭 Response (no tools available):\n{response}")
            print("=" * 70)
            print("\n✅ Agent finished!")
            return response
        
        # Use ReAct engine with tools
        full_output = []
        for step in self._engine.run(self.system_prompt, question, stream_final=False):
            if verbose:
                print(step, end="")
            full_output.append(step)
        
        print("=" * 70)
        print("\n✅ Agent finished!")
        return "".join(full_output)


class EmptyTools:
    """
    Placeholder tools object with no tools.
    Used to demonstrate an agent without capabilities.
    """
    
    def get_tools_list(self) -> list[str]:
        return []
    
    def get_tools_documentation(self) -> str:
        return "(No tools available)"


# ============================================================
# MCP TOOLS WRAPPER - Connects AI Agent to MCP Server
# ============================================================


class MCPToolsWrapper:
    """
    Wraps an MCP client to provide a tools interface for the AI Agent.
    
    This allows the agent to use MCP server tools as if they were local methods,
    demonstrating the decoupled architecture.
    """
    
    def __init__(self, mcp_client):
        """
        Initialize with an MCP client.
        
        Args:
            mcp_client: A SyncMCPClient or similar with call_tool method
        """
        self._client = mcp_client
        self._tools_cache = {}
        self._setup_tools()
    
    def _setup_tools(self):
        """Dynamically create methods for each MCP tool."""
        for tool_name in self._client.get_tools_list():
            # Create a closure to capture tool_name
            def make_tool_method(name):
                def tool_method(args_str):
                    args = self._parse_args(args_str)
                    return self._client.call_tool(name, args)
                tool_method.__doc__ = f"MCP Tool: {name}"
                return tool_method
            
            setattr(self, tool_name, make_tool_method(tool_name))
            self._tools_cache[tool_name] = getattr(self, tool_name)
    
    def _parse_args(self, args_str) -> dict:
        """Parse JSON string arguments into dict."""
        if isinstance(args_str, dict):
            return args_str
        if isinstance(args_str, str):
            args_str = args_str.strip()
            if args_str.startswith("{"):
                try:
                    return json.loads(args_str)
                except json.JSONDecodeError:
                    pass
            return {"query": args_str}
        return {}
    
    def get_tools_list(self) -> list[str]:
        """Get list of available tool names."""
        return self._client.get_tools_list()
    
    def get_tools_documentation(self) -> str:
        """Get formatted documentation for all tools."""
        return self._client.get_tools_documentation()


# ============================================================
# MCP SERVER BUILDER - Easy Server Creation for Workshop
# ============================================================


@dataclass
class ToolPack:
    """Pre-defined pack of tools that can be added to an MCP server."""
    name: str
    description: str
    tools: list[str]
    

# Available tool packs for the workshop
TOOL_PACKS = {
    "filesystem": ToolPack(
        name="Filesystem Tools",
        description="Read files, list directories, get file info",
        tools=["read_file", "list_directory", "get_file_info"]
    ),
    "database": ToolPack(
        name="Database Tools", 
        description="Query products, sales data, and analytics",
        tools=["query_products", "query_sales", "get_analytics"]
    ),
    "actions": ToolPack(
        name="Action Tools",
        description="Generate reports, send notifications, create tasks",
        tools=["generate_report", "send_notification", "create_task"]
    ),
}


class MCPServerBuilder:
    """
    Simple builder for creating MCP servers in the workshop.
    
    This provides a user-friendly interface for participants with
    minimal technical ability to create and configure MCP servers.
    
    Usage:
        server = MCPServerBuilder("my-server")
        server.add_tool_pack("filesystem")
        server.add_tool_pack("database")
        server.start()
    """
    
    def __init__(self, name: str = "workshop-server"):
        """
        Create a new MCP server builder.
        
        Args:
            name: Name for your MCP server
        """
        self.name = name
        self.tool_packs: list[str] = []
        self._process = None
        self._client = None
        self._async_client = None
        self._started = False
        
        print(f"🏗️  Created MCP Server Builder: '{name}'")
        print(f"   Add tool packs using: server.add_tool_pack('name')")
        print(f"   Available packs: {', '.join(TOOL_PACKS.keys())}")
    
    @staticmethod
    def list_available_tool_packs():
        """Show all available tool packs."""
        print("\n📦 Available Tool Packs:")
        print("=" * 50)
        for key, pack in TOOL_PACKS.items():
            print(f"\n   '{key}'")
            print(f"   {pack.description}")
            print(f"   Tools: {', '.join(pack.tools)}")
        print("\n" + "=" * 50)
    
    def add_tool_pack(self, pack_name: str) -> "MCPServerBuilder":
        """
        Add a tool pack to your server.
        
        Args:
            pack_name: One of 'filesystem', 'database', or 'actions'
            
        Returns:
            self (for chaining)
        """
        if pack_name not in TOOL_PACKS:
            print(f"❌ Unknown tool pack: '{pack_name}'")
            print(f"   Available: {', '.join(TOOL_PACKS.keys())}")
            return self
        
        if pack_name in self.tool_packs:
            print(f"⚠️  Tool pack '{pack_name}' already added")
            return self
        
        self.tool_packs.append(pack_name)
        pack = TOOL_PACKS[pack_name]
        print(f"✅ Added '{pack.name}' to server")
        print(f"   New tools: {', '.join(pack.tools)}")
        return self
    
    def show_configuration(self):
        """Display the current server configuration."""
        print(f"\n🖥️  MCP Server: {self.name}")
        print("=" * 50)
        
        if not self.tool_packs:
            print("⚠️  No tool packs added yet!")
            print("   Use server.add_tool_pack('name') to add tools")
        else:
            print(f"📦 Tool Packs ({len(self.tool_packs)}):")
            all_tools = []
            for pack_name in self.tool_packs:
                pack = TOOL_PACKS[pack_name]
                print(f"\n   {pack.name}:")
                for tool in pack.tools:
                    print(f"      • {tool}")
                    all_tools.append(tool)
            print(f"\n📊 Total Tools: {len(all_tools)}")
        
        print("=" * 50)
        if self._started:
            print("🟢 Server Status: RUNNING")
        else:
            print("🔴 Server Status: NOT STARTED")
            print("   Call server.start() to launch the server")
    
    def start(self) -> "SyncMCPClient":
        """
        Start the MCP server and return a connected client.
        
        Returns:
            SyncMCPClient connected to the server
        """
        if self._started:
            print("⚠️  Server already started!")
            return self._client
        
        if not self.tool_packs:
            print("❌ Cannot start: no tool packs added!")
            print("   Use server.add_tool_pack('name') first")
            return None
        
        # Import here to avoid circular imports
        from stdio_mcp_client import StdioMCPClient, SyncMCPClient
        
        print(f"\n🚀 Starting MCP Server '{self.name}'...")
        print("   (Running as subprocess in background)")
        
        # Create async client and connect
        self._async_client = StdioMCPClient()
        
        # Build args based on selected tool packs
        tool_packs_arg = ",".join(self.tool_packs)
        
        async def connect():
            return await self._async_client.connect(
                command=sys.executable,
                args=["run_mcp_server.py", "--packs", tool_packs_arg]
            )
        
        try:
            loop = asyncio.get_event_loop()
            tools = loop.run_until_complete(connect())
            
            self._client = SyncMCPClient(self._async_client)
            self._started = True
            
            print(f"\n✅ Server started successfully!")
            print(f"📦 Discovered {len(tools)} tools:")
            for tool in tools:
                print(f"   • {tool.name}: {tool.description}")
            print(f"\n🎯 Use server.get_client() to get the connected client")
            
            return self._client
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return None
    
    def get_client(self):
        """Get the connected MCP client."""
        if not self._started:
            print("⚠️  Server not started. Call server.start() first.")
            return None
        return self._client
    
    def stop(self):
        """Stop the MCP server."""
        if not self._started:
            print("⚠️  Server not running")
            return
        
        print("🛑 Stopping MCP server...")
        
        async def close():
            await self._async_client.close()
        
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(close())
        except:
            pass
        
        self._started = False
        self._client = None
        self._async_client = None
        print("✅ Server stopped")


# ============================================================
# MCP CLIENT CONNECTION HELPER
# ============================================================


class MCPConnection:
    """
    Helper class to connect to any MCP server.
    
    Can connect to:
    1. Local servers (via subprocess)
    2. Remote servers (via URL - placeholder for company servers)
    """
    
    def __init__(self):
        self._clients = {}
        self._async_clients = {}
    
    def connect_local(self, server_script: str = "run_mcp_server.py", 
                      name: str = "local") -> "SyncMCPClient":
        """
        Connect to a local MCP server.
        
        Args:
            server_script: Path to the server script
            name: Friendly name for this connection
            
        Returns:
            SyncMCPClient connected to the server
        """
        from stdio_mcp_client import StdioMCPClient, SyncMCPClient
        
        print(f"🔌 Connecting to local MCP server...")
        
        async_client = StdioMCPClient()
        
        async def connect():
            return await async_client.connect(
                command=sys.executable,
                args=[server_script]
            )
        
        loop = asyncio.get_event_loop()
        tools = loop.run_until_complete(connect())
        
        client = SyncMCPClient(async_client)
        self._clients[name] = client
        self._async_clients[name] = async_client
        
        print(f"✅ Connected! Found {len(tools)} tools")
        return client
    
    def connect_remote(self, url: str, name: str = "remote"):
        """
        Connect to a remote MCP server via URL.
        
        Args:
            url: The server URL (e.g., company internal server)
            name: Friendly name for this connection
            
        NOTE: This is a placeholder for connecting to company MCP servers.
        The actual implementation will be provided when the URL is available.
        """
        print(f"🌐 Remote MCP Connection")
        print(f"   URL: {url}")
        print(f"   Name: {name}")
        print(f"\n⚠️  Remote connection not yet implemented.")
        print(f"   This will connect to the company MCP server when available.")
        return None
    
    def get_client(self, name: str = "local"):
        """Get a connected client by name."""
        return self._clients.get(name)
    
    def disconnect(self, name: str = "local"):
        """Disconnect a client."""
        if name in self._async_clients:
            async def close():
                await self._async_clients[name].close()
            
            loop = asyncio.get_event_loop()
            loop.run_until_complete(close())
            
            del self._clients[name]
            del self._async_clients[name]
            print(f"✅ Disconnected from '{name}'")
    
    def disconnect_all(self):
        """Disconnect all clients."""
        for name in list(self._async_clients.keys()):
            self.disconnect(name)
