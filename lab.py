"""
Reusable ReAct Agent Framework using FreeFlow LLM
"""

import re

from dotenv import load_dotenv
from freeflow_llm import FreeFlowClient

load_dotenv()

# ============================================================
# LLM WRAPPER
# ============================================================


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
        with FreeFlowClient() as client:
            response = client.chat(messages=messages)
            return response.content

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
        with FreeFlowClient() as client:
            stream = client.chat_stream(messages=messages)
            for chunk in stream:
                if chunk.content:
                    yield chunk.content


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

    def __init__(self, llm, tools, max_iterations=5):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations

    def _tool_list(self):
        """
        Generate a formatted list of available tools for the LLM.
        
        Dynamically inspects the tools object to find all callable methods
        that don't start with underscore (public methods).
        
        Returns:
            str: Newline-separated list of tool names prefixed with dashes.
                 Example: "- get_stock_price\n- search_news\n- scrape_hacker_news"
        """
        return "\n".join(
            f"- {name}"
            for name in dir(self.tools)
            if not name.startswith("_") and callable(getattr(self.tools, name))
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

        if action_match and input_match:
            return {
                "type": "action",
                "tool": action_match.group(1).strip(),
                "input": input_match.group(1).strip(),
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

        for _ in range(self.max_iterations):
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Question: {user_input}\n\n{scratchpad}"},
            ]
            llm_output = self.llm(messages)
            parsed = self._parse(llm_output)

            # Always yield the LLM output for visibility
            yield f"LLM Output: {llm_output}\n"

            if parsed["type"] == "final":
                yield f"Final Answer: {parsed['content']}\n"
                if stream_final:
                    stream_messages = [
                        {
                            "role": "system",
                            "content": "Return only the final answer text.",
                        },
                        {"role": "user", "content": parsed["content"]},
                    ]
                    for chunk in self.llm.stream(stream_messages):
                        yield chunk
                return

            elif parsed["type"] == "action":
                tool_name = parsed["tool"]
                tool_input = parsed["input"]
                tool = getattr(self.tools, tool_name, None)
                if not tool:
                    raise ValueError(f"Tool {tool_name} not found")
                observation = tool(tool_input)
                step = f"Action: {tool_name}\nAction Input: {tool_input}\nObservation: {observation}\n"
                yield step
                scratchpad += f"""
                                Thought:
                                Action: {tool_name}
                                Action Input: {tool_input}
                                Observation: {observation}
                                """
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
        self.llm = FreeFlowLLM()
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
        ticker = self.yfinance.Ticker(symbol)
        data = ticker.history(period="1d")
        if data.empty:
            return "No data found."

        price = data["Close"].iloc[-1]
        open_price = data["Open"].iloc[-1]
        change = price - open_price
        percent_change = (change / open_price) * 100

        return {
            "symbol": symbol,
            "price": float(price),
            "change": float(change),
            "percent_change": float(percent_change),
        }

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
        return list(self.DDGS().news(query, max_results=3))

    def scrape_hacker_news(self):
        """
        Scrape the top trending stories from Hacker News.
        
        Fetches the current front page of Hacker News and extracts the top 3 headlines
        with their corresponding URLs.
        
        Returns:
            list[tuple]: List of (title, url) tuples for the top 3 stories.
        
        Examples:
            >>> tools = Tools()
            >>> stories = tools.scrape_hacker_news()
            >>> for title, url in stories:
            ...     print(f"{title}: {url}")
        """
        url = "https://news.ycombinator.com/"
        resp = self.requests.get(url)
        soup = self.BeautifulSoup(resp.text, "html.parser")
        links = soup.select(".titleline > a")
        return [(a.text, a["href"]) for a in links[:3]]

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
        return list(self.DDGS().text(query, max_results=1))

    def get_crypto_prices(self, symbols_csv):
        """
        Fetch current cryptocurrency prices from Binance.
        
        Retrieves real-time prices for multiple cryptocurrencies against USDT.
        
        Args:
            symbols_csv (str or list): Comma-separated string of crypto symbols
                                       (e.g., "BTC,ETH,SOL") or list of symbols
                                       (e.g., ["BTC", "ETH", "SOL"]).
        
        Returns:
            dict: Dictionary mapping symbols to their current prices in USDT.
                  Returns None for symbols that fail to fetch.
        
        Raises:
            TypeError: If symbols_csv is neither string nor list.
        
        Examples:
            >>> tools = Tools()
            >>> prices = tools.get_crypto_prices("BTC,ETH,SOL")
            >>> print(prices)
            {'BTC': 65432.10, 'ETH': 3521.45, 'SOL': 142.89}
            
            >>> prices = tools.get_crypto_prices(["BTC", "ETH"])
            >>> print(prices['BTC'])
            65432.10
        """
        if isinstance(symbols_csv, str):
            symbols = [s.strip().upper() for s in symbols_csv.split(",")]
        elif isinstance(symbols_csv, list):
            symbols = [s.strip().upper() for s in symbols_csv]
        else:
            raise TypeError("symbols_csv must be a string or list")
        prices = {}

        for symbol in symbols:
            pair = f"{symbol}USDT"
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={pair}"
            resp = self.requests.get(url)
            if resp.status_code == 200:
                prices[symbol] = float(resp.json()["price"])
            else:
                prices[symbol] = None

        return prices

    def generate_chart(self, price_dict):
        """
        Generate a QuickChart bar chart URL from price data.
        
        Creates a shareable URL that renders a bar chart visualization of prices.
        Uses the QuickChart.io service to generate charts without requiring local
        chart rendering libraries.
        
        Args:
            price_dict (dict): Dictionary mapping labels to numeric values.
                              Example: {"BTC": 65000, "ETH": 3500, "SOL": 140}
        
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
        import json

        labels = list(price_dict.keys())
        data = list(price_dict.values())

        chart_config = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{"label": "Price (USDT)", "data": data}],
            },
        }

        base_url = "https://quickchart.io/chart"
        return f"{base_url}?c={self.requests.utils.quote(json.dumps(chart_config))}"
