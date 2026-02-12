"""
utils.py

Skeletons for AI agents and a set of tools for workshop students.

Agents:
- MarketIntelligenceAgent
- TechTrendDiscoveryAgent
- CryptoPortfolioVisualizer

Tools:
- Tools class with methods for price fetching, news searching, web scraping, and chart generation.

Example usage:
    from utils import Tools, MarketIntelligenceAgent
    tools = Tools()
    agent = MarketIntelligenceAgent(tools)
    agent.run('AAPL')
"""


class Tools:
    """
    Toolbox for agent actions.

    Example:
        tools = Tools()
        price = tools.get_stock_price('AAPL')
        news = tools.search_news('AAPL')
        chart_url = tools.generate_chart({'BTC': 65000, 'ETH': 3500, 'SOL': 140})
    """

    import requests
    import yfinance
    from bs4 import BeautifulSoup
    from ddgs import DDGS

    def get_stock_price(self, symbol):
        """
        Fetch current stock price and % change using yfinance.
        Returns: dict with 'price', 'change', 'percent_change'.
        """
        ticker = self.yfinance.Ticker(symbol)
        data = ticker.history(period="1d")
        if data.empty:
            return None
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

    def search_news(self, query, max_results=5):
        """
        Search for recent news using duckduckgo-search.
        Returns: list of news dicts.
        """
        return list(self.DDGS().news(query, max_results=max_results))

    def scrape_hacker_news(self, top_n=3):
        """
        Scrape top Hacker News headlines using requests and BeautifulSoup.
        Returns: list of (title, url) tuples.
        """
        url = "https://news.ycombinator.com/"
        resp = self.requests.get(url)
        soup = self.BeautifulSoup(resp.text, "html.parser")
        links = soup.select(".titleline > a")
        return [(a.text, a["href"]) for a in links[:top_n]]

    def get_project_summary(self, top_hn_story, max_results=1):
        """
        Fetch project summary or documentation using duckduckgo-search.
        Returns: list of search result dicts.
        """
        return list(self.DDGS().text(top_hn_story, max_results=max_results))

    def get_crypto_prices(self, symbols):
        """
        Fetch current crypto prices from Binance public API.
        Returns: dict of symbol: price.
        """
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
        Generate QuickChart URL for price data.
        Returns: chart image URL.
        """
        labels = list(price_dict.keys())
        data = list(price_dict.values())
        chart_config = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{"label": "Price (USDT)", "data": data}],
            },
        }
        import json

        base_url = "https://quickchart.io/chart"
        url = f"{base_url}?c={self.requests.utils.quote(json.dumps(chart_config))}"
        return url


class MarketIntelligenceAgent:
    """
    The Market Intelligence Specialist

    Problem:
        Individual investors often see a stock price move but struggle to quickly find the specific news event or market sentiment causing the change.

    Example:
        agent = MarketIntelligenceAgent(tools)
        agent.run('AAPL')
    """

    def __init__(self, tools):
        self.tools = tools

    def run(self, symbol):
        """
        ReAct Loop:
        1. Thought: The user is asking about Apple (AAPL). I first need to check its current price and performance today.
        2. Action: Pull the current price and % change.
        3. Observation: AAPL is down 3% today.
        4. Thought: Now that I see a significant drop, I need to search for recent news from the last 24 hours to explain this movement.
        5. Action: Search for "AAPL stock price drop news today."
        6. Final Answer: Synthesize the price data and the specific news (e.g., a supply chain report) into a concise briefing.
        """
        # TODO: Implement ReAct loop here
        pass


class TechTrendDiscoveryAgent:
    """
    The Tech Trend Discovery Agent

    Problem:
        For product managers and developers, keeping up with trending topics on platforms like Hacker News is time-consuming. They need context beyond just a headline.

    Example:
        agent = TechTrendDiscoveryAgent(tools)
        agent.run()
    """

    def __init__(self, tools):
        self.tools = tools

    def run(self):
        """
        ReAct Loop:
        1. Thought: I need to identify what the tech community is currently focused on. I will start by grabbing the top 3 headlines from Hacker News.
        2. Action: Scrape the site.
        3. Observation: The top story is "New Rust Framework for WebAssembly."
        4. Thought: This framework is trending, but the user may not know what it is. I should find a brief technical summary of this specific framework.
        5. Action: Search to find the official documentation or a summary of that specific project.
        6. Final Answer: Present the trending topic along with a "Why it matters" summary.
        """
        # TODO: Implement ReAct loop here
        pass


class CryptoPortfolioVisualizer:
    """
    The Crypto Portfolio Visualizer

    Problem:
        Crypto traders need to monitor multiple assets simultaneously and prefer visual summaries of current market state over raw text logs.

    Example:
        agent = CryptoPortfolioVisualizer(tools)
        agent.run(['BTC', 'ETH', 'SOL'])
    """

    def __init__(self, tools):
        self.tools = tools

    def run(self, symbols):
        """
        ReAct Loop:
        1. Thought: The user wants a visual status report for BTC, ETH, and SOL. I need their current USDT prices from a live exchange.
        2. Action: Call the Binance public ticker endpoint for the three symbols.
        3. Observation: BTC: 65k, ETH: 3.5k, SOL: 140.
        4. Thought: Now that I have the numerical data, I will format it into a JSON structure that QuickChart can use to generate a bar chart.
        5. Action: Generate a QuickChart URL containing the price data.
        6. Final Answer: Return the live chart image URL to the user for an instant visual briefing.
        """
        # TODO: Implement ReAct loop here
        pass
