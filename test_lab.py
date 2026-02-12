from lab import BaseAgent, Tools

tools = Tools()
# ============================================================
# SPECIALIZED AGENTS
# ============================================================


class MarketIntelligenceAgent(BaseAgent):

    @property
    def system_prompt(self):
        return """
                You are a Market Intelligence Specialist.

                If the stock moves significantly, search for news explaining the movement.
                Provide a concise professional briefing.
                """


class TechTrendDiscoveryAgent(BaseAgent):

    @property
    def system_prompt(self):
        return """
                    You are a Tech Trend Discovery Agent.

                    Identify trending topics on Hacker News. For each topic, provide:
                    - The headline
                    - A brief technical summary (from web search or documentation)
                    - Why it matters for developers or product managers
                    Return a detailed list of the top 3 topics.
                """


class CryptoPortfolioVisualizer(BaseAgent):

    @property
    def system_prompt(self):
        return """
                You are a Crypto Portfolio Visualizer.

                Fetch live crypto prices and generate a QuickChart bar chart URL.
                Always return the chart URL as the final answer.
                """


# Market Agent
def test_market_agent():
    print("\n=== Market Intelligence Agent Output ===")
    market_agent = MarketIntelligenceAgent(tools)
    result = market_agent.run("AAPL")
    if hasattr(result, "__iter__") and not isinstance(result, str):
        for chunk in result:
            print(chunk, end="", flush=True)
        print()
    else:
        print(result)


# Streaming example
def test_market_agent_stream():
    print("\n=== Market Intelligence Agent Stream Output ===")
    market_agent = MarketIntelligenceAgent(tools)
    for chunk in market_agent.stream("AAPL"):
        print(chunk, end="", flush=True)
    print()


# Tech Agent
def test_tech_agent():
    print("\n=== Tech Trend Discovery Agent Output ===")
    tech_agent = TechTrendDiscoveryAgent(tools)
    result = tech_agent.run("What is trending?")
    if hasattr(result, "__iter__") and not isinstance(result, str):
        for chunk in result:
            print(chunk, end="", flush=True)
        print()
    else:
        print(result)


# Crypto Agent
def test_crypto_agent():
    print("\n=== Crypto Portfolio Visualizer Output ===")
    crypto_agent = CryptoPortfolioVisualizer(tools)
    result = crypto_agent.run("BTC,ETH,SOL")
    if hasattr(result, "__iter__") and not isinstance(result, str):
        for chunk in result:
            print(chunk, end="", flush=True)
        print()
    else:
        print(result)
