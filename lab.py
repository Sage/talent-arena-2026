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
    """

    def __call__(self, messages):
        with FreeFlowClient() as client:
            response = client.chat(messages=messages)
            return response.content

    def stream(self, messages):
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
    Generic ReAct loop reusable by all agents.
    """

    def __init__(self, llm, tools, max_iterations=5):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations

    def _tool_list(self):
        return "\n".join(
            f"- {name}"
            for name in dir(self.tools)
            if not name.startswith("_") and callable(getattr(self.tools, name))
        )

    def _parse(self, output: str):
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
    Base class for all specialized agents.
    """

    def __init__(self, tools):
        self.tools = tools
        self.llm = FreeFlowLLM()
        self.engine = ReActEngine(self.llm, tools)

    @property
    def system_prompt(self):
        raise NotImplementedError

    def run(self, user_input):
        return self.engine.run(self.system_prompt, user_input)

    def stream(self, user_input):
        return self.engine.run(self.system_prompt, user_input, stream_final=True)


# ============================================================
# TOOLS
# ============================================================


class Tools:
    """
    Toolbox for agent actions.
    """

    import requests
    import yfinance
    from bs4 import BeautifulSoup
    from ddgs import DDGS

    def get_stock_price(self, symbol):
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
        return list(self.DDGS().news(query, max_results=3))

    def scrape_hacker_news(self):
        url = "https://news.ycombinator.com/"
        resp = self.requests.get(url)
        soup = self.BeautifulSoup(resp.text, "html.parser")
        links = soup.select(".titleline > a")
        return [(a.text, a["href"]) for a in links[:3]]

    def get_project_summary(self, query):
        return list(self.DDGS().text(query, max_results=1))

    def get_crypto_prices(self, symbols_csv):
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
