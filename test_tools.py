import pytest

from utils import Tools


@pytest.fixture
def tools():
    return Tools()


def test_stock_price(tools: Tools):
    result = tools.get_stock_price("AAPL")
    print(result, "\n")
    assert (
        result["symbol"] == "AAPL"
        and "price" in result
        and "change" in result
        and "percent_change" in result
    )


def test_news(tools: Tools):
    news = tools.search_news("AAPL stock price drop news today")
    print(news[0], "\n")
    assert isinstance(news, list)


def test_hacker_news(tools: Tools):
    headlines = tools.scrape_hacker_news()
    print(headlines, "\n")
    assert isinstance(headlines, list)
    assert len(headlines) <= 3


def test_project_summary(tools: Tools):
    summary = tools.get_project_summary("New Rust Framework for WebAssembly")
    print(summary, "\n")
    assert isinstance(summary, list)


def test_crypto_prices(tools: Tools):
    prices = tools.get_crypto_prices(["BTC", "ETH", "SOL"])
    print(prices, "\n")
    assert isinstance(prices, dict)
    for symbol in ["BTC", "ETH", "SOL"]:
        assert symbol in prices


def test_chart(tools: Tools):
    prices = {"BTC": 65000, "ETH": 3500, "SOL": 140}
    chart_url = tools.generate_chart(prices)
    print(chart_url, "\n")
    assert isinstance(chart_url, str)
    assert chart_url.startswith("https://quickchart.io/chart")
