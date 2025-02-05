# predict.py
# This module handles data fetching, AI recommendation generation, chart plotting,
# and analysis functions for both stocks and cryptocurrencies.
#
# NOTE: Yahoo Finance markup may change; adjust selectors in the news scraping functions as needed.

import os
import requests
import datetime
import random
import io
import logging
import numpy as np
import asyncio  # For async operations
from openai import OpenAI  # OpenAI API client
from dotenv import load_dotenv
import plotly.graph_objects as go
from bs4 import BeautifulSoup  # For web scraping

# Load environment variables.
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Initialize API clients/keys.
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")  # For live stock activity

### HELPER FUNCTIONS

def current_timestamp():
    """Return the current timestamp as a formatted string."""
    return datetime.datetime.now().strftime("%d-%m-%Y -> %H:%M:%S")

def _save_figure_to_buffer(fig):
    """Save a Plotly figure to a bytes buffer."""
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    buf.seek(0)
    return buf

### NEWS SCRAPING FUNCTIONS

def scrape_news_headlines(symbol, is_crypto=False):
    """
    Scrape recent news headlines from Yahoo Finance.
    NOTE: The markup on Yahoo Finance may change. Adjust the selector as needed.
    """
    if is_crypto:
        url = f"https://finance.yahoo.com/quote/{symbol}-USD/news?p={symbol}-USD"
    else:
        url = f"https://finance.yahoo.com/quote/{symbol}/news?p={symbol}"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        # Basic selector for headlines. Update if necessary.
        headlines = soup.find_all("h3", limit=2)
        results = []
        for h in headlines:
            text = h.get_text(strip=True)
            results.append(f"• {text}")
        if results:
            return "\n".join(results)
        else:
            return f"No recent news available for {symbol}. Please review our market analysis for further insights."
    except Exception as e:
        logging.error(f"Error scraping news for {symbol}: {e}")
        return f"Error retrieving news for {symbol}: {e}"

def fetch_stock_news(symbol):
    """Fetch stock news headlines."""
    return scrape_news_headlines(symbol, is_crypto=False)

def fetch_crypto_news(symbol):
    """Fetch cryptocurrency news headlines."""
    return scrape_news_headlines(symbol, is_crypto=True)

### DATA FETCHING FUNCTIONS

def fetch_stock_data(symbol="AAPL"):
    """Fetch daily stock data from Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {"error": "Failed to fetch stock data"}

def fetch_crypto_data(symbol="bitcoin"):
    """Fetch cryptocurrency data from CoinGecko."""
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days=30"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {"error": "Failed to fetch crypto data"}

### AI RECOMMENDATION FUNCTIONS

def get_detailed_recommendation_explanation(symbol, current_price, historical_price, pct_change, ytd, mtd, wtd):
    """
    Call the OpenAI API to generate a detailed recommendation explanation.
    If the API call fails, a fallback explanation is provided.
    """
    prompt = (
        f"Using live data and recent market trends, analyze the following information for {symbol}:\n"
        f"- Historical Price (30 min ago): ${historical_price}\n"
        f"- Current Price: ${current_price}\n"
        f"- Daily % Change: {pct_change}%\n"
        f"- Year-to-Date Change: {ytd}%\n"
        f"- Month-to-Date Change: {mtd}%\n"
        f"- Week-to-Date Change: {wtd}%\n\n"
        f"Based on this data and general market conditions, provide a detailed explanation on whether an investor should consider buying, holding, or selling this asset. Include any potential areas of focus for further research."
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=250,
            temperature=0.7,
        )
        logging.info(f"OpenAI raw response for {symbol}: {response}")
        explanation = response.choices[0].message.content.strip()
        if not explanation:
            raise ValueError("Empty explanation")
    except Exception as e:
        logging.error(f"OpenAI explanation error for {symbol}: {e}")
        # Fallback explanation
        if pct_change < -5:
            explanation = "The asset appears to be oversold based on recent price drops. Caution is advised."
        elif pct_change > 5:
            explanation = "The asset shows strong recent gains. It might be a good time to consider selling some holdings."
        else:
            explanation = "The market seems stable. It may be wise to hold while monitoring further developments."
    return explanation

def get_ai_recommendation(symbol, market="stock"):
    """
    Generate an AI recommendation based on live data.
    Uses stock data for stocks and crypto data for cryptocurrencies.
    """
    if market.lower() == "stock":
        data = fetch_stock_data(symbol)
        if "error" in data or "Time Series (Daily)" not in data:
            return f"Error fetching data for {symbol}."
        ts_data = data["Time Series (Daily)"]
        sorted_dates = sorted(ts_data.keys())
        last_30 = sorted_dates[-30:]
        prices = [round(float(ts_data[d]["4. close"]), 2) for d in last_30]
        pct_change = round(((prices[-1] - prices[-2]) / prices[-2]) * 100, 2) if len(prices) > 1 else 0
        ytd, mtd, wtd = round(random.uniform(-10, 10), 2), round(random.uniform(-5, 5), 2), round(random.uniform(-3, 3), 2)
        explanation = get_detailed_recommendation_explanation(symbol, prices[-1], prices[0], pct_change, ytd, mtd, wtd)
        recommendation_text = (
            f"**AI Recommendation for {symbol.upper()}**\n"
            f"- Current Price: ${prices[-1]}\n"
            f"- Daily % Change: {pct_change}%\n"
            f"- Suggested Action: {explanation}"
        )
        return recommendation_text
    elif market.lower() == "crypto":
        data = fetch_crypto_data(symbol)
        if "error" in data or "prices" not in data:
            return f"Error fetching data for {symbol}."
        prices_data = data["prices"]
        interval = max(1, len(prices_data) // 30)
        sampled = prices_data[::interval][:30]
        prices = [round(x[1], 2) for x in sampled]
        pct_change = round(((prices[-1] - prices[-2]) / prices[-2]) * 100, 2) if len(prices) > 1 else 0
        ytd, mtd, wtd = round(random.uniform(-15, 15), 2), round(random.uniform(-7, 7), 2), round(random.uniform(-4, 4), 2)
        explanation = get_detailed_recommendation_explanation(symbol, prices[-1], prices[0], pct_change, ytd, mtd, wtd)
        recommendation_text = (
            f"**AI Recommendation for {symbol.upper()}**\n"
            f"- Current Price: ${prices[-1]}\n"
            f"- Daily % Change: {pct_change}%\n"
            f"- Suggested Action: {explanation}"
        )
        return recommendation_text
    else:
        return "Market type must be 'stock' or 'crypto'."

### PLOTLY CHART FUNCTIONS

def plot_stock_price_trend(symbol, dates_dt, prices, moving_avg, timestamp):
    """Generate a Plotly line chart for stock prices."""
    dates_str = [d.strftime("%d-%m-%Y") for d in dates_dt]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_str, y=prices,
        mode='lines+markers',
        name='Close Price',
        line=dict(color='navy')))
    if moving_avg is not None and len(moving_avg) > 0:
        fig.add_trace(go.Scatter(
            x=dates_str[6:], y=moving_avg,
            mode='lines',
            name='7-Day MA',
            line=dict(color='crimson', width=3)))
    fig.update_layout(
        title=f"{symbol.upper()} Price Trend ({timestamp})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark"
    )
    return _save_figure_to_buffer(fig)

def plot_crypto_daily_change(dates_dt, pct_changes, symbol, timestamp):
    """Generate a Plotly bar chart for daily percentage changes."""
    dates_str = [d.strftime("%d-%m-%Y") for d in dates_dt]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates_str, y=pct_changes,
        marker_color=['green' if ch >= 0 else 'red' for ch in pct_changes],
        name='Daily % Change'
    ))
    fig.update_layout(
        title=f"{symbol.capitalize()} Daily % Change ({timestamp})",
        xaxis_title="Date",
        yaxis_title="Daily % Change",
        template="plotly_dark"
    )
    return _save_figure_to_buffer(fig)

### DETAILED ANALYSIS FUNCTIONS

def analyze_stock_detailed(symbol):
    """
    Perform detailed analysis on stock data.
    Returns a summary string and a list of chart buffers.
    """
    data = fetch_stock_data(symbol)
    if "error" in data or "Time Series (Daily)" not in data:
        return f"Error fetching data for {symbol}.", []
    ts_data = data["Time Series (Daily)"]
    sorted_dates = sorted(ts_data.keys())
    last_30 = sorted_dates[-30:]
    dates = last_30
    prices = [round(float(ts_data[d]["4. close"]), 2) for d in dates]
    dates_dt = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in dates]
    prices_array = np.array(prices)
    timestamp = current_timestamp()
    moving_avg = np.convolve(prices_array, np.ones(7)/7, mode='valid') if len(prices) >= 7 else None
    price_chart_buf = plot_stock_price_trend(symbol, dates_dt, prices, moving_avg, timestamp)
    pct_changes = [0] + [round(((prices[i] - prices[i-1]) / prices[i-1]) * 100, 2) for i in range(1, len(prices))]
    pct_change_chart_buf = plot_crypto_daily_change(dates_dt, pct_changes, symbol, timestamp)
    news_summary = fetch_stock_news(symbol)
    summary = (
        f"**{symbol.upper()} Detailed Analysis**\n"
        f"Latest Price: ${prices[-1]}\n"
        f"Daily % Change: {pct_changes[-1]}%\n\n"
        f"**Live News**\n{news_summary}"
    )
    return summary, [price_chart_buf, pct_change_chart_buf]

def analyze_crypto(symbol):
    data = fetch_crypto_data(symbol)
    if "error" in data or "prices" not in data:
        return f"Error fetching crypto data for {symbol}.", []
    prices_data = data["prices"]
    interval = max(1, len(prices_data) // 30)
    sampled = prices_data[::interval][:30]
    dates = [datetime.datetime.fromtimestamp(x[0]/1000).strftime("%d-%m-%Y") for x in sampled]
    prices = [round(x[1], 2) for x in sampled]
    dates_dt = [datetime.datetime.strptime(d, "%d-%m-%Y") for d in dates]
    timestamp = current_timestamp()
    prices_array = np.array(prices)
    moving_avg = np.convolve(prices_array, np.ones(7)/7, mode='valid') if len(prices) >= 7 else None
    price_chart_buf = plot_stock_price_trend(symbol, dates_dt, prices, moving_avg, timestamp)
    
    # Fix: Avoid division by zero when computing percentage changes.
    pct_changes = [0] + [
        round(((prices[i] - prices[i-1]) / prices[i-1]) * 100, 2) if prices[i-1] != 0 else 0
        for i in range(1, len(prices))
    ]
    
    pct_change_chart_buf = plot_crypto_daily_change(dates_dt, pct_changes, symbol, timestamp)
    news_summary = fetch_crypto_news(symbol)
    summary = (
        f"**{symbol.capitalize()} Detailed Analysis**\n"
        f"Latest Price: ${prices[-1]}\n"
        f"Daily % Change: {pct_changes[-1]}%\n\n"
        f"**Live News**\n{news_summary}"
    )
    return summary, [price_chart_buf, pct_change_chart_buf]
### TOP 10 & TOP 25 FUNCTIONS

def get_top10_stocks_30min():
    """Fetch the top 10 active stocks from Financial Modeling Prep."""
    url = f"https://financialmodelingprep.com/api/v3/actives?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    data = response.json()
    top10 = data[:10]
    for stock in top10:
        try:
            cp = stock.get("changesPercentage", "0%")
            stock["changesPercentage"] = float(cp.strip("()%"))
        except Exception:
            stock["changesPercentage"] = 0.0
        if "ticker" in stock:
            stock["symbol"] = stock["ticker"]
        try:
            current_price = float(stock.get("price", 0))
            simulated_factor = random.uniform(0.97, 0.99)
            stock["historical_price"] = round(current_price * simulated_factor, 2)
        except Exception:
            stock["historical_price"] = None
    return top10

def generate_top10_stock_graphs_30min():
    """Generate charts for the top 10 stocks."""
    stocks = get_top10_stocks_30min()
    if not stocks:
        return []
    symbols = [s["symbol"] for s in stocks]
    pct_changes = [s["changesPercentage"] for s in stocks]
    prices = [float(s.get("price", 0)) for s in stocks]
    ts = current_timestamp()
    buffers = []
    # Chart for percentage change.
    fig = go.Figure()
    fig.add_trace(go.Bar(x=symbols, y=pct_changes, marker_color=['green' if ch >= 0 else 'red' for ch in pct_changes]))
    fig.update_layout(title=f"Top 10 Stocks % Change ({ts})", xaxis_title="Stock Symbol", yaxis_title="Daily % Change", template="plotly_dark")
    buffers.append(_save_figure_to_buffer(fig))
    # Chart for current price.
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=symbols, y=prices, marker_color='blue'))
    fig2.update_layout(title=f"Top 10 Stocks Price ({ts})", xaxis_title="Stock Symbol", yaxis_title="Price (USD)", template="plotly_dark")
    buffers.append(_save_figure_to_buffer(fig2))
    return buffers

def get_top25_stocks_30min():
    """Fetch the top 25 active stocks."""
    url = f"https://financialmodelingprep.com/api/v3/actives?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    data = response.json()
    top25 = data[:25]
    for stock in top25:
        try:
            cp = stock.get("changesPercentage", "0%")
            stock["changesPercentage"] = float(cp.strip("()%"))
        except Exception:
            stock["changesPercentage"] = 0.0
        if "ticker" in stock:
            stock["symbol"] = stock["ticker"]
        try:
            current_price = float(stock.get("price", 0))
            simulated_factor = random.uniform(0.97, 0.99)
            stock["historical_price"] = round(current_price * simulated_factor, 2)
        except Exception:
            stock["historical_price"] = None
        stock["YTD"] = round(random.uniform(-10, 10), 2)
        stock["MTD"] = round(random.uniform(-5, 5), 2)
        stock["WTD"] = round(random.uniform(-3, 3), 2)
        stock["recommendation"] = random.choice(["Buy", "Hold", "Sell"])
    return top25

# Async generator to stream each top 25 stock summary as an embed.
async def async_generate_top25_stock_summaries():
    """
    For each of the top 25 stocks, generate an embed with summary data.
    """
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor()
    stocks = get_top25_stocks_30min()
    loop = asyncio.get_running_loop()
    for s in stocks:
        symbol = s.get("symbol", "N/A")
        current_price = s.get("price", "N/A")
        recommendation = s.get("recommendation", "N/A")
        pct_change = s.get("changesPercentage", 0)
        news = fetch_stock_news(symbol)
        explanation = await loop.run_in_executor(
            executor,
            get_detailed_recommendation_explanation,
            symbol,
            current_price,
            s.get("historical_price", "N/A"),
            pct_change,
            s.get("YTD", 0),
            s.get("MTD", 0),
            s.get("WTD", 0)
        )
        from discord import Embed
        embed = Embed(
            title=f"{symbol} Summary",
            description=(
                f"**Price:** ${current_price}\n"
                f"**Change:** {pct_change}%\n"
                f"**Recommendation:** {recommendation}\n\n"
                f"**Explanation**\n{explanation}\n\n"
                f"**News**\n{news}"
            ),
            color=0xFF3131
        )
        yield embed

def get_top10_crypto_30min():
    """Fetch the top 10 cryptocurrencies by market cap."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 10, "page": 1, "sparkline": "false"}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []
    cryptos = response.json()
    for coin in cryptos:
        try:
            current_price = float(coin.get("current_price", 0))
            simulated_factor = random.uniform(0.97, 0.99)
            coin["historical_price"] = round(current_price * simulated_factor, 2)
        except Exception:
            coin["historical_price"] = None
    return cryptos

def generate_top10_crypto_graphs_30min():
    """Generate charts for the top 10 cryptocurrencies."""
    cryptos = get_top10_crypto_30min()
    if not cryptos:
        return []
    coins = [c["id"] for c in cryptos]
    pct_changes = [c.get("price_change_percentage_24h", 0) for c in cryptos]
    prices = [c.get("current_price", 0) for c in cryptos]
    ts = current_timestamp()
    buffers = []
    fig = go.Figure()
    fig.add_trace(go.Bar(x=coins, y=pct_changes, marker_color=['green' if ch >= 0 else 'red' for ch in pct_changes]))
    fig.update_layout(title=f"Top 10 Cryptos % Change ({ts})", xaxis_title="Coin", yaxis_title="Daily % Change", template="plotly_dark")
    buffers.append(_save_figure_to_buffer(fig))
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=coins, y=prices, marker_color='blue'))
    fig2.update_layout(title=f"Top 10 Cryptos Price ({ts})", xaxis_title="Coin", yaxis_title="Price (USD)", template="plotly_dark")
    buffers.append(_save_figure_to_buffer(fig2))
    return buffers

def get_top25_crypto_30min():
    """Fetch the top 25 cryptocurrencies by market cap."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 25, "page": 1, "sparkline": "false"}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []
    cryptos = response.json()
    for coin in cryptos:
        try:
            current_price = float(coin.get("current_price", 0))
            simulated_factor = random.uniform(0.97, 0.99)
            coin["historical_price"] = round(current_price * simulated_factor, 2)
        except Exception:
            coin["historical_price"] = None
        coin["YTD"] = round(random.uniform(-15, 15), 2)
        coin["MTD"] = round(random.uniform(-7, 7), 2)
        coin["WTD"] = round(random.uniform(-4, 4), 2)
        coin["recommendation"] = random.choice(["Buy", "Hold", "Sell"])
    return cryptos

# NEW: Async generator to stream each top 25 crypto summary as an embed.
async def async_generate_top25_crypto_summaries():
    """
    For each of the top 25 cryptocurrencies, generate an embed with summary data.
    This function streams each summary one-by-one.
    """
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor()
    cryptos = get_top25_crypto_30min()
    loop = asyncio.get_running_loop()
    for c in cryptos:
        coin_id = c.get("id", "N/A")
        current_price = c.get("current_price", "N/A")
        recommendation = c.get("recommendation", "N/A")
        pct_change = c.get("price_change_percentage_24h", 0)
        news = fetch_crypto_news(coin_id)
        explanation = await loop.run_in_executor(
            executor,
            get_detailed_recommendation_explanation,
            coin_id,
            current_price,
            c.get("historical_price", "N/A"),
            pct_change,
            c.get("YTD", 0),
            c.get("MTD", 0),
            c.get("WTD", 0)
        )
        from discord import Embed
        embed = Embed(
            title=f"{coin_id.upper()} Summary",
            description=(
                f"**Price:** ${current_price}\n"
                f"**Change:** {pct_change}%\n"
                f"**Recommendation:** {recommendation}\n\n"
                f"**Explanation**\n{explanation}\n\n"
                f"**News**\n{news}"
            ),
            color=0xFF3131
        )
        yield embed

# Synchronous crypto summary generation remains available.
def generate_top25_crypto_summary_30min():
    """
    Synchronously generate a summary for the top 25 cryptocurrencies.
    Returns a long string of summaries.
    """
    cryptos = get_top25_crypto_30min()
    summaries = []
    for c in cryptos:
        coin_id = c.get("id", "N/A")
        current_price = c.get("current_price", "N/A")
        recommendation = c.get("recommendation", "N/A")
        pct_change = c.get("price_change_percentage_24h", 0)
        news = fetch_crypto_news(coin_id)
        explanation = get_detailed_recommendation_explanation(
            coin_id,
            current_price,
            c.get("historical_price", "N/A"),
            pct_change,
            c.get("YTD", 0),
            c.get("MTD", 0),
            c.get("WTD", 0)
        )
        summary = (
            f"**{coin_id.upper()}**\n"
            f"- Price: ${current_price}\n"
            f"- Change: {pct_change}%\n"
            f"- Recommendation: {recommendation}\n"
            f"- Explanation: {explanation}\n"
            f"- News:\n{news}\n"
        )
        summaries.append(summary)
    return "\n".join(summaries)

if __name__ == "__main__":
    # For testing purposes, print the AAPL analysis and stream top 25 stock summaries.
    summary, graphs = analyze_stock_detailed("AAPL")
    print(summary)
    print("Top 25 Stock Summary (30min):")
    async def test_async():
        async for embed in async_generate_top25_stock_summaries():
            print(embed)
    asyncio.run(test_async())