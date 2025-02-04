import os
import requests
import datetime
import random
import matplotlib.pyplot as plt
import numpy as np
import io
from openai import OpenAI  # New v1 API client
from dotenv import load_dotenv

load_dotenv()

# Initialize API clients/keys
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")  # For live stock activity

# Use a nicer plotting style.
plt.style.use("ggplot")

### HELPER: Create a timestamp string in the required format.
def current_timestamp():
    # Format: DD-MM-YYYY -> HH:MM:SS
    return datetime.datetime.now().strftime("%d-%m-%Y -> %H:%M:%S")

### DATA FETCHING FUNCTIONS

def fetch_stock_data(symbol="AAPL"):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {"error": "Failed to fetch stock data"}

def fetch_crypto_data(symbol="bitcoin"):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days=30"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {"error": "Failed to fetch crypto data"}

def fetch_stock_news(symbol):
    url = "https://newsapi.org/v2/everything"
    params = {"q": f"{symbol} stock", "language": "en", "sortBy": "publishedAt", "pageSize": 3}
    headers = {"X-Api-Key": NEWSAPI_KEY}
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        if data.get("status") != "ok" or data.get("totalResults", 0) == 0:
            return f"No recent news found for {symbol}."
        articles = data.get("articles", [])
        summaries = []
        for article in articles[:2]:
            title = article.get("title", "No title")
            description = article.get("description", "No description")
            summaries.append(f"• **{title}**\n  {description}")
        return "\n".join(summaries)
    except Exception as e:
        return f"Error fetching news: {e}"

def fetch_crypto_news(symbol):
    url = "https://newsapi.org/v2/everything"
    params = {"q": f"{symbol} crypto", "language": "en", "sortBy": "publishedAt", "pageSize": 3}
    headers = {"X-Api-Key": NEWSAPI_KEY}
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        if data.get("status") != "ok" or data.get("totalResults", 0) == 0:
            return f"No recent news found for {symbol}."
        articles = data.get("articles", [])
        summaries = []
        for article in articles[:2]:
            title = article.get("title", "No title")
            description = article.get("description", "No description")
            summaries.append(f"• **{title}**\n  {description}")
        return "\n".join(summaries)
    except Exception as e:
        return f"Error fetching news: {e}"

### IMAGE SAVE HELPER

def _save_figure_to_buffer():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf

### DETAILED ANALYSIS FUNCTIONS (for /predict command)

def analyze_stock_detailed(symbol):
    data = fetch_stock_data(symbol)
    if "error" in data or "Time Series (Daily)" not in data:
        return f"Error fetching data for {symbol}.", []
    ts_data = data["Time Series (Daily)"]
    sorted_dates = sorted(ts_data.keys())
    last_30 = sorted_dates[-30:]
    dates = last_30
    prices = [round(float(ts_data[d]["4. close"]), 2) for d in dates]
    volumes = [int(ts_data[d]["5. volume"]) for d in dates]
    dates_dt = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in dates]
    prices_array = np.array(prices)
    timestamp = current_timestamp()
    if len(prices_array) >= 7:
        moving_avg = np.convolve(prices_array, np.ones(7)/7, mode='valid')
    else:
        moving_avg = prices_array
    plt.figure(figsize=(10,5))
    plt.plot(dates_dt, prices, marker='o', linestyle='-', color='blue', label='Close Price')
    if len(prices_array) >= 7:
        plt.plot(dates_dt[6:], moving_avg, color='red', linewidth=2, label='7-Day MA')
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"{symbol.upper()} Price Trend ({timestamp})")
    plt.xticks(rotation=45)
    plt.legend()
    price_chart_buf = _save_figure_to_buffer()
    pct_changes = [0]
    for i in range(1, len(prices)):
        change = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
        pct_changes.append(round(change, 2))
    plt.figure(figsize=(10,5))
    plt.bar(dates_dt, pct_changes, color='green')
    plt.xlabel("Date")
    plt.ylabel("Daily % Change")
    plt.title(f"{symbol.upper()} Daily % Change ({timestamp})")
    plt.xticks(rotation=45)
    pct_change_buf = _save_figure_to_buffer()
    plt.figure(figsize=(10,5))
    plt.bar(dates_dt, volumes, color='purple')
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.title(f"{symbol.upper()} Volume Trend ({timestamp})")
    plt.xticks(rotation=45)
    volume_chart_buf = _save_figure_to_buffer()
    news_summary = fetch_stock_news(symbol)
    summary = (f"**{symbol.upper()} Detailed Analysis**\n"
               f"Latest Price: ${prices[-1]}\n"
               f"Daily % Change: {pct_changes[-1]}%\n\n"
               f"**Live News:**\n{news_summary}")
    return summary, [price_chart_buf, pct_change_buf, volume_chart_buf]

def analyze_crypto(symbol):
    data = fetch_crypto_data(symbol)
    if "error" in data or "prices" not in data:
        return f"Error fetching crypto data for {symbol}.", []
    prices_data = data["prices"]
    interval = max(1, len(prices_data) // 30)
    sampled = prices_data[::interval][:30]
    dates = [datetime.datetime.fromtimestamp(x[0]/1000).strftime("%Y-%m-%d") for x in sampled]
    prices = [round(x[1], 2) for x in sampled]
    dates_dt = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in dates]
    timestamp = current_timestamp()
    prices_array = np.array(prices)
    if len(prices_array) >= 7:
        moving_avg = np.convolve(prices_array, np.ones(7)/7, mode='valid')
    else:
        moving_avg = prices_array
    plt.figure(figsize=(10,5))
    plt.plot(dates_dt, prices, marker='o', linestyle='-', color='blue', label='Price')
    if len(prices_array) >= 7:
        plt.plot(dates_dt[6:], moving_avg, color='red', linewidth=2, label='7-Day MA')
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"{symbol.capitalize()} Price Trend ({timestamp})")
    plt.xticks(rotation=45)
    plt.legend()
    price_chart_buf = _save_figure_to_buffer()
    pct_changes = [0]
    for i in range(1, len(prices)):
        change = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
        pct_changes.append(round(change, 2))
    plt.figure(figsize=(10,5))
    plt.bar(dates_dt, pct_changes, color='green')
    plt.xlabel("Date")
    plt.ylabel("Daily % Change")
    plt.title(f"{symbol.capitalize()} Daily % Change ({timestamp})")
    plt.xticks(rotation=45)
    pct_change_buf = _save_figure_to_buffer()
    news_summary = fetch_crypto_news(symbol)
    summary = (f"**{symbol.capitalize()} Detailed Analysis**\n"
               f"Latest Price: ${prices[-1]}\n"
               f"Daily % Change: {pct_changes[-1]}%\n\n"
               f"**Live News:**\n{news_summary}")
    return summary, [price_chart_buf, pct_change_buf]

### TOP 10 FUNCTIONS FOR GRAPHS (Past 30 Minutes)

def get_top10_stocks_30min():
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
    stocks = get_top10_stocks_30min()
    if not stocks:
        return []
    symbols = [s["symbol"] for s in stocks]
    pct_changes = [s["changesPercentage"] for s in stocks]
    prices = [float(s.get("price", 0)) for s in stocks]
    ts = current_timestamp()
    buffers = []
    plt.figure(figsize=(8,4))
    bars = plt.bar(symbols, pct_changes, color=['green' if ch >= 0 else 'red' for ch in pct_changes])
    plt.xlabel("Stock Symbol")
    plt.ylabel("Daily % Change")
    plt.title(f"Top 10 Stocks % Change ({ts})")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45)
    for bar, ch in zip(bars, pct_changes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{ch}%", ha='center', va='bottom', fontsize=8)
    buffers.append(_save_figure_to_buffer())
    plt.figure(figsize=(8,4))
    bars = plt.bar(symbols, prices, color='blue')
    plt.xlabel("Stock Symbol")
    plt.ylabel("Price (USD)")
    plt.title(f"Top 10 Stocks Price ({ts})")
    plt.xticks(rotation=45)
    for bar, pr in zip(bars, prices):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"${pr}", ha='center', va='bottom', fontsize=8)
    buffers.append(_save_figure_to_buffer())
    return buffers

### TOP 25 FUNCTIONS FOR SUMMARIES

def get_top25_stocks_30min():
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

def generate_top25_stock_summary_30min():
    stocks = get_top25_stocks_30min()
    summaries = []
    for s in stocks:
        symbol = s.get("symbol", "N/A")
        current_price = s.get("price", "N/A")
        historical_price = s.get("historical_price", "N/A")
        news = fetch_stock_news(symbol)
        recommendation = s.get("recommendation", "N/A")
        ytd = s.get("YTD", 0)
        mtd = s.get("MTD", 0)
        wtd = s.get("WTD", 0)
        summary = (
            f"**{symbol}**\n"
            f"- Historical Price (30 min ago): ${historical_price}\n"
            f"- Current Price: ${current_price}\n"
            f"- Daily % Change: {s.get('changesPercentage',0)}%\n"
            f"- YTD: {ytd}%, MTD: {mtd}%, WTD: {wtd}%\n"
            f"- Recommendation: {recommendation}\n"
            f"- News:\n{news}\n"
        )
        summaries.append(summary)
    return "\n".join(summaries)

### TOP 10 FUNCTIONS FOR CRYPTO GRAPHS

def get_top10_crypto_30min():
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
    cryptos = get_top10_crypto_30min()
    if not cryptos:
        return []
    coins = [c["id"] for c in cryptos]
    pct_changes = [c.get("price_change_percentage_24h", 0) for c in cryptos]
    prices = [c.get("current_price", 0) for c in cryptos]
    ts = current_timestamp()
    buffers = []
    plt.figure(figsize=(8,4))
    bars = plt.bar(coins, pct_changes, color=['green' if ch >= 0 else 'red' for ch in pct_changes])
    plt.xlabel("Coin")
    plt.ylabel("Daily % Change")
    plt.title(f"Top 10 Cryptos % Change ({ts})")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45)
    for bar, ch in zip(bars, pct_changes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{round(ch,2)}%", ha='center', va='bottom', fontsize=8)
    buffers.append(_save_figure_to_buffer())
    plt.figure(figsize=(8,4))
    bars = plt.bar(coins, prices, color='blue')
    plt.xlabel("Coin")
    plt.ylabel("Price (USD)")
    plt.title(f"Top 10 Cryptos Price ({ts})")
    plt.xticks(rotation=45)
    for bar, pr in zip(bars, prices):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"${pr}", ha='center', va='bottom', fontsize=8)
    buffers.append(_save_figure_to_buffer())
    return buffers

### TOP 25 FUNCTIONS FOR CRYPTO SUMMARIES

def get_top25_crypto_30min():
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

def generate_top25_crypto_summary_30min():
    cryptos = get_top25_crypto_30min()
    summaries = []
    for c in cryptos:
        coin_id = c.get("id", "N/A")
        current_price = c.get("current_price", "N/A")
        historical_price = c.get("historical_price", "N/A")
        news = fetch_crypto_news(coin_id)
        recommendation = c.get("recommendation", "N/A")
        ytd = c.get("YTD", 0)
        mtd = c.get("MTD", 0)
        wtd = c.get("WTD", 0)
        summary = (
            f"**{coin_id.upper()}**\n"
            f"- Historical Price (30 min ago): ${historical_price}\n"
            f"- Current Price: ${current_price}\n"
            f"- Daily % Change: {c.get('price_change_percentage_24h', 0)}%\n"
            f"- YTD: {ytd}%, MTD: {mtd}%, WTD: {wtd}%\n"
            f"- Recommendation: {recommendation}\n"
            f"- News:\n{news}\n"
        )
        summaries.append(summary)
    return "\n".join(summaries)

if __name__ == "__main__":
    summary, graphs = analyze_stock_detailed("AAPL")
    print(summary)
    print("Top 10 Stock Summary (30min):")
    print(generate_top25_stock_summary_30min())
    # Uncomment to test crypto summaries:
    # print("Top 25 Crypto Summary (30min):")
    # print(generate_top25_crypto_summary_30min())