# download_data.py
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
symbol = "AAPL"
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
response = requests.get(url)
data = response.json()

# Extract the daily time series
ts = data.get("Time Series (Daily)")
if not ts:
    print("Error: Could not retrieve time series data.")
    exit()

# Convert to a DataFrame
df = pd.DataFrame.from_dict(ts, orient="index")
df = df.rename(columns={
    "1. open": "open",
    "2. high": "high",
    "3. low": "low",
    "4. close": "close",
    "5. volume": "volume"
})
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Save to CSV
df.to_csv("aapl_daily.csv")
print("Data saved to aapl_daily.csv")