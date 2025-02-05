# train_model.py
import yfinance as yf
import pandas as pd
import numpy as np
import time
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# --- STEP 1: Get S&P 500 Tickers ---
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url, header=0)
    sp500_table = tables[0]
    tickers = sp500_table['Symbol'].tolist()
    # Replace dots with hyphens (e.g., BRK.B -> BRK-B) as yfinance expects
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

# --- STEP 2: Download Historical Data for All Tickers ---
def download_sp500_data(tickers, start_date="2020-01-01", end_date="2023-01-01"):
    all_data = []
    for ticker in tickers:
        try:
            print(f"Downloading {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                continue
            data.reset_index(inplace=True)
            data["Ticker"] = ticker
            # Keep only the expected columns
            expected_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
            data = data[[col for col in expected_columns if col in data.columns]]
            all_data.append(data)
            time.sleep(0.1)  # Brief pause to avoid rate-limiting
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

# --- STEP 3: Compute Technical Indicators and Labels for Each Ticker ---
def compute_indicators_for_group(group):
    # Sort by date and reset index
    group = group.sort_index().reset_index(drop=True)
    
    # Ensure the "Close" column is present and convert it to a 1D float Series.
    if "Close" not in group.columns:
        raise ValueError("Column 'Close' not found in group")
    close_series = group["Close"].astype(float)
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    
    # Compute technical indicators using the 1D close_series
    group["RSI"] = RSIIndicator(close=close_series, window=14).rsi()
    group["SMA50"] = SMAIndicator(close=close_series, window=50).sma_indicator()
    group["SMA20"] = SMAIndicator(close=close_series, window=20).sma_indicator()
    macd = MACD(close=close_series)
    group["MACD"] = macd.macd()
    group["MACD_signal"] = macd.macd_signal()
    
    # Compute future return using the 1D series
    group["future_return"] = close_series.shift(-5) / close_series - 1
    
    # Set thresholds and assign label based on future_return.
    buy_threshold = 0.05
    sell_threshold = -0.05
    group["label"] = group.apply(
        lambda row: "Buy" if float(row["future_return"]) > buy_threshold
                    else ("Sell" if float(row["future_return"]) < sell_threshold else "Hold"),
        axis=1
    )
    return group

# --- STEP 4: Main Training Routine ---
def main():
    tickers = get_sp500_tickers()
    print(f"Number of tickers to download: {len(tickers)}")
    
    # Download data
    data = download_sp500_data(tickers, start_date="2020-01-01", end_date="2023-01-01")
    if data.empty:
        raise ValueError("No data was downloaded. Check your ticker list and date range.")
    print("Data downloaded, shape:", data.shape)
    
    # Set the Date as index and sort by date
    data.set_index("Date", inplace=True)
    data.sort_index(inplace=True)
    
    # Compute technical indicators for each ticker group
    grouped = data.groupby("Ticker").apply(compute_indicators_for_group)
    
    # After groupby, select only the columns of interest:
    features = ["RSI", "SMA50", "SMA20", "MACD", "MACD_signal"]
    # It is assumed that 'label' is also computed
    df = grouped[features + ["label"]]
    
    # Drop rows with missing values in the selected columns.
    df.dropna(inplace=True)
    print("After dropping NaNs, shape:", df.shape)
    
    if df.empty:
        raise ValueError("Data is empty after dropping NaNs. Consider using a longer historical period or adjusting window sizes.")
    
    # Define features (X) and labels (y)
    X = df[features]
    y = df["label"]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save the trained model
    joblib.dump(clf, "rf_model_sp500.pkl")
    print("Trained model saved as rf_model_sp500.pkl")

if __name__ == "__main__":
    main()