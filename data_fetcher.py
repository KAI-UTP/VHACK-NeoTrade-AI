import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_ohlcv_data(symbol, timeframe='15m', start_date=None, end_date=None):
    """Fetch OHLCV data from OKX using CCXT with UTC timestamps."""
    exchange = ccxt.okx()
    start_timestamp = exchange.parse8601(f"{start_date}T00:00:00Z") if start_date else None
    end_timestamp = exchange.parse8601(f"{end_date}T23:59:59Z") if end_date else None
    
    all_data = []
    current_timestamp = start_timestamp
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_timestamp)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            current_timestamp = ohlcv[-1][0] + 1  # Move forward
            
            if end_timestamp and current_timestamp > end_timestamp:
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)  # Ensure UTC
    df.set_index('timestamp', inplace=True)
    
    return df

def create_features(data):
    """Compute technical indicators and features from OHLCV data."""
    if data.empty:
        return data
    
    # Price transformations
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close']).diff()
    
    # Volatility
    data['volatility_24h'] = data['returns'].rolling(24).std()
    data['volatility_7d'] = data['returns'].rolling(168).std()
    
    # Moving averages
    data['sma_50'] = data['close'].rolling(50).mean()
    data['sma_200'] = data['close'].rolling(200).mean()
    data['ema_20'] = data['close'].ewm(span=20).mean()
    
    # Momentum indicators
    data['rsi'] = compute_rsi(data['close'], 14)
    data['macd'] = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
    
    # Drop NA values from feature creation
    return data.dropna()

def compute_rsi(series, window=14):
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def preprocess_data(symbol, timeframe='15m', start_date=None, end_date=None):
    """Fetch and preprocess data with incremental updates."""
    new_data = fetch_ohlcv_data(symbol, timeframe, start_date, end_date)
    if not new_data.empty:
        print("✅ Data fetched successfully")
        features = create_features(new_data)
        # Convert timestamp to string without timezone
        features.index = features.index.tz_localize(None)  # Remove timezone info
        features.index = features.index.strftime('%Y-%m-%d %H:%M:%S')  # Format correctly

        features.to_csv("processed_data.csv")
        return features
    
