import pandas as pd
import numpy as np
import os
import time
import live_data_fetcher
import asyncio
from data_fetcher import preprocess_data, create_features
from cnn_feature_extractor import build_cnn, update_cnn, save_model as save_cnn, load_model as load_cnn
from hmm_regime_detector import train_hmm, save_hmm, load_hmm, update_hmm
from hybrid_model import build_model, update_model, save_model as save_hybrid, load_model as load_hybrid
from signal_generator import generate_signals
from sklearn.preprocessing import StandardScaler
from live_data_fetcher import main_collect_live_data, input_listener

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

data_live = pd.DataFrame()
signals = pd.DataFrame()

def load_cnn_model(path):
    """Load CNN model if it exists; otherwise, raise an error."""
    if model_exists(path):
        cnn = load_cnn(path)
        if cnn.optimizer is None:
            cnn.compile(optimizer='adam', loss='mse')
        print("✅ CNN model loaded successfully.")
        return cnn
    raise FileNotFoundError(f"❌ CNN model not found at {path}")

def load_hmm_model(path):
    """Load HMM model if it exists; otherwise, raise an error."""
    if model_exists(path):
        print("✅ HMM model loaded successfully.")
        return load_hmm(path)
    raise FileNotFoundError(f"❌ HMM model not found at {path}")

def load_hybrid_model(path):
    """Load Hybrid model if it exists; otherwise, raise an error."""
    if model_exists(path):
        hybrid = load_hybrid(path)
        if hybrid.optimizer is None:
            hybrid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("✅ Hybrid model loaded successfully.")
        return hybrid
    raise FileNotFoundError(f"❌ Hybrid model not found at {path}")

def model_exists(model_path):
    """Check if the model file exists."""
    return os.path.exists(model_path) and os.path.getsize(model_path) > 0

from sklearn.preprocessing import StandardScaler

def generate_trade_probabilities_live(data, num_features, cnn, hmm, hybrid):
    """
    Generate trade probabilities using CNN, HMM, and the hybrid model.

    Parameters:
    - data (pd.DataFrame): Preprocessed OHLCV data.
    - num_features (int): Number of features in the dataset.
    - cnn (model): Pre-trained CNN model.
    - hmm (model): Pre-trained HMM model.
    - hybrid (model): Pre-trained hybrid model.

    Returns:
    - probs (np.ndarray): Probability predictions from the hybrid model.
    - aligned_prices (pd.Series): Price data aligned with predictions.
    """

    if data.empty:
        print("❌ No data available!")
        return None, None

    window_size = 24
    truncated_len = (len(data) // window_size) * window_size
    data_trunc = data.iloc[:truncated_len]

    # ✅ Prepare CNN input
    cnn_input = data_trunc.values.reshape(-1, window_size, num_features)

    # ✅ Generate features
    cnn_output = cnn.predict(cnn_input)
    cnn_features = cnn_output.reshape(cnn_output.shape[0], -1)  # Ensure shape (samples, features)
    cnn_features = StandardScaler().fit_transform(cnn_features)

    # ✅ Prepare hybrid model input
    regimes = hmm.predict(cnn_features)
    aligned_prices = data_trunc['close'].iloc[window_size - 1::window_size]
    hybrid_input = np.concatenate([cnn_features, regimes.reshape(-1, 1)], axis=1)

    # ✅ Generate trade probabilities
    probs = hybrid.predict(hybrid_input)

    return probs, aligned_prices

async def monitor_trade_profit():
    global signals

    while not signals.empty:
        """Monitor the real-time profit/loss of active trades."""
        latest_price = live_data_fetcher.data_current_df['close'].iloc[-1]  # ✅ Get latest price from live data

        print(f"\n📈 Current Price: {latest_price:.2f} USDT")  # ✅ Print current price
        to_close = [] # Store trade indices to close

        for idx, row in signals.iterrows():
            entry_price = row['entry_price']
            stop_loss = row['stop_loss']
            take_profit = row['take_profit']
            profit_pct = ((latest_price - entry_price) / entry_price) * 100

            print(f"\n📊 Trade: {row['trade_type']}")
            print(f"   🎯 Entry Price: {entry_price:.2f}")
            print(f"   🛑 Stop Loss: {row['stop_loss']:.2f}")
            print(f"   🎯 Take Profit: {row['take_profit']:.2f}")
            print(f"   📊 Current Price: {latest_price:.2f}")
            print(f"   💰 Profit/Loss: {profit_pct:.2f}%\n")

            # ✅ Check if stop-loss or take-profit is hit
            if (row['trade_type'] == "BUY" and latest_price <= stop_loss) or \
                (row['trade_type'] == "SELL" and latest_price >= stop_loss):
                print(f"❌ Trade {idx} stopped out at {latest_price:.2f} (Stop Loss hit).")
                to_close.append(idx)

            elif (row['trade_type'] == "BUY" and latest_price >= take_profit) or \
                    (row['trade_type'] == "SELL" and latest_price <= take_profit):
                print(f"✅ Trade {idx} closed at {latest_price:.2f} (Take Profit reached).")
                to_close.append(idx)

        # ✅ Remove closed trades
        signals.drop(to_close, inplace=True)

        await asyncio.sleep(10)

# process last 24 candles to predict
async def live_trade(cnn, hmm, hybrid):
    """Fetch 24 candles, append new ones, and return updated features."""
    global data_live, signals

    print("\n⏳ Fetching new 15-minute candle data...")
    
    if live_data_fetcher.data_list_df.empty:
        print("❌ No new data available.")
        await asyncio.sleep(10)
        return await live_trade(cnn, hmm, hybrid)  # ✅ Retry trade processing

    data_live = create_features(live_data_fetcher.data_list_df)
    print(f"📊 Processed data_live Shape: {data_live.shape}")

    if data_live.empty: # added
        print(" Processed data is empty. Skipping trade execution.")
        return

    data_live = data_live.tail(24)

    data_live.to_csv("processed_data.csv")
    num_features = data_live.shape[1]
    
    probs, aligned_price = generate_trade_probabilities_live(data_live, num_features, cnn, hmm, hybrid)

    if probs is None or aligned_price is None:#added
        print("\n Failed to generate probabilities. Skipping trade execution.\n")
        return

    pd.DataFrame(probs).to_csv("trade_probs.csv", index=False)

    signals = generate_signals(probs, aligned_price)

    if signals.empty:
        print("\n📉 No signals at this moment. Waiting for the next candle.")
    else:
        print("\n✅ Live Trading Signals Generated! 📈")
        print(signals[['trade_type', 'entry_price', 'stop_loss', 'take_profit']])

    await asyncio.sleep(60 * 15) # stop processing until the next 15 minute candle

async def main():
    try:
        cnn = load_cnn_model("saved_models/cnn.keras")
        hmm = load_hmm_model("saved_models/hmm.pkl")
        hybrid = load_hybrid_model("saved_models/hybrid.keras")
    except FileNotFoundError as e:
        print(e)

    print("\n🔴 Live Trading Mode Activated! Press 'x' to stop.\n")
    if cnn is None or hmm is None or hybrid is None:
        print("❌ One or more models failed to load. Cannot proceed with live trading.")
        return

    global data_live, signals
    stop_event = asyncio.Event()
    asyncio.create_task(input_listener(stop_event))

    # collect live 15 and 1 minute data
    asyncio.create_task(main_collect_live_data(symbol='BTCUSDT', interval='15m', window=295))
    
    while len(live_data_fetcher.data_list_df) < 295:
        await asyncio.sleep(10) # check again after 10 seconds
    
    while not stop_event.is_set():
        if len(live_data_fetcher.data_list_df) == 295:
            await live_trade(cnn, hmm, hybrid)

            if not signals.empty:
                asyncio.create_task(monitor_trade_profit())

        else:
            print("Fail to process live data. Less than 24 candles")
        
        await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())