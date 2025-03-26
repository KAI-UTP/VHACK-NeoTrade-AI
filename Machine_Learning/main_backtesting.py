import pandas as pd
import numpy as np
import os
from data_fetcher import preprocess_data
from cnn_feature_extractor import load_model as load_cnn
from hmm_regime_detector import load_hmm
from hybrid_model import load_model as load_hybrid
from signal_generator import generate_signals
from sklearn.preprocessing import StandardScaler
from backtester import backtest

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    try: 
        cnn = load_or_build_cnn(f"{MODEL_DIR}/cnn.keras")
        hmm = load_or_build_hmm(f"{MODEL_DIR}/hmm.pkl")
        hybrid = load_or_build_hybrid(f"{MODEL_DIR}/hybrid.keras")
            
    except Exception as e:
        print(f"Error: {e}")

    while True:  # Loop indefinitely
        print("\n--- Main Menu ---")
        print("1. Backtest")
        print("2. Exit")
        mode = input("Enter choice (1 or 2): ")

        if mode == '2':  # Exit the program
            print("Exiting the program. Goodbye!")
            break

        if mode not in ['1', '2']:
            print("Invalid choice. Please try again.")
            continue

        print("\nSelect symbol:")
        print("1. BTC/USDT")
        print("2. ETH/USDT")
        print("3. SOL/USDT")
        print("4. XRP/USDT")
        symbol_choice = input("Enter choice (1, 2, 3, or 4): ")

        symbols = {
            '1': 'BTC/USDT',
            '2': 'ETH/USDT',
            '3': 'SOL/USDT',
            '4': 'XRP/USDT'
        }

        if symbol_choice not in symbols:
            print("Invalid symbol choice. Please try again.")
            continue

        symbol = symbols[symbol_choice]

        if mode == '1':  # Backtest mode
            # Date range
            start_date = input("Start date (YYYY-MM-DD): ")
            end_date = input("End date (YYYY-MM-DD): ")
            initial_balance = input("Initial balance: ")
            initial_balance = float(initial_balance)

            # Fetch and preprocess data 
            data = preprocess_data(symbol, '15m', start_date, end_date)
            if data.empty:
                print("No data available!")
                return
            num_features = data.shape[1]
    
            window_size = 24

            if len(data) < window_size:
                raise ValueError(f"Not enough data to create a rolling window. Need at least {window_size} rows.")

            # ✅ Convert Data to Float32 for Memory Efficiency
            data_trunc = data.iloc[: (len(data) // window_size) * window_size].astype(np.float32)

            # ✅ Efficient Rolling Window Creation
            rolling_data = np.lib.stride_tricks.sliding_window_view(
                data_trunc.values, (window_size, num_features)
            ).reshape(-1, window_size, num_features)

            cnn_input = rolling_data

            # ✅ Load & Process CNN Model
            cnn = load_cnn("saved_models/cnn.keras")
            cnn_output = cnn.predict(cnn_input)
            cnn_features = cnn_output.reshape(cnn_output.shape[0], -1)
            cnn_features = StandardScaler().fit_transform(cnn_features)

            # ✅ Select HMM Features
            hmm_features = data_trunc[['returns', 'log_returns', 'volatility_24h', 'volatility_7d',
                                       'rsi', 'macd', 'sma_50', 'sma_200', 'ema_20', 'open', 'high', 'low', 'close', 'volume']]

            # ✅ Efficient Rolling Window for HMM
            hmm_input = np.lib.stride_tricks.sliding_window_view(
                hmm_features.values, (window_size, len(hmm_features.columns))
            ).reshape(-1, window_size, len(hmm_features.columns))

            hmm_input = np.mean(hmm_input, axis=1)

            # ✅ Ensure Correct HMM Shape
            expected_features = len(hmm_features.columns)
            if hmm_input.shape[1] != expected_features:
                raise ValueError(f"Incorrect HMM input shape: Expected ({None}, {expected_features}), got {hmm_input.shape}")

            # ✅ Load & Predict with HMM
            regimes = hmm.predict(hmm_input)

            # ✅ Prepare Hybrid Model Input
            aligned_prices = data_trunc['close'].iloc[window_size - 1:]
            hybrid_input = np.concatenate([cnn_features, regimes.reshape(-1, 1)], axis=1)
            
            # Generate signals
            probs = hybrid.predict(hybrid_input)
            # Write probability to a csv file
            pd.DataFrame(probs).to_csv("trade_probs.csv", index=False)
            signals = generate_signals(probs, aligned_prices)
        
            # Run backtest
            if not signals.empty:
                results = backtest(signals, aligned_prices,initial_balance)
                print(f"\nBacktest Results:")
                print(f"Initial Balance: ${results['initial']:,.2f}")
                print(f"Final Balance: ${results['final']:,.2f}")
                print(f"ROI: {results['roi']:.2f}%")
                print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
                print(f"Total Trades: {results['total_trades']}")
                print(f"Win Rate: {results['win_rate']:.2f}%")

                # Save trades
                trades_df = results['trades']
                trades_df['action'] = trades_df['type']
                trades_df['profit_loss'] = trades_df['pnl']
                trades_df.rename(columns={'entry': 'entry_price', 'exit': 'exit_price', 'pnl': 'profit_loss'}, inplace=True)
                trades_df[['action', 'entry_price', 'exit_price', 'profit_loss']].to_csv("trade_records.csv", index=False)

                print("\nTrade records saved to 'trade_records.csv'")
            else:
                print("No signals generated due to errors")

        elif mode == '2':  # Live trade mode
            print("\nLive trading is not implemented yet. Returning to main menu.")

def load_or_build_cnn(path):
    if model_exists(path):
        cnn = load_cnn(path)
        if cnn.optimizer is None:
            cnn.compile(optimizer='adam', loss='mse')
        print("CNN model loaded successfully.")
        return cnn
    print("CNN model does not exist.")
    return None

def load_or_build_hmm(path):
    if model_exists(path):
        print("HMM model loaded successfully.")
        return load_hmm(path)
    print("HMM model does not exist.")
    return None

def load_or_build_hybrid(path):
    """Load hybrid model if it exists; otherwise, return None."""
    try:
        if model_exists(path):
            hybrid = load_hybrid(path)
            if hybrid.optimizer is None:
                hybrid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print("Hybrid model loaded successfully.")
            return hybrid
        print("Hybrid model does not exist.")
        return None
    except Exception as e:
        print(f"Error while loading Hybrid model: {e}")
        return None  # Return None if an error occurs

def model_exists(model_path):
    """Check if the model file exists."""
    return os.path.exists(model_path) and os.path.getsize(model_path) > 0

if __name__ == "__main__":
    main()