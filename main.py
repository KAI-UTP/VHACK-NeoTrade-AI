import pandas as pd
import numpy as np
import os
from data_fetcher import preprocess_data
from cnn_feature_extractor import build_cnn, update_cnn, save_model as save_cnn, load_model as load_cnn
from hmm_regime_detector import train_hmm, save_hmm, load_hmm, update_hmm
from hybrid_model import build_model, update_model, save_model as save_hybrid, load_model as load_hybrid
from signal_generator import generate_signals
from backtester import run_backtest
from sklearn.preprocessing import StandardScaler
from reward import save_rewards, load_rewards

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    while True:  # Loop indefinitely
        print("\n--- Main Menu ---")
        print("1. Backtest")
        print("2. Live Trade")
        print("3. Exit")
        mode = input("Enter choice (1, 2, or 3): ")

        if mode == '3':  # Exit the program
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

            try: 
                cnn = load_or_build_cnn(f"{MODEL_DIR}/cnn.keras", (24, num_features))
                hmm = load_or_build_hmm(f"{MODEL_DIR}/hmm.pkl", data.values)
                hybrid = load_or_build_hybrid(f"{MODEL_DIR}/hybrid.keras", cnn, hmm, data)
                    
            except Exception as e:
                print(f"Error: {e}")
    
            # Prepare CNN input
            window_size = 24
            truncated_len = (len(data) // window_size) * window_size
            data_trunc = data.iloc[:truncated_len]
            cnn_input = data_trunc.values.reshape(-1, window_size, num_features)
        
            # Generate features
            cnn_output = cnn.predict(cnn_input)
            cnn_features = cnn_output.reshape(cnn_output.shape[0], -1)  # ✅ Ensure (samples, features)
            cnn_features = StandardScaler().fit_transform(cnn_features)

            # Prepare hybrid model data
            regimes = hmm.predict(cnn_features)
            # ✅ Ensure labels match hybrid_input length
            aligned_prices = data_trunc['close'].iloc[window_size - 1:len(data_trunc):window_size]  # Extract correctly
            labels = (aligned_prices.pct_change().shift(-1) > 0).astype(int).values  # Convert to NumPy array

        
            hybrid_input = np.concatenate([cnn_features, regimes.reshape(-1, 1)], axis=1)
            
            # Generate signals
            probs = hybrid.predict(hybrid_input)
            # Write probability to a csv file
            pd.DataFrame(probs).to_csv("trade_probs.csv", index=False)
            signals = generate_signals(probs, aligned_prices)
            
            # Run backtest
            if not signals.empty:
                results = run_backtest(initial_balance)
                print(f"\nBacktest Results:")
                print(f"Initial Balance: ${results['initial']:,.2f}")
                print(f"Final Balance: ${results['final']:,.2f}")
                print(f"ROI: {results['roi']:.2f}%")
                #print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
                print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
                print(f"Total Trades: {results['total_trades']}")
                print(f"Win Rate: {results['win_rate']:.2f}%")

                rewards = results['trades']['pnl']
                # ✅ Ensure `rewards` is always a NumPy array
                rewards = np.array(rewards) if isinstance(rewards, (pd.Series, list)) else rewards

                all_rewards = np.concatenate([load_rewards(), rewards.reshape(-1)]) 

                # ✅ Check if rewards is empty before using max() and min()
                if rewards.size > 0:  # ✅ Use `.size` instead of `len(rewards)`
                    if np.max(all_rewards) > np.min(all_rewards):  # ✅ Use NumPy functions
                        all_rewards = (all_rewards - np.min(all_rewards)) / (np.max(all_rewards) - np.min(all_rewards)) * 2 - 1  # Normalize to [-1, 1]
                    else:
                        all_rewards = np.zeros_like(all_rewards)  # Set all rewards to 0 if no variation
                else:
                    all_rewards = np.array([0])  # ✅ Set rewards to an array with a single zero if empty
                """
                # Update models
                cnn = update_cnn(cnn, cnn_input, all_rewards)
                save_cnn(cnn, f"{MODEL_DIR}/cnn.keras")
                hmm = update_hmm(hmm, cnn_features, all_rewards)
                save_hmm(hmm, f"{MODEL_DIR}/hmm.pkl")
                hybrid = update_model(hybrid, hybrid_input, labels, all_rewards)
                save_hybrid(hybrid, f"{MODEL_DIR}/hybrid.keras")
                save_rewards(all_rewards)
                """
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

def load_or_build_cnn(path, input_shape):
    if model_exists(path):
        cnn = load_cnn(path)
        if cnn.optimizer is None:
            cnn.compile(optimizer='adam', loss='mse')
        print("CNN model loaded successfully.")
        return cnn
    print("Initializing new CNN model.")
    cnn = build_cnn(input_shape)
    cnn.compile(optimizer='adam', loss='mse')
    return cnn

def load_or_build_hmm(path, features):
    if model_exists(path):
        print("HMM model loaded successfully.")
        return load_hmm(path)
    print("Training new HMM model...")
    hmm = train_hmm(features, n_components=3)
    save_hmm(hmm, path)
    return hmm

def load_or_build_hybrid(path, cnn, hmm, data):
    """Load hybrid model if it exists; otherwise, train a new one with error handling."""
    try:
        if model_exists(path):
            hybrid = load_hybrid(path)
            if hybrid.optimizer is None:
                hybrid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print("Hybrid model loaded successfully.")
            return hybrid
        print("Hybrid model not found. Training a new hybrid model...")

        # ✅ Data Preprocessing
        window_size = 24
        num_features = data.shape[1]
        truncated_len = (len(data) // window_size) * window_size
        data_trunc = data.iloc[:truncated_len]
        cnn_input = data_trunc.values.reshape(-1, window_size, num_features)
        cnn_features = StandardScaler().fit_transform(cnn.predict(cnn_input))
        regimes = hmm.predict(cnn_features)
        hybrid_input = np.concatenate([cnn_features, regimes.reshape(-1, 1)], axis=1)

        # ✅ Prepare Labels for Training
        aligned_prices = data_trunc['close'].iloc[window_size - 1::window_size]
        labels = (aligned_prices.pct_change().shift(-1) > 0).astype(int)

        # ✅ Train Hybrid Model
        hybrid = build_model((hybrid_input.shape[1],))
        hybrid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        hybrid.fit(hybrid_input, labels, epochs=10, verbose=1)

        # ✅ Save Model
        save_hybrid(hybrid, path)
        print("Hybrid model trained and saved.")

        return hybrid
    except Exception as e:
        print(f"Error while building Hybrid model: {e}")
        return None  # Return None if an error occurs

def model_exists(model_path):
    """Check if the model file exists."""
    return os.path.exists(model_path) and os.path.getsize(model_path) > 0

if __name__ == "__main__":
    main()