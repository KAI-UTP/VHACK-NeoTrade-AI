import pandas as pd
import numpy as np

def generate_signals(probs, prices, signal_ratio=0.9, range_buffer=0.002):
    """Generate trading signals with entry range, timestamps, and correct time zones."""
    min_length = min(len(probs), len(prices))
    probs = probs[:min_length]
    prices = prices.iloc[:min_length]

    signals = pd.DataFrame(index=prices.index)
    signals['timestamp'] = prices.index  # Store UTC timestamps
    signals['price'] = prices.values
    signals['signal'] = 0

    n_signals = int(len(prices) * signal_ratio)

    # Generate buy/sell indices
    buy_indices = np.argsort(probs[:, 0])[-n_signals:]
    sell_indices = np.argsort(probs[:, 1])[-n_signals:]

    # Convert numerical indices to actual timestamps
    buy_timestamps = signals.index[buy_indices]
    sell_timestamps = signals.index[sell_indices]

    signals.loc[buy_timestamps, 'signal'] = 1
    signals.loc[sell_timestamps, 'signal'] = -1

    # Define entry range instead of a fixed price
    signals['entry_price_low'] = signals['price'] * (1 - range_buffer)
    signals['entry_price_high'] = signals['price'] * (1 + range_buffer)

    # Stop loss and take profit
    signals['stop_loss'] = signals.apply(
        lambda x: x['entry_price_low'] * (1 - 0.005) if x['signal'] == 1 else x['entry_price_high'] * (1 + 0.005),
        axis=1
    )
    signals['take_profit'] = signals.apply(
        lambda x: x['entry_price_high'] * (1 + 0.02) if x['signal'] == 1 else x['entry_price_low'] * (1 - 0.02),
        axis=1
    )

    filtered_signals = signals[signals['signal'] != 0]

    if filtered_signals.empty:
        print("⚠️ No valid signals generated. Returning empty DataFrame.")
        return pd.DataFrame()

    # ✅ Ensure 'timestamp' column exists before formatting
    if 'timestamp' not in filtered_signals.columns:
        filtered_signals.insert(0, 'timestamp', signals.index)  # Add timestamp from index

    # ✅ Convert timestamp to correct format
    filtered_signals['timestamp'] = pd.to_datetime(filtered_signals['timestamp'])  # Ensure datetime type
    filtered_signals['timestamp'] = filtered_signals['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Format

    filtered_signals = filtered_signals.reset_index(drop=True)  # Reset index safely

    # Save signals with timestamp
    filtered_signals.to_csv("trade_signals.csv", index=False)  # Ensure index is not duplicated
    print("✅ Trade signals saved to 'trade_signals.csv'")
    
    return filtered_signals
