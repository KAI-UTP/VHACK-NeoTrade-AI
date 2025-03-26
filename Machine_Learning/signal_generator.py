import pandas as pd
import numpy as np

def generate_signals(probs, prices, threshold=0.7):
    """Generate trading signals where probability > 70%"""

    # ✅ Ensure `probs` and `prices` have the same length
    min_length = min(len(probs), len(prices))
    probs = probs[:min_length]
    prices = prices.iloc[:min_length]

    signals = pd.DataFrame(index=pd.RangeIndex(start=0, stop=min_length, step=1))  # Ensure a valid index
    signals['price'] = prices.values  # Ensure alignment with NumPy array
    signals['signal'] = 0  # Default: No signal

    # ✅ Only consider trades where probability > 70%
    buy_indices = np.where(probs[:, 0] > threshold)[0]  # Buy signals if P(Buy) > 70%
    sell_indices = np.where(probs[:, 1] > threshold)[0]  # Sell signals if P(Sell) > 70%

    # Apply buy and sell signals
    signals.loc[buy_indices, 'signal'] = 1
    signals.loc[sell_indices, 'signal'] = -1

    # Define trade type
    signals['trade_type'] = signals['signal'].map({1: "BUY", -1: "SELL", 0: "NONE"})

    # ✅ Calculate risk parameters only for trades that meet the threshold
    signals['entry_price'] = signals['price']
    signals['stop_loss'] = signals.apply(
        lambda x: x['entry_price'] * (1 - 0.005) if x['signal'] == 1 else x['entry_price'] * (1 + 0.005),
        axis=1
    )
    signals['take_profit'] = signals.apply(
        lambda x: x['entry_price'] * (1 + 0.015) if x['signal'] == 1 else x['entry_price'] * (1 - 0.015),
        axis=1
    )

    # ✅ Filter out signals with no trade (signal == 0)
    filtered_signals = signals[signals['signal'] != 0]

    # Ensure signals and prices index match
    if len(signals) != len(prices.index):
        min_length = min(len(signals), len(prices.index))
        signals = signals[:min_length]
        prices.index = prices.index[:min_length]

    filtered_signals.index = prices.index[:len(filtered_signals)]  # ✅ Ensure index consistency

    # ✅ If no valid signals are generated, return empty DataFrame
    if filtered_signals.empty:
        print("⚠️ No valid signals above the 70% probability threshold. Returning empty DataFrame.")
        return pd.DataFrame()

    filtered_signals = filtered_signals.groupby(filtered_signals.index).first()

    # ✅ Save signals to CSV
    filtered_signals[['trade_type', 'entry_price', 'stop_loss', 'take_profit']].to_csv('trade_signals.csv', index=False)
    
    print(f"✅ Trade signals saved to 'trade_signals.csv' (filtered by probability > {threshold*100}%)")
    return filtered_signals
