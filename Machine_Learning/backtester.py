import pandas as pd
import numpy as np

def backtest(signals, prices, initial, risk_pct=0.005, trade_pct=0.01):
    """Backtester with percentage-based position sizing and balance tracking."""
    balance = initial
    positions = []
    pnl_list = []
    equity_curve = [initial]
    wins, total_trades = 0, 0
    rewards = []  # Store rewards

    for idx, row in signals.iterrows():
        if balance <= 0:
            break  # Stop trading if balance is depleted

        pnl = 0  # Default PnL
        trade_amount = balance * trade_pct  # Allocate a percentage of balance
        risk_amount = min(trade_amount * risk_pct, balance)  # Ensure risk does not exceed balance
        exit_price = None

        if row['signal'] == 1:  # Long Trade
            price_diff = row['entry_price'] - row['stop_loss']
            size = risk_amount / abs(price_diff) if price_diff != 0 else 0  # Avoid division by zero

            if not isinstance(idx, pd.Timestamp):
                idx = prices.index[min(idx, len(prices) - 1)]  

            for price in prices.loc[idx:]:
                if price <= row['stop_loss']:  
                    exit_price = row['stop_loss']
                    break
                if price >= row['take_profit']:  
                    exit_price = row['take_profit']
                    break
            if exit_price is None:
                exit_price = prices.iloc[-1]  

            pnl = size * (exit_price - row['entry_price'])

        elif row['signal'] == -1:  # Short Trade
            price_diff = row['stop_loss'] - row['entry_price']
            size = risk_amount / abs(price_diff) if price_diff != 0 else 0  # Avoid division by zero

            for price in prices.loc[idx:]:
                if price >= row['stop_loss']:  
                    exit_price = row['stop_loss']
                    break
                if price <= row['take_profit']:  
                    exit_price = row['take_profit']
                    break
            if exit_price is None:
                exit_price = prices.iloc[-1]  

            pnl = size * (row['entry_price'] - exit_price)

        balance += pnl
        balance = max(balance, 0)  # Ensure balance does not go negative
        pnl_list.append(pnl)
        equity_curve.append(balance)
        positions.append(('buy' if row['signal'] == 1 else 'sell', row['entry_price'], exit_price, pnl))

        # Compute reward as profit/loss relative to initial balance
        reward = pnl / initial  
        rewards.append(reward)  

        if pnl > 0:
            wins += 1
        total_trades += 1

    # Normalize rewards to range [-1, 1]
    rewards = np.array(rewards)
    if len(rewards) > 0 and rewards.max() > rewards.min():
        rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min()) * 2 - 1  
    # ✅ Convert PnL to percentage returns
    returns = np.array(pnl_list) / initial  # Normalize PnL by initial balance

    # ✅ Calculate the number of trades
    total_trades = len(returns)

    # ✅ Calculate annualized Sharpe Ratio based on actual trades
    if total_trades > 1 and np.std(returns) != 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(total_trades)  # Dynamic adjustment
    else:
        sharpe_ratio = 0  # Avoid division by zero
    # Compute existing backtest metrics
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_drawdown = np.min(drawdowns) * 100  

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0  

    return {
        'initial': initial,
        'final': balance,
        'roi': (balance - initial) / initial * 100,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'trades': pd.DataFrame(positions, columns=['type', 'entry', 'exit', 'pnl']),
        'rewards': rewards  # ✅ Added rewards without modifying other outputs
    }
