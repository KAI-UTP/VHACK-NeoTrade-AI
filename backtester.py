import backtrader as bt
import pandas as pd
import numpy as np


class SignalStrategy(bt.Strategy):
    params = (('risk_pct', 0.1),)  # Risk per trade (10% of available capital)

    def __init__(self):
        self.data_close = self.datas[0].close
        self.orders = {}  # Track open orders
        self.signal_data = pd.read_csv("trade_signals.csv", parse_dates=['timestamp'])
        self.signal_data.set_index('timestamp', inplace=True)
        self.total_trades = 0
        self.winning_trades = 0
        self.trade_log = []  # Store trade results

    def next(self):
        current_time = self.datetime.datetime(0)
        if current_time not in self.signal_data.index:
            return

        row = self.signal_data.loc[current_time]
        signal = row['signal']
        entry_price_low = row['entry_price_low']
        entry_price_high = row['entry_price_high']
        stop_loss = row['stop_loss']
        take_profit = row['take_profit']

        # ✅ Ensure Valid SL/TP Placement
        if signal == 1:  # Long Trade
            if stop_loss >= entry_price_low:
                stop_loss = entry_price_low * 0.995  # Set SL 0.5% below entry
            if take_profit <= entry_price_high:
                take_profit = entry_price_high * 1.01  # Set TP 1% above entry

        elif signal == -1:  # Short Trade
            if stop_loss <= entry_price_high:
                stop_loss = entry_price_high * 1.005  # Set SL 0.5% above entry
            if take_profit >= entry_price_low:
                take_profit = entry_price_low * 0.99  # Set TP 1% below entry

        cash_available = self.broker.get_cash()
        position_size = self.getposition().size  

        # ✅ Adjust trade size dynamically
        # ✅ Ensure `trade_size` is properly defined before using min()
        max_trade_size = (cash_available * self.params.risk_pct) / self.data_close[0]
        trade_size = max_trade_size  # ✅ Assign `trade_size` first
        trade_size = max(0.01, trade_size)  # ✅ Prevents uninitialized variable error

        buffer = 0.001  # ✅ 0.1% buffer to increase execution rate

        if signal == 1 and entry_price_low * (1 - buffer) <= self.data_close[0] <= entry_price_high * (1 + buffer):
            self.orders[current_time] = self.buy_bracket(
                size=trade_size,
                limitprice=take_profit,  
                stopprice=stop_loss,
                valid=None  # ✅ Ensures order stays active
            )

        elif signal == -1 and entry_price_low * (1 - buffer) <= self.data_close[0] <= entry_price_high * (1 + buffer):
            self.orders[current_time] = self.sell_bracket(
                size=trade_size,
                limitprice=take_profit,  
                stopprice=stop_loss,
                valid=None
            )

    def notify_trade(self, trade):
        """Handles trade closure and logs profit/loss."""
        if trade.isclosed:
            self.total_trades += 1
            profit_or_loss = trade.pnl  # Profit/Loss for the trade
            percent_return = (profit_or_loss / self.broker.get_value()) * 100  # % return on capital

            if profit_or_loss > 0:
                self.winning_trades += 1

            self.trade_log.append({
                'Entry Price': trade.price,
                'Exit Price': trade.price + trade.pnlcomm,  # Adjusted for commission
                'Profit/Loss': profit_or_loss,
                'Return (%)': percent_return
            })
        #print(f"📌 Trade Closed: PnL: {profit_or_loss:.2f}, Return: {percent_return:.2f}%")
            

    def notify_order(self, order):
        """Handles order execution, stop-loss triggering, and rejected orders."""
        if order.status in [order.Submitted, order.Accepted]:
            return  # Order is processing

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"🟢 BUY EXECUTED at {order.executed.price}")
            else:
                print(f"🔴 SELL EXECUTED at {order.executed.price}")

            print(f"💰 Cash: {self.broker.get_cash()} | 📊 Portfolio Value: {self.broker.get_value()}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"❌ Order REJECTED: {order.status}")
        

# Analyzer for Performance Metrics
def run_backtest(initial_balance, datafile="processed_data.csv"):
    cerebro = bt.Cerebro()
    data = bt.feeds.GenericCSVData(
        dataname=datafile,
        dtformat='%Y-%m-%d %H:%M:%S',
        timeframe=bt.TimeFrame.Minutes,
        compression=15,
        openinterest=-1
    )

    cerebro.adddata(data)
    strategy = cerebro.addstrategy(SignalStrategy)

    # ✅ Enable margin trading and leverage
    cerebro.broker.setcommission(commission=0.001, leverage=10.0)  # ✅ 10x leverage

    # Set initial balance
    cerebro.broker.set_cash(initial_balance)
    cerebro.broker.set_shortcash(True)  # ✅ Allows short selling even without owning the asset
    
    # Add analyzers for performance metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # Run backtest
    results = cerebro.run()
    strat = results[0]

    # Get final portfolio value
    final_value = cerebro.broker.get_value()
    roi = ((final_value - initial_balance) / initial_balance) * 100

    # Get performance metrics
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    max_drawdown = strat.analyzers.drawdown.get_analysis().max.drawdown
    win_rate = (strat.winning_trades / strat.total_trades * 100) if strat.total_trades > 0 else 0

    # Print Performance Summary
    print("\n📈 Performance Summary:")
    print(f"💰 Final Portfolio Value: {final_value:.2f}")
    print(f"📊 ROI: {roi:.2f}%")
    print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
    print(f"📏 Sharpe Ratio: {sharpe_ratio:.2f}" if sharpe_ratio is not None else "⚠️ Sharpe Ratio: N/A")
    print(f"🏆 Win Rate: {win_rate:.2f}%")
    print(f"📈 Total Trades: {strat.total_trades}")

    # ✅ Get trade rewards (PnL for each trade)
    rewards = [trade['Profit/Loss'] for trade in strat.trade_log]
    raw_rewards = np.array(rewards)  # ✅ Store raw rewards before modifying

    # ✅ Return all results
    return {
        'initial': initial_balance,
        'final': final_value,
        'roi': roi,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': strat.total_trades,
        'trade_rewards': rewards,  # ✅ List of profit/loss per trade
        'raw_rewards': raw_rewards  # ✅ Return unmodified raw rewards
    }

# Run Backtest and Get Results
results = run_backtest(initial_balance=10000, datafile="processed_data.csv")
