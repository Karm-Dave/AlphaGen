# config/constants.py

ASSETS = ['AAPL', 'IBM', 'MCD', 'KO']
INDICATORS = ['SMA', 'EMA', 'RSI', 'MACD']
OPERATORS = ['crosses_above', 'crosses_below', 'greater_than', 'less_than']
TIMEFRAMES = ['1d']
PRICE_TYPES = ['Close', 'Open', 'High', 'Low']

# Numerical feature ranges
PERIOD_RANGE = (5, 100)
PRICE_THRESHOLD_RANGE = (0, 1000)
STOP_LOSS_RANGE = (0.5, 5.0)
RISK_PER_TRADE_RANGE = (0.5, 5.0)
PROFIT_TARGET_RANGE = (1.0, 20.0)
TRANSACTION_COST = 0.001  # 0.1% per trade

# Strategy JSON template
STRATEGY_TEMPLATE = {
    "strategy": {
        "metadata": {
            "name": "",
            "asset": "",
            "timeframe": "",
            "asset_class": "Equities"
        },
        "entry_rules": {
            "type": "indicator_comparison",
            "indicator_left": {"name": "", "parameters": {"period": 0, "source": "Close"}},
            "operator": "",
            "indicator_right": {"name": "", "parameters": {"period": 0, "source": "Close"}}
        },
        "exit_rules": {
            "type": "profit_target",
            "condition": {"profit_percentage": 0.0}
        },
        "stop_loss_rules": {
            "type": "fixed_percentage",
            "percentage": 0.0
        },
        "position_sizing": {
            "type": "risk_based",
            "risk_per_trade_percentage": 0.0
        }
    }
}