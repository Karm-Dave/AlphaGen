# data/data_fetcher.py

"""
Module for fetching and processing financial data.
"""

import logging
import numpy as np
import yfinance as yf

from config.constants import PRICE_TYPES

class DataFetcher:
    """
    Handles data fetching and indicator calculations using Yahoo Finance data.
    """

    def __init__(self, start_date='2015-01-01'):
        self.start_date = start_date

    def fetch(self, symbol: str, end_date: str, interval: str = '1d') -> np.ndarray:
        try:
            data = yf.download(symbol, start=self.start_date, end=end_date, interval=interval, auto_adjust=False)
            if data.empty or len(data) < 100:
                logging.error(f"Insufficient data for {symbol} at {interval}: {len(data)} rows.")
                return np.array([])
            logging.info(f"Fetched {interval} data for {symbol} from {self.start_date} to {end_date}, rows: {len(data)}")
            return np.array(data[['Open', 'High', 'Low', 'Close', 'Volume']])
        except Exception as e:
            logging.error(f"Error fetching data for {symbol} at {interval}: {e}")
            return np.array([])

    def compute_volatility(self, data: np.ndarray) -> float:
        if data.size == 0:
            return 1.0
        returns = np.diff(data[:, 3]) / data[:-1, 3]
        return np.std(returns) * np.sqrt(252) if len(returns) > 1 else 1.0

    def compute_indicator(self, data: np.ndarray, indicator: str, period: int, source: str = 'Close') -> np.ndarray:
        if data.size == 0 or len(data) < period:
            logging.warning(f"Insufficient data for indicator {indicator}, period {period}")
            return np.zeros(max(period, len(data)))

        source_idx = {'Open': 0, 'High': 1, 'Low': 2, 'Close': 3, 'Volume': 4}
        if source not in source_idx:
            logging.warning(f"Invalid source '{source}'. Defaulting to 'Close'.")
            source = 'Close'

        prices = data[:, source_idx[source]]

        try:
            if indicator == 'SMA':
                result = np.zeros(len(prices))
                for i in range(period - 1, len(prices)):
                    result[i] = np.mean(prices[i-period+1:i+1])
                return result
            elif indicator == 'EMA':
                result = np.zeros(len(prices))
                k = 2 / (period + 1)
                result[0] = prices[0]
                for i in range(1, len(prices)):
                    result[i] = prices[i] * k + result[i-1] * (1 - k)
                return result
            elif indicator == 'RSI':
                with np.errstate(divide='ignore', invalid='ignore'):
                    delta = np.diff(prices, prepend=prices[0])
                    gain = np.where(delta > 0, delta, 0)
                    loss = np.where(delta < 0, -delta, 0)
                    avg_gain = np.zeros(len(prices))
                    avg_loss = np.zeros(len(prices))
                    for i in range(period - 1, len(prices)):
                        avg_gain[i] = np.mean(gain[i-period+1:i+1])
                        avg_loss[i] = np.mean(loss[i-period+1:i+1])
                    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
                    rsi = np.where(rs != 0, 100 - (100 / (1 + rs)), 50)
                    return np.nan_to_num(rsi, nan=50, posinf=50, neginf=50)
            elif indicator == 'MACD':
                ema12 = np.zeros(len(prices))
                ema26 = np.zeros(len(prices))
                k12 = 2 / (12 + 1)
                k26 = 2 / (26 + 1)
                ema12[0] = prices[0]
                ema26[0] = prices[0]
                for i in range(1, len(prices)):
                    ema12[i] = prices[i] * k12 + ema12[i-1] * (1 - k12)
                    ema26[i] = prices[i] * k26 + ema26[i-1] * (1 - k26)
                return np.nan_to_num(ema12 - ema26, nan=0)
            return np.zeros(len(prices))
        except Exception as e:
            logging.error(f"Error computing {indicator}: {e}")
            return np.zeros(len(prices))