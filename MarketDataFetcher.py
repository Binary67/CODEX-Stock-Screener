import os
from typing import List, Optional
import logging
import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)


class MarketDataFetcher:
    """Fetches and caches market data."""

    def __init__(self, cache_dir: str = "cache") -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        LOGGER.info("Initialized MarketDataFetcher with cache_dir=%s", cache_dir)

    def MarketDataCache(self, ticker: str, data: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """Save or load data for a ticker from cache."""
        path = os.path.join(self.cache_dir, f"{ticker}.csv")
        if data is None:
            if os.path.exists(path):
                LOGGER.debug("Loading %s from cache", ticker)
                return pd.read_csv(path, index_col=0, parse_dates=True)
            return None
        LOGGER.debug("Caching data for %s", ticker)
        data.to_csv(path)
        return data

    def MarketDataAdapter(self, tickers: List[str]) -> pd.DataFrame:
        """Pull adjusted close prices for tickers."""
        LOGGER.info("Fetching market data for tickers: %s", tickers)
        frames: List[pd.DataFrame] = []
        failures = []
        for ticker in tickers:
            df = self.MarketDataCache(ticker)
            if df is None:
                try:
                    LOGGER.debug("Downloading data for %s", ticker)
                    raw = yf.download(ticker, progress=False, auto_adjust=False)
                except Exception as exc:
                    failures.append((ticker, str(exc)))
                    LOGGER.warning("Failed to download %s: %s", ticker, exc)
                    continue
                if raw.empty:
                    failures.append((ticker, "empty data"))
                    LOGGER.warning("Received empty data for %s", ticker)
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    adj = raw[("Adj Close", ticker)]
                else:
                    adj = raw["Adj Close"]
                df = adj.to_frame(name=ticker)
                self.MarketDataCache(ticker, df)
            frames.append(df.rename(columns={df.columns[0]: ticker}))
        if failures:
            LOGGER.error("Failed to retrieve data for: %s", failures)
            raise RuntimeError(f"Failed to retrieve data for: {failures}")
        return self.MarketDataParser(frames)

    def MarketDataParser(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        """Parse and combine API responses into a single DataFrame."""
        if not frames:
            raise ValueError("No data to parse")
        LOGGER.info("Parsing and combining market data frames")
        return pd.concat(frames, axis=1).sort_index()
