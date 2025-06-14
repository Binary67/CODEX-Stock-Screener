import pandas as pd
from typing import List
from backtesting import Backtest, Strategy
from MarketDataFetcher import MarketDataFetcher
from PortfolioEngine import PortfolioEngine


class BuyAndHoldStrategy(Strategy):
    """Simple buy-and-hold strategy."""

    def init(self) -> None:  # noqa: D401
        """Buy at the first available bar."""
        self.buy()

    def next(self) -> None:  # noqa: D401
        """No-op for each bar."""
        pass


class BacktestingEngine:
    """Backtest a portfolio of tickers using backtesting.py."""

    def __init__(self, fetcher: MarketDataFetcher | None = None) -> None:
        self.fetcher = fetcher or MarketDataFetcher()
        self.portfolio = PortfolioEngine()

    def AllocationFromHistory(self, tickers: List[str]) -> pd.Series:
        """Calculate allocations based on 2020-2023 performance."""
        data = self.fetcher.MarketDataAdapter(tickers)
        history = data.loc["2020-01-01":"2023-12-31"]
        cumulative = history.pct_change().add(1).prod() - 1
        top_count = max(int(len(tickers) * 0.3), 1)
        selected = cumulative.sort_values(ascending=False).head(top_count)
        return self.portfolio.AllocationCalculator(selected, method="score")

    def PortfolioBacktest(self, allocations: pd.Series) -> float:
        """Run backtest for 2024 based on predetermined allocations."""
        results = []
        for ticker, weight in allocations.items():
            data = self.fetcher.MarketDataAdapter([ticker])
            trade_data = data.loc["2024-01-01":"2024-12-31"]
            if trade_data.empty:
                continue
            df = pd.DataFrame(
                {
                    "Open": trade_data[ticker],
                    "High": trade_data[ticker],
                    "Low": trade_data[ticker],
                    "Close": trade_data[ticker],
                }
            )
            initial_cash = 10000 * weight
            backtest = Backtest(df, BuyAndHoldStrategy, cash=initial_cash, trade_on_close=True)
            stats = backtest.run()
            ret = stats["Equity Final [$]"] / initial_cash - 1
            results.append(ret * weight)
        if results:
            return sum(results)
        return 0.0

    def BuyAndHoldReturn(self, tickers: List[str]) -> float:
        """Compute buy-and-hold return for equally weighted tickers."""
        if not tickers:
            raise ValueError("No tickers provided")
        weight = 1.0 / len(tickers)
        allocations = pd.Series(weight, index=tickers)
        return self.PortfolioBacktest(allocations)
