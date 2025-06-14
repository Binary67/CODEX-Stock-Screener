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

    def AllocationUntilDate(self, tickers: List[str], end_date: str) -> pd.Series:
        """Calculate allocations using data up to a given end date."""
        data = self.fetcher.MarketDataAdapter(tickers)
        history = data.loc["2020-01-01":end_date]
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

    def IntervalBacktest(self, tickers: List[str], months: int) -> float:
        """Backtest with periodic rebalancing using a month interval."""
        if months <= 0:
            raise ValueError("months must be positive")

        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")
        current = start_date
        portfolio_value = 1.0

        while current <= end_date:
            lookback_end = (current - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            allocations = self.AllocationUntilDate(tickers, lookback_end)

            period_end = current + pd.DateOffset(months=months) - pd.Timedelta(days=1)
            if period_end > end_date:
                period_end = end_date

            period_returns = []
            for ticker, weight in allocations.items():
                data = self.fetcher.MarketDataAdapter([ticker])
                trade = data.loc[current:period_end]
                if trade.empty:
                    continue
                start_price = trade[ticker].iloc[0]
                end_price = trade[ticker].iloc[-1]
                ret = end_price / start_price - 1
                period_returns.append(ret * weight)

            if period_returns:
                period_total = sum(period_returns)
                portfolio_value *= 1 + period_total

            current = period_end + pd.Timedelta(days=1)

        return portfolio_value - 1
