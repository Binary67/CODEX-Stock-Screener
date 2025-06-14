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

    def _RunBacktestForPeriod(
        self,
        allocations: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
        cash: float,
    ) -> tuple[float, list]:
        """Run a single backtest period and return final value and stats."""
        stats_list = []
        final_value = 0.0
        for ticker, weight in allocations.items():
            data = self.fetcher.MarketDataAdapter([ticker])
            trade_data = data.loc[start:end]
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
            initial_cash = cash * weight
            backtest = Backtest(
                df,
                BuyAndHoldStrategy,
                cash=initial_cash,
                trade_on_close=True,
            )
            stats = backtest.run()
            stats_list.append(stats)
            final_value += stats["Equity Final [$]"]
        if not stats_list:
            return cash, stats_list
        return final_value, stats_list

    def PortfolioBacktest(self, allocations: pd.Series, rebalance_months: int = 0) -> float:
        """Run backtest for 2024 with optional monthly rebalancing."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")
        cash = 10000.0

        if rebalance_months <= 0:
            final_value, _ = self._RunBacktestForPeriod(
                allocations, start_date, end_date, cash
            )
            return final_value / cash - 1

        current = start_date
        current_alloc = allocations
        while current <= end_date:
            period_end = current + pd.DateOffset(months=rebalance_months) - pd.Timedelta(days=1)
            if period_end > end_date:
                period_end = end_date

            cash, _ = self._RunBacktestForPeriod(current_alloc, current, period_end, cash)

            current = period_end + pd.Timedelta(days=1)
            if current > end_date:
                break
            lookback_end = (current - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            current_alloc = self.AllocationUntilDate(list(allocations.index), lookback_end)

        return cash / 10000.0 - 1

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
        allocations = self.AllocationUntilDate(tickers, "2023-12-31")
        return self.PortfolioBacktest(allocations, rebalance_months=months)
