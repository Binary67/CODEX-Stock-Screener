import logging
import pandas as pd
from typing import List
from backtesting import Backtest, Strategy
from MarketDataFetcher import MarketDataFetcher
from PortfolioEngine import PortfolioEngine
from ConfigManager import ConfigManager

LOGGER = logging.getLogger(__name__)


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

    def __init__(
        self,
        fetcher: MarketDataFetcher | None = None,
        config: dict | None = None,
    ) -> None:
        self.fetcher = fetcher or MarketDataFetcher()
        self.portfolio = PortfolioEngine()
        manager = ConfigManager()
        config_data = config or manager.LoadConfig()
        self.initial_cash = float(config_data.get("InitialCash", 10000))
        self.allocation_method = config_data.get("AllocationMethod", "equal")
        LOGGER.info(
            "BacktestingEngine initialized with initial_cash=%s, allocation_method=%s",
            self.initial_cash,
            self.allocation_method,
        )

    def AllocationFromHistory(self, tickers: List[str]) -> pd.Series:
        """Calculate allocations based on 2020-2023 performance."""
        LOGGER.info("Calculating historical allocation for tickers: %s", tickers)
        data = self.fetcher.MarketDataAdapter(tickers)
        history = data.loc["2020-01-01":"2023-12-31"]
        cumulative = history.pct_change(fill_method=None).add(1).prod() - 1
        top_count = max(int(len(tickers) * 0.3), 1)
        selected = cumulative.sort_values(ascending=False).head(top_count)
        if self.allocation_method == "volatility":
            vol_series = history.pct_change(fill_method=None).std().reindex(selected.index)
            return self.portfolio.AllocationCalculator(vol_series, method="volatility")
        return self.portfolio.AllocationCalculator(selected, method=self.allocation_method)

    def AllocationUntilDate(self, tickers: List[str], end_date: str) -> pd.Series:
        """Calculate allocations using data up to a given end date."""
        LOGGER.info("Calculating allocation until %s for tickers: %s", end_date, tickers)
        data = self.fetcher.MarketDataAdapter(tickers)
        history = data.loc["2020-01-01":end_date]
        cumulative = history.pct_change(fill_method=None).add(1).prod() - 1
        top_count = max(int(len(tickers) * 0.3), 1)
        selected = cumulative.sort_values(ascending=False).head(top_count)
        if self.allocation_method == "volatility":
            vol_series = history.pct_change(fill_method=None).std().reindex(selected.index)
            return self.portfolio.AllocationCalculator(vol_series, method="volatility")
        return self.portfolio.AllocationCalculator(selected, method=self.allocation_method)

    def _RunBacktestForPeriod(
        self,
        allocations: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
        cash: float,
    ) -> tuple[float, list]:
        """Run a single backtest period and return final value and stats."""
        LOGGER.debug(
            "Running backtest period from %s to %s with cash=%s", start, end, cash
        )
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
        LOGGER.info(
            "Starting portfolio backtest with rebalance_months=%s", rebalance_months
        )
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")
        cash = self.initial_cash

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

        return cash / self.initial_cash - 1

    def BuyAndHoldReturn(self, tickers: List[str]) -> float:
        """Compute buy-and-hold return for equally weighted tickers."""
        LOGGER.info("Calculating buy-and-hold return for tickers: %s", tickers)
        if not tickers:
            raise ValueError("No tickers provided")
        weight = 1.0 / len(tickers)
        allocations = pd.Series(weight, index=tickers)
        return self.PortfolioBacktest(allocations)

    def IntervalBacktest(self, tickers: List[str], months: int) -> float:
        """Backtest with periodic rebalancing using a month interval."""
        LOGGER.info(
            "Running interval backtest for tickers: %s with months=%s", tickers, months
        )
        if months <= 0:
            raise ValueError("months must be positive")
        allocations = self.AllocationUntilDate(tickers, "2023-12-31")
        return self.PortfolioBacktest(allocations, rebalance_months=months)
