import unittest
import pandas as pd
from unittest.mock import patch
from BacktestingEngine import BacktestingEngine
from MarketDataFetcher import MarketDataFetcher
from PortfolioEngine import PortfolioEngine


class TestBacktestingEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.fetcher = MarketDataFetcher()
        self.engine = BacktestingEngine(self.fetcher)
        self.engine_config = BacktestingEngine(self.fetcher, {"InitialCash": 5000})
        dates_train = pd.date_range("2020-01-01", periods=4, freq="D")
        dates_test = pd.date_range("2024-01-01", periods=90, freq="D")
        all_dates = dates_train.append(dates_test)
        self.data = pd.DataFrame(
            {
                "AAA": list(range(1, 95)),
                "BBB": [1] * 94,
                "CCC": list(range(94, 0, -1)),
            },
            index=all_dates,
        )

    def fake_adapter(self, tickers):
        return self.data[tickers]

    @patch.object(MarketDataFetcher, "MarketDataAdapter")
    def test_allocation_and_backtest(self, mock_adapter):
        mock_adapter.side_effect = self.fake_adapter
        alloc = self.engine.AllocationFromHistory(["AAA", "BBB", "CCC"])
        self.assertListEqual(list(alloc.index), ["AAA"])
        result = self.engine.PortfolioBacktest(alloc, rebalance_months=0)
        self.assertGreater(result, 0)

    @patch.object(MarketDataFetcher, "MarketDataAdapter")
    def test_buy_and_hold_return(self, mock_adapter):
        mock_adapter.side_effect = self.fake_adapter
        result = self.engine.BuyAndHoldReturn(["AAA", "BBB"])
        self.assertGreater(result, 0)

    @patch.object(MarketDataFetcher, "MarketDataAdapter")
    def test_interval_backtest(self, mock_adapter):
        mock_adapter.side_effect = self.fake_adapter
        result = self.engine.IntervalBacktest(["AAA", "BBB", "CCC"], 1)
        self.assertGreater(result, 0)

    def test_initial_cash_from_config(self):
        self.assertEqual(self.engine_config.initial_cash, 5000)

    @patch.object(MarketDataFetcher, "MarketDataAdapter")
    def test_portfolio_backtest_rebalance(self, mock_adapter):
        mock_adapter.side_effect = self.fake_adapter
        alloc = self.engine.AllocationFromHistory(["AAA", "BBB", "CCC"])
        result = self.engine.PortfolioBacktest(alloc, rebalance_months=1)
        self.assertGreater(result, 0)

    @patch.object(MarketDataFetcher, "MarketDataAdapter")
    @patch.object(PortfolioEngine, "AllocationCalculator")
    def test_allocation_method_respected(self, mock_alloc_calc, mock_adapter):
        mock_adapter.side_effect = self.fake_adapter
        mock_alloc_calc.return_value = pd.Series([1.0], index=["AAA"])
        engine = BacktestingEngine(self.fetcher, {"AllocationMethod": "equal"})
        engine.AllocationFromHistory(["AAA", "BBB", "CCC"])
        _, kwargs = mock_alloc_calc.call_args
        self.assertEqual(kwargs.get("method"), "equal")


if __name__ == "__main__":
    unittest.main()
