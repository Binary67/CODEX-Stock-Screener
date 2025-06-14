import os
import unittest
import yaml
import pandas as pd
from unittest.mock import patch
from LookbackOptimizer import LookbackOptimizer
from ConfigManager import ConfigManager
from MarketDataFetcher import MarketDataFetcher


class TestLookbackOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        config = {
            "Tickers": ["AAA", "BBB"],
            "IndicatorParameters": {
                "SMAWindow": 2,
                "EMAWindow": 2,
                "VolatilityWindow": 2,
                "MACDShort": 2,
                "MACDLong": 3,
                "MACDSignal": 1,
                "BBWindow": 2,
                "BBStd": 1,
                "ADIWindow": 2,
            },
            "MomentumLookbacks": [1, 2],
            "IndicatorWeights": {
                "SMA": 1.0,
                "EMA": 1.0,
                "RSI": 1.0,
                "Volatility": 1.0,
                "MACD": 1.0,
                "BB": 1.0,
                "ADI": 1.0,
            },
            "MomentumWeight": 1.0,
            "LookbackWeights": {"Lookback_1": 1.0, "Lookback_2": 1.0},
            "AllocationMethod": "equal",
            "RebalanceIntervalMonths": 0,
            "TrainingEndDate": "2020-02-20",
            "InitialCash": 1000,
        }
        with open("opt_params.yaml", "w", encoding="utf-8") as file:
            yaml.dump(config, file)
        self.manager = ConfigManager("opt_params.yaml")
        self.optimizer = LookbackOptimizer(self.manager)

    def tearDown(self) -> None:
        os.remove("opt_params.yaml")

    def fake_data(self, tickers):
        index = pd.date_range("2020-01-01", periods=60, freq="D")
        data = pd.DataFrame({t: range(1, 61) for t in tickers}, index=index)
        return data

    @patch.object(MarketDataFetcher, "MarketDataAdapter")
    def test_optimize_returns_dict(self, mock_adapter):
        mock_adapter.side_effect = self.fake_data
        result = self.optimizer.Optimize(n_calls=2)
        self.assertIsInstance(result, dict)
        self.assertIn("Lookback_1", result)
        self.assertIn("Lookback_2", result)


if __name__ == "__main__":
    unittest.main()
