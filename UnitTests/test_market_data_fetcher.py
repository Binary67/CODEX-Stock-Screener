import os
import unittest
from unittest.mock import patch
import pandas as pd
from market_data_fetcher import MarketDataFetcher


class TestMarketDataAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.fetcher = MarketDataFetcher(cache_dir="test_cache")
        os.makedirs("test_cache", exist_ok=True)

    def tearDown(self) -> None:
        for f in os.listdir("test_cache"):
            os.remove(os.path.join("test_cache", f))
        os.rmdir("test_cache")

    @patch("yfinance.download")
    def test_successful_fetch(self, mock_download):
        index = pd.date_range("2020-01-01", periods=2)
        df = pd.DataFrame({"Adj Close": [10.0, 12.0]}, index=index)
        mock_download.return_value = df
        data = self.fetcher.MarketDataAdapter(["AAPL"])
        self.assertIn("AAPL", data.columns)
        self.assertEqual(len(data), 2)

    @patch("yfinance.download")
    def test_failure_fetch(self, mock_download):
        mock_download.return_value = pd.DataFrame()
        with self.assertRaises(RuntimeError):
            self.fetcher.MarketDataAdapter(["INVALID"]) 


if __name__ == "__main__":
    unittest.main()
