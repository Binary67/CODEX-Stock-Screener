import os
import unittest
import pandas as pd
from PortfolioEngine import PortfolioEngine


class TestPortfolioEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PortfolioEngine()
        self.scores = pd.Series({"AAA": 3.0, "BBB": 1.0, "CCC": 2.0})

    def test_portfolio_selector(self):
        result = self.engine.PortfolioSelector(self.scores, 2)
        self.assertListEqual(list(result.index), ["AAA", "CCC"])

    def test_allocation_calculator_equal(self):
        selected = self.engine.PortfolioSelector(self.scores, 2)
        alloc = self.engine.AllocationCalculator(selected, method="equal")
        self.assertAlmostEqual(alloc.sum(), 1.0)
        self.assertTrue((alloc == 0.5).all())

    def test_allocation_calculator_score(self):
        selected = self.engine.PortfolioSelector(self.scores, 2)
        alloc = self.engine.AllocationCalculator(selected, method="score")
        self.assertAlmostEqual(alloc.sum(), 1.0)
        expected = selected / selected.sum()
        pd.testing.assert_series_equal(alloc, expected)

    def test_volatility_adjusted_allocation(self):
        VolSeries = pd.Series({"AAA": 0.2, "BBB": 0.1, "CCC": 0.4})
        alloc = self.engine.VolatilityAdjustedAllocation(VolSeries)
        Expected = (1 / VolSeries) / (1 / VolSeries).sum()
        pd.testing.assert_series_equal(alloc, Expected)

    def test_allocation_calculator_volatility(self):
        VolSeries = pd.Series({"AAA": 0.2, "BBB": 0.1})
        alloc = self.engine.AllocationCalculator(VolSeries, method="volatility")
        Expected = (1 / VolSeries) / (1 / VolSeries).sum()
        pd.testing.assert_series_equal(alloc, Expected)

    def test_portfolio_exporter(self):
        selected = self.engine.PortfolioSelector(self.scores, 2)
        alloc = self.engine.AllocationCalculator(selected)
        self.engine.PortfolioExporter(selected, alloc, "test_port.csv", "test_port.json")
        self.assertTrue(os.path.exists("test_port.csv"))
        self.assertTrue(os.path.exists("test_port.json"))
        os.remove("test_port.csv")
        os.remove("test_port.json")

    def test_test_portfolio_selector(self):
        self.assertTrue(self.engine.TestPortfolioSelector())


if __name__ == "__main__":
    unittest.main()
