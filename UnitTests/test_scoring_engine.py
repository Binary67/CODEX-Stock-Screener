import unittest
import pandas as pd
from ScoringEngine import ScoringEngine


class TestScoringEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ScoringEngine()
        self.df_norm = pd.DataFrame({
            "IndA": [0.0, 1.0],
            "IndB": [1.0, 0.0],
        }, index=["A", "B"])
        self.weights = {"IndA": 2.0, "IndB": 1.0}
        self.momentum = pd.Series({"A": 1, "B": 2})
        self.momentum_df = pd.DataFrame({
            "Lookback_5": [1, 2],
            "Lookback_10": [1, 2],
        }, index=["A", "B"])

    def test_indicator_weighter(self):
        weighted = self.engine.IndicatorWeighter(self.df_norm, self.weights)
        self.assertAlmostEqual(weighted.loc["A"], 1.0)
        self.assertAlmostEqual(weighted.loc["B"], 2.0)

    def test_score_aggregator(self):
        weighted = self.engine.IndicatorWeighter(self.df_norm, self.weights)
        combined = self.engine.ScoreAggregator(weighted, self.momentum, momentum_weight=0.5)
        self.assertAlmostEqual(combined.loc["A"], weighted.loc["A"] - 0.5)
        self.assertAlmostEqual(combined.loc["B"], weighted.loc["B"] - 1.0)

    def test_momentum_weighter(self):
        agg = self.engine.MomentumWeighter(self.momentum_df)
        self.assertAlmostEqual(agg.loc["A"], 1.0)
        self.assertAlmostEqual(agg.loc["B"], 2.0)

    def test_score_aggregator_dataframe(self):
        weighted = self.engine.IndicatorWeighter(self.df_norm, self.weights)
        combined = self.engine.ScoreAggregator(weighted, self.momentum_df, momentum_weight=0.5)
        expected = weighted - self.engine.MomentumWeighter(self.momentum_df) * 0.5
        pd.testing.assert_series_equal(combined, expected)

    def test_score_aggregator_with_weights(self):
        weighted = self.engine.IndicatorWeighter(self.df_norm, self.weights)
        LookbackWeights = {"Lookback_5": 0.3, "Lookback_10": 0.7}
        combined = self.engine.ScoreAggregator(
            weighted,
            self.momentum_df,
            momentum_weight=1.0,
            lookback_weights=LookbackWeights,
        )
        expected = weighted - self.engine.MomentumWeighter(
            self.momentum_df,
            LookbackWeights,
        )
        pd.testing.assert_series_equal(combined, expected)

    def test_score_scaler(self):
        series = pd.Series([1.0, 3.0], index=["A", "B"])
        scaled = self.engine.ScoreScaler(series)
        self.assertAlmostEqual(scaled.loc["A"], 0.0)
        self.assertAlmostEqual(scaled.loc["B"], 100.0)

    def test_test_score_aggregation(self):
        self.assertTrue(self.engine.TestScoreAggregation())


if __name__ == "__main__":
    unittest.main()
