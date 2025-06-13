import unittest
import pandas as pd
from IndicatorNormalizer import IndicatorNormalizer


class TestIndicatorNormalizer(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = IndicatorNormalizer()
        self.data = pd.DataFrame({
            "IndicatorA": [1.0, 2.0, 3.0],
            "IndicatorB": [2.0, 2.0, 2.0],
            "IndicatorC": [1.0, None, 3.0],
        }, index=["A", "B", "C"])

    def test_zscore_normalizer(self):
        cleaned = self.normalizer.MissingValueHandler(self.data, method="ffill")
        normalized = self.normalizer.ZScoreNormalizer(cleaned)
        self.assertAlmostEqual(normalized["IndicatorB"].abs().max(), 0.0)
        for col in normalized.columns:
            self.assertAlmostEqual(normalized[col].mean(), 0.0, places=6)

    def test_missing_value_handler_drop(self):
        dropped = self.normalizer.MissingValueHandler(self.data, method="drop")
        self.assertEqual(len(dropped), 2)

    def test_missing_value_handler_zero(self):
        filled = self.normalizer.MissingValueHandler(self.data, method="zero")
        self.assertFalse(filled.isna().any().any())
        self.assertEqual(filled.loc["B", "IndicatorC"], 0)

    def test_test_normalization(self):
        self.assertTrue(self.normalizer.TestNormalization())


if __name__ == "__main__":
    unittest.main()
