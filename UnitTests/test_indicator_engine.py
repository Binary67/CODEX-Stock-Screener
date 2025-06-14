import unittest
import pandas as pd
from IndicatorEngine import IndicatorEngine


class TestIndicatorEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = IndicatorEngine()
        self.data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

    def test_moving_average(self):
        sma = self.engine.MovingAverageIndicator(self.data, 3)
        self.assertAlmostEqual(sma.iloc[-1], 9.0)
        ema = self.engine.MovingAverageIndicator(self.data, 3, exponential=True)
        expected_ema = self.data.ewm(span=3, adjust=False).mean().iloc[-1]
        self.assertAlmostEqual(ema.iloc[-1], expected_ema)

    def test_rsi_indicator(self):
        rsi = self.engine.RSI_Indicator(self.data, 2)
        self.assertTrue(((0 <= rsi.dropna()) & (rsi.dropna() <= 100)).all())

    def test_volatility_indicator(self):
        vol = self.engine.VolatilityIndicator(self.data, 3)
        expected_vol = self.data.pct_change(fill_method=None).rolling(3).std().iloc[-1]
        self.assertAlmostEqual(vol.iloc[-1], expected_vol)

    def test_macd_indicator(self):
        hist = self.engine.MACDIndicator(self.data)
        ema_short = self.data.ewm(span=12, adjust=False).mean()
        ema_long = self.data.ewm(span=26, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=9, adjust=False).mean()
        expected = macd - signal
        self.assertAlmostEqual(hist.dropna().iloc[-1], expected.dropna().iloc[-1])

    def test_bollinger_bands_indicator(self):
        bb = self.engine.BollingerBandsIndicator(self.data, 3)
        sma = self.data.rolling(3).mean()
        std = self.data.rolling(3).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        expected = (self.data - lower) / (upper - lower)
        self.assertAlmostEqual(bb.dropna().iloc[-1], expected.dropna().iloc[-1])

    def test_adi_indicator(self):
        adi = self.engine.ADIIndicator(self.data, 3)
        up = (self.data.diff() > 0).astype(int)
        down = (self.data.diff() < 0).astype(int)
        expected = up.rolling(3).sum() - down.rolling(3).sum()
        self.assertAlmostEqual(adi.dropna().iloc[-1], expected.dropna().iloc[-1])

    def test_test_indicators(self):
        self.assertTrue(self.engine.TestIndicators())


if __name__ == "__main__":
    unittest.main()
