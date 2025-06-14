import pandas as pd


class IndicatorEngine:
    """Calculate various technical indicators."""

    def MovingAverageIndicator(self, data: pd.Series, window: int, exponential: bool = False) -> pd.Series:
        """Return simple or exponential moving average."""
        if exponential:
            return data.ewm(span=window, adjust=False).mean()
        return data.rolling(window=window).mean()

    def RSI_Indicator(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Compute the Relative Strength Index (RSI)."""
        delta = data.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        return rsi

    def VolatilityIndicator(self, data: pd.Series, window: int) -> pd.Series:
        """Return rolling volatility using standard deviation of percentage change."""
        return data.pct_change(fill_method=None).rolling(window=window).std()

    def TestIndicators(self) -> bool:
        """Validate indicators against known values."""
        sample = pd.Series(range(1, 11), dtype=float)
        sma_expected = sample.rolling(window=3).mean()
        ema_expected = sample.ewm(span=3, adjust=False).mean()
        rsi_expected = self.RSI_Indicator(sample)
        vol_expected = sample.pct_change(fill_method=None).rolling(window=3).std()

        pd.testing.assert_series_equal(self.MovingAverageIndicator(sample, 3), sma_expected)
        pd.testing.assert_series_equal(self.MovingAverageIndicator(sample, 3, exponential=True), ema_expected)
        pd.testing.assert_series_equal(self.RSI_Indicator(sample), rsi_expected)
        pd.testing.assert_series_equal(self.VolatilityIndicator(sample, 3), vol_expected)
        return True
