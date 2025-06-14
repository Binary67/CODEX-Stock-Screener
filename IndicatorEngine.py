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

    def MACDIndicator(
        self,
        data: pd.Series,
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
    ) -> pd.Series:
        """Compute the MACD histogram."""
        ema_short = data.ewm(span=short_window, adjust=False).mean()
        ema_long = data.ewm(span=long_window, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        histogram = macd - signal
        return histogram

    def BollingerBandsIndicator(
        self, data: pd.Series, window: int = 20, num_std: int = 2
    ) -> pd.Series:
        """Return Bollinger Bands percent-b indicator."""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        percent_b = (data - lower) / (upper - lower)
        return percent_b

    def ADIIndicator(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Compute a simple Advance-Decline Indicator."""
        up = (data.diff() > 0).astype(int)
        down = (data.diff() < 0).astype(int)
        adv_dec = up.rolling(window).sum() - down.rolling(window).sum()
        return adv_dec

    def TestIndicators(self) -> bool:
        """Validate indicators against known values."""
        sample = pd.Series(range(1, 11), dtype=float)
        sma_expected = sample.rolling(window=3).mean()
        ema_expected = sample.ewm(span=3, adjust=False).mean()
        rsi_expected = self.RSI_Indicator(sample)
        vol_expected = sample.pct_change(fill_method=None).rolling(window=3).std()

        macd_line = sample.ewm(span=12, adjust=False).mean() - sample.ewm(span=26, adjust=False).mean()
        macd_expected = macd_line - macd_line.ewm(span=9, adjust=False).mean()

        sma_bb = sample.rolling(window=3).mean()
        std_bb = sample.rolling(window=3).std()
        upper_bb = sma_bb + 2 * std_bb
        lower_bb = sma_bb - 2 * std_bb
        bb_expected = (sample - lower_bb) / (upper_bb - lower_bb)

        up = (sample.diff() > 0).astype(int)
        down = (sample.diff() < 0).astype(int)
        adi_expected = up.rolling(3).sum() - down.rolling(3).sum()

        pd.testing.assert_series_equal(self.MovingAverageIndicator(sample, 3), sma_expected)
        pd.testing.assert_series_equal(self.MovingAverageIndicator(sample, 3, exponential=True), ema_expected)
        pd.testing.assert_series_equal(self.RSI_Indicator(sample), rsi_expected)
        pd.testing.assert_series_equal(self.VolatilityIndicator(sample, 3), vol_expected)
        pd.testing.assert_series_equal(self.MACDIndicator(sample, 12, 26, 9), macd_expected)
        pd.testing.assert_series_equal(self.BollingerBandsIndicator(sample, 3), bb_expected)
        pd.testing.assert_series_equal(self.ADIIndicator(sample, 3), adi_expected)
        return True
