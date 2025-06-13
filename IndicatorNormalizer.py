import pandas as pd


class IndicatorNormalizer:
    """Normalize indicator values across tickers."""

    def ZScoreNormalizer(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score standardization across tickers per indicator."""
        means = data.mean(axis=0)
        stds = data.std(axis=0, ddof=0)
        normalized = (data - means) / stds.replace(0, pd.NA)
        normalized = normalized.fillna(0)
        return normalized

    def MissingValueHandler(self, data: pd.DataFrame, method: str = "drop") -> pd.DataFrame:
        """Impute or drop missing values to ensure continuity."""
        if method == "drop":
            return data.dropna()
        if method == "ffill":
            return data.ffill().bfill()
        if method == "zero":
            return data.fillna(0)
        raise ValueError(f"Unknown method: {method}")

    def TestNormalization(self) -> bool:
        """Validate normalization logic and edge cases."""
        df = pd.DataFrame({
            "IndicatorA": [1.0, 2.0, 3.0],
            "IndicatorB": [1.0, 1.0, 1.0],
            "IndicatorC": [1.0, pd.NA, 3.0],
        }, index=["A", "B", "C"])

        cleaned = self.MissingValueHandler(df, method="ffill")
        assert not cleaned.isna().any().any()

        normalized = self.ZScoreNormalizer(cleaned)
        assert (normalized.loc[:, "IndicatorB"] == 0).all()
        assert abs(normalized.mean()).max() < 1e-8
        return True
