import pandas as pd
from typing import Optional


class PortfolioEngine:
    """Construct portfolios based on composite scores."""

    def PortfolioSelector(self, scores: pd.Series, top_n: int) -> pd.Series:
        """Return the top N tickers sorted by composite score."""
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if scores.empty:
            raise ValueError("No scores provided")
        sorted_scores = scores.sort_values(ascending=False).dropna()
        return sorted_scores.head(top_n)

    def AllocationCalculator(self, selected: pd.Series, method: str = "equal") -> pd.Series:
        """Calculate allocations for selected tickers."""
        if selected.empty:
            raise ValueError("No tickers selected")
        if method not in ("equal", "score"):
            raise ValueError("Unknown allocation method")
        if method == "equal":
            weight = 1.0 / len(selected)
            return pd.Series(weight, index=selected.index)
        total = selected.sum()
        if total == 0:
            return pd.Series(1.0 / len(selected), index=selected.index)
        return selected / total

    def PortfolioExporter(self, selected: pd.Series, allocations: pd.Series, csv_path: str, json_path: str) -> pd.DataFrame:
        """Export portfolio selections to CSV and JSON files."""
        df = pd.DataFrame({"Score": selected, "Allocation": allocations})
        df.to_csv(csv_path)
        df.to_json(json_path, orient="records")
        return df

    def TestPortfolioSelector(self) -> bool:
        """Validate portfolio selection logic."""
        scores = pd.Series({"AAA": 3.0, "BBB": 1.0, "CCC": 2.0})
        top = self.PortfolioSelector(scores, 2)
        assert list(top.index) == ["AAA", "CCC"]
        alloc_equal = self.AllocationCalculator(top, method="equal")
        alloc_score = self.AllocationCalculator(top, method="score")
        assert abs(alloc_equal.sum() - 1.0) < 1e-8
        assert abs(alloc_score.sum() - 1.0) < 1e-8
        return True
