import pandas as pd
from typing import Dict, Optional


class ScoringEngine:
    """Aggregate indicator and momentum scores."""

    def IndicatorWeighter(self, normalized: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """Weight normalized indicators based on provided weights."""
        if weights is None:
            weights = {col: 1.0 for col in normalized.columns}
        for key in weights:
            if key not in normalized.columns:
                raise KeyError(f"Unknown indicator: {key}")
        weight_series = pd.Series(weights)
        aligned = normalized.reindex(columns=weight_series.index)
        weighted = aligned.mul(weight_series, axis=1)
        return weighted.sum(axis=1)

    def ScoreAggregator(self, weighted_scores: pd.Series, momentum_ranks: pd.Series, momentum_weight: float = 1.0) -> pd.Series:
        """Combine weighted indicator scores with momentum rank scores."""
        if not weighted_scores.index.equals(momentum_ranks.index):
            momentum_ranks = momentum_ranks.reindex(weighted_scores.index)
        momentum_component = -momentum_ranks * momentum_weight
        return weighted_scores + momentum_component

    def ScoreScaler(self, scores: pd.Series) -> pd.Series:
        """Scale composite scores to a 0-100 range."""
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return pd.Series(50.0, index=scores.index)
        return (scores - min_score) / (max_score - min_score) * 100

    def TestScoreAggregation(self) -> bool:
        """Ensure scoring is deterministic."""
        df_norm = pd.DataFrame({
            "IndA": [0.0, 1.0],
            "IndB": [1.0, 0.0],
        }, index=["A", "B"])
        weights = {"IndA": 2.0, "IndB": 1.0}
        momentum = pd.Series({"A": 1, "B": 2})
        first = self.ScoreScaler(
            self.ScoreAggregator(
                self.IndicatorWeighter(df_norm, weights), momentum, momentum_weight=0.5
            )
        )
        second = self.ScoreScaler(
            self.ScoreAggregator(
                self.IndicatorWeighter(df_norm, weights), momentum, momentum_weight=0.5
            )
        )
        pd.testing.assert_series_equal(first, second)
        return True
