import pandas as pd
from typing import List, Union


class MomentumEngine:
    """Calculate momentum scores and ranks."""

    def LookbackWindowValidator(self, lookbacks: Union[int, List[int]]) -> List[int]:
        """Validate user look-back periods and return a sorted unique list."""
        if isinstance(lookbacks, int):
            lookbacks = [lookbacks]
        elif isinstance(lookbacks, (list, tuple, set)):
            lookbacks = list(lookbacks)
        else:
            raise TypeError("lookbacks must be an int or list of ints")
        validated = []
        for lb in lookbacks:
            try:
                lb_int = int(lb)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"invalid lookback: {lb}") from exc
            if lb_int <= 0:
                raise ValueError("lookback periods must be positive")
            validated.append(lb_int)
        return sorted(set(validated))

    def CumulativeReturnCalculator(self, data: pd.DataFrame, lookbacks: Union[int, List[int]]) -> pd.DataFrame:
        """Compute cumulative percent returns for each look-back period."""
        windows = self.LookbackWindowValidator(lookbacks)
        returns = pd.DataFrame(index=data.columns)
        for window in windows:
            ret = data.pct_change(periods=window).iloc[-1] * 100
            returns[f"Lookback_{window}"] = ret
        return returns

    def MomentumRanker(self, data: pd.DataFrame, lookbacks: Union[int, List[int]]) -> pd.DataFrame:
        """Rank tickers by momentum for each look-back window."""
        returns = self.CumulativeReturnCalculator(data, lookbacks)
        ranks = pd.DataFrame(index=returns.index)
        for column in returns.columns:
            sorted_series = returns[column].sort_values(ascending=False)
            rank = range(1, len(sorted_series) + 1)
            ranks[column] = pd.Series(rank, index=sorted_series.index)
        return ranks.reindex(returns.index)
