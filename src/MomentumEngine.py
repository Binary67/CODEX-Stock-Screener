import pandas as pd
from typing import Iterable, List


class MomentumEngine:
    """Calculate momentum rankings based on cumulative returns."""

    def __init__(self, lookbacks: Iterable[int]):
        self.Lookbacks = self.LookbackWindowValidator(lookbacks)

    @staticmethod
    def LookbackWindowValidator(lookbacks: Iterable[int]) -> List[int]:
        """Validate lookback windows and return a sorted list of positive integers."""
        if lookbacks is None:
            raise ValueError("Lookback windows must be provided")
        if isinstance(lookbacks, int):
            lookback_list = [lookbacks]
        else:
            # convert each element to int
            try:
                lookback_list = [int(lb) for lb in lookbacks]
            except Exception as exc:  # catch non-convertible
                raise ValueError("Invalid lookback window values") from exc
        # ensure positives
        lookback_list = [lb for lb in lookback_list if lb > 0]
        if not lookback_list:
            raise ValueError("Lookback windows must contain positive integers")
        # remove duplicates and sort
        lookback_list = sorted(set(lookback_list))
        return lookback_list

    def CumulativeReturnCalculator(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute cumulative percent return for each lookback window."""
        if prices.empty:
            raise ValueError("Price data is empty")
        returns = {}
        for lb in self.Lookbacks:
            returns[lb] = prices.pct_change(lb).iloc[-1]
        return pd.DataFrame(returns)

    def MomentumRanker(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Rank tickers by momentum using cumulative returns."""
        cumulative_returns = self.CumulativeReturnCalculator(prices)
        cumulative_returns['AverageReturn'] = cumulative_returns.mean(axis=1)
        ranked = cumulative_returns.sort_values('AverageReturn', ascending=False)
        ranked['Rank'] = range(1, len(ranked) + 1)
        return ranked

