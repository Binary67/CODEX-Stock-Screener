import logging
import pandas as pd
from typing import List, Union

LOGGER = logging.getLogger(__name__)


class MomentumEngine:
    """Calculate momentum scores and ranks."""

    def LookbackWindowValidator(self, lookbacks: Union[int, List[int]]) -> List[int]:
        """Validate user look-back periods and return a sorted unique list."""
        LOGGER.debug("Validating lookbacks: %s", lookbacks)
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
        """Compute risk-adjusted momentum for each look-back period."""
        LOGGER.debug("Calculating cumulative returns for lookbacks: %s", lookbacks)
        windows = self.LookbackWindowValidator(lookbacks)
        momentum = pd.DataFrame(index=data.columns)
        for window in windows:
            pct_return = data.pct_change(periods=window, fill_method=None).iloc[-1]
            volatility = data.pct_change(fill_method=None).rolling(window).std().iloc[-1].replace(0, pd.NA)
            risk_adjusted = (pct_return / volatility) * 100
            momentum[f"Lookback_{window}"] = risk_adjusted
        return momentum

    def MomentumRanker(self, data: pd.DataFrame, lookbacks: Union[int, List[int]]) -> pd.DataFrame:
        """Rank tickers by momentum for each look-back window."""
        LOGGER.info("Ranking momentum for lookbacks: %s", lookbacks)
        returns = self.CumulativeReturnCalculator(data, lookbacks)
        ranks = pd.DataFrame(index=returns.index)
        for column in returns.columns:
            sorted_series = returns[column].sort_values(ascending=False)
            rank = range(1, len(sorted_series) + 1)
            ranks[column] = pd.Series(rank, index=sorted_series.index)
        return ranks.reindex(returns.index)
