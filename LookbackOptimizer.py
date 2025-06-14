import logging
import pandas as pd
from typing import Dict, List
from skopt import gp_minimize
from skopt.space import Real
from ConfigManager import ConfigManager
from MarketDataFetcher import MarketDataFetcher
from IndicatorEngine import IndicatorEngine
from IndicatorNormalizer import IndicatorNormalizer
from MomentumEngine import MomentumEngine
from ScoringEngine import ScoringEngine
from PortfolioEngine import PortfolioEngine
from BacktestingEngine import BacktestingEngine

LOGGER = logging.getLogger(__name__)


class LookbackOptimizer:
    """Optimize look-back weights using Bayesian optimization."""

    def __init__(self, manager: ConfigManager) -> None:
        self.manager = manager
        self.config = manager.LoadConfig()
        self.lookbacks = self.config.get("MomentumLookbacks", [])
        self.bounds = [Real(0.0, 3.0) for _ in self.lookbacks]
        LOGGER.info("LookbackOptimizer initialized for lookbacks: %s", self.lookbacks)

    def _RunBacktest(self, weights: Dict[str, float]) -> float:
        fetcher = MarketDataFetcher()
        engine = IndicatorEngine()
        normalizer = IndicatorNormalizer()
        momentum_engine = MomentumEngine()
        scorer = ScoringEngine()
        portfolio = PortfolioEngine()
        backtester = BacktestingEngine(fetcher, self.config)
        tickers = self.config.get("Tickers", [])
        training_end_date = self.manager.GetTrainingEndDate()
        data = fetcher.MarketDataAdapter(tickers)
        history = data.loc[:training_end_date]
        rows: List[Dict[str, float]] = []
        for ticker in tickers:
            close = history[ticker].dropna()
            if close.empty:
                continue
            params = self.config["IndicatorParameters"]
            rows.append(
                {
                    "Ticker": ticker,
                    "SMA": engine.MovingAverageIndicator(close, params["SMAWindow"]).iloc[-1],
                    "EMA": engine.MovingAverageIndicator(close, params["EMAWindow"], exponential=True).iloc[-1],
                    "RSI": engine.RSI_Indicator(close).iloc[-1],
                    "Volatility": engine.VolatilityIndicator(close, params["VolatilityWindow"]).iloc[-1],
                    "MACD": engine.MACDIndicator(
                        close,
                        params["MACDShort"],
                        params["MACDLong"],
                        params["MACDSignal"],
                    ).iloc[-1],
                    "BB": engine.BollingerBandsIndicator(
                        close,
                        params["BBWindow"],
                        params["BBStd"],
                    ).iloc[-1],
                    "ADI": engine.ADIIndicator(close, params["ADIWindow"]).iloc[-1],
                }
            )
        df = pd.DataFrame(rows).set_index("Ticker")
        df_clean = normalizer.MissingValueHandler(df, method="ffill")
        df_norm = normalizer.ZScoreNormalizer(df_clean)
        ranks = momentum_engine.MomentumRanker(history, self.lookbacks)
        weighted_scores = scorer.IndicatorWeighter(df_norm, self.config.get("IndicatorWeights", {}))
        combined = scorer.ScoreAggregator(
            weighted_scores,
            ranks,
            momentum_weight=self.config.get("MomentumWeight", 1.0),
            lookback_weights=weights,
        )
        scaled = scorer.ScoreScaler(combined)
        top_count = max(int(len(tickers) * 0.3), 1)
        top = portfolio.PortfolioSelector(scaled, top_count)
        allocation_method = self.config.get("AllocationMethod", "equal")
        if allocation_method == "volatility":
            vol_series = df_clean.loc[top.index, "Volatility"]
            alloc = portfolio.AllocationCalculator(vol_series, method="volatility")
        else:
            alloc = portfolio.AllocationCalculator(top, method=allocation_method)
        return backtester.PortfolioBacktest(
            alloc, self.config.get("RebalanceIntervalMonths", 0)
        )

    def _Objective(self, weight_list: List[float]) -> float:
        weight_dict = {f"Lookback_{lb}": w for lb, w in zip(self.lookbacks, weight_list)}
        result = self._RunBacktest(weight_dict)
        LOGGER.info("Weights %s yielded return %.4f", weight_dict, result)
        return -result

    def Optimize(self, n_calls: int = 20) -> Dict[str, float]:
        initial = min(5, n_calls)
        res = gp_minimize(
            self._Objective,
            self.bounds,
            n_calls=n_calls,
            n_initial_points=initial,
            random_state=0,
        )
        best = {f"Lookback_{lb}": w for lb, w in zip(self.lookbacks, res.x)}
        LOGGER.info("Optimization completed. Best weights: %s", best)
        self.config["LookbackWeights"] = best
        return best
