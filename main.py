import logging
from MarketDataFetcher import MarketDataFetcher
from IndicatorEngine import IndicatorEngine
from IndicatorNormalizer import IndicatorNormalizer
from MomentumEngine import MomentumEngine
from ScoringEngine import ScoringEngine
from PortfolioEngine import PortfolioEngine
from BacktestingEngine import BacktestingEngine
from ConfigManager import ConfigManager
from LoggingManager import LoggingManager
from LookbackOptimizer import LookbackOptimizer
import pandas as pd

LOGGER = logging.getLogger(__name__)


def main() -> None:
    LoggingManager.SetupLogging()
    manager = ConfigManager()
    config = manager.LoadConfig()
    optimizer = LookbackOptimizer(manager)
    best_weights = optimizer.Optimize(n_calls=5)
    config["LookbackWeights"] = best_weights
    training_end_date = manager.GetTrainingEndDate()
    tickers = config.get("Tickers", [])
    fetcher = MarketDataFetcher()
    engine = IndicatorEngine()
    normalizer = IndicatorNormalizer()
    momentum_engine = MomentumEngine()
    scorer = ScoringEngine()
    portfolio = PortfolioEngine()
    try:
        data = fetcher.MarketDataAdapter(tickers)
        History = data.loc[:training_end_date]
        rows = []
        for ticker in tickers:
            close = History[ticker].dropna()
            if close.empty:
                continue
            sma = engine.MovingAverageIndicator(
                close, config["IndicatorParameters"]["SMAWindow"]
            ).iloc[-1]
            ema = engine.MovingAverageIndicator(
                close,
                config["IndicatorParameters"]["EMAWindow"],
                exponential=True,
            ).iloc[-1]
            rsi = engine.RSI_Indicator(close).iloc[-1]
            vol = engine.VolatilityIndicator(
                close, config["IndicatorParameters"]["VolatilityWindow"]
            ).iloc[-1]
            macd = engine.MACDIndicator(
                close,
                config["IndicatorParameters"]["MACDShort"],
                config["IndicatorParameters"]["MACDLong"],
                config["IndicatorParameters"]["MACDSignal"],
            ).iloc[-1]
            bb = engine.BollingerBandsIndicator(
                close,
                config["IndicatorParameters"]["BBWindow"],
                config["IndicatorParameters"]["BBStd"],
            ).iloc[-1]
            adi = engine.ADIIndicator(
                close,
                config["IndicatorParameters"]["ADIWindow"],
            ).iloc[-1]
            rows.append({
                "Ticker": ticker,
                "SMA": sma,
                "EMA": ema,
                "RSI": rsi,
                "Volatility": vol,
                "MACD": macd,
                "BB": bb,
                "ADI": adi,
            })

        df = pd.DataFrame(rows).set_index("Ticker")
        df_clean = normalizer.MissingValueHandler(df, method="ffill")
        df_norm = normalizer.ZScoreNormalizer(df_clean)
        
        lookbacks = config.get("MomentumLookbacks", [])
        risk_adjusted_ranks = momentum_engine.MomentumRanker(History, lookbacks)
        lookback_weights = config.get(
            "LookbackWeights", {f"Lookback_{lb}": 1.0 for lb in lookbacks}
        )

        weights = config.get("IndicatorWeights", {})
        weighted = scorer.IndicatorWeighter(df_norm, weights)
        combined = scorer.ScoreAggregator(
            weighted,
            risk_adjusted_ranks,
            momentum_weight=config.get("MomentumWeight", 1.0),
            lookback_weights=lookback_weights,
        )
        scaled = scorer.ScoreScaler(combined)

        TopCount = max(int(len(tickers) * 0.3), 1)
        top = portfolio.PortfolioSelector(scaled, TopCount)
        allocation_method = config.get("AllocationMethod", "equal")
        if allocation_method == "volatility":
            VolatilitySeries = df_clean.loc[top.index, "Volatility"]
            alloc = portfolio.AllocationCalculator(
                VolatilitySeries, method="volatility"
            )
        else:
            alloc = portfolio.AllocationCalculator(top, method=allocation_method)
        portfolio.PortfolioExporter(
            top,
            alloc,
            config.get("CsvPath", "portfolio.csv"),
            config.get("JsonPath", "portfolio.json"),
        )
        print("Top selections:\n", top)
        print("Allocations:\n", alloc)
        LOGGER.info("Top selections:\n%s", top)
        LOGGER.info("Allocations:\n%s", alloc)

        backtester = BacktestingEngine(fetcher, config)
        hist_alloc = backtester.AllocationFromHistory(tickers)
        interval = config.get("RebalanceIntervalMonths", 0)
        backtest_return = backtester.PortfolioBacktest(hist_alloc, interval)
        baseline_return = backtester.BuyAndHoldReturn(hist_alloc.index)
        if interval:
            message = f"Interval Backtest return: {backtest_return:.2%}"
        else:
            message = f"Backtest return: {backtest_return:.2%}"
        print(message)
        print(f"Buy and Hold return: {baseline_return:.2%}")
        LOGGER.info(message)
        LOGGER.info("Buy and Hold return: %.2f%%", baseline_return * 100)
    except Exception as exc:
        LOGGER.exception("Failed to fetch data: %s", exc)
        print(f"Failed to fetch data: {exc}")


if __name__ == "__main__":
    main()
