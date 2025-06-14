from MarketDataFetcher import MarketDataFetcher
from IndicatorEngine import IndicatorEngine
from IndicatorNormalizer import IndicatorNormalizer
from MomentumEngine import MomentumEngine
from ScoringEngine import ScoringEngine
from PortfolioEngine import PortfolioEngine
from BacktestingEngine import BacktestingEngine
from ConfigManager import ConfigManager
import pandas as pd


def main() -> None:
    manager = ConfigManager()
    config = manager.LoadConfig()
    tickers = config.get("Tickers", [])
    fetcher = MarketDataFetcher()
    engine = IndicatorEngine()
    normalizer = IndicatorNormalizer()
    momentum_engine = MomentumEngine()
    scorer = ScoringEngine()
    portfolio = PortfolioEngine()
    try:
        data = fetcher.MarketDataAdapter(tickers)
        rows = []
        for ticker in tickers:
            close = data[ticker].dropna()
            sma = engine.MovingAverageIndicator(
                close, config["IndicatorParameters"]["SMAWindow"]
            ).dropna().iloc[-1]
            ema = engine.MovingAverageIndicator(
                close,
                config["IndicatorParameters"]["EMAWindow"],
                exponential=True,
            ).dropna().iloc[-1]
            rsi = engine.RSI_Indicator(close).dropna().iloc[-1]
            vol = engine.VolatilityIndicator(
                close, config["IndicatorParameters"]["VolatilityWindow"]
            ).dropna().iloc[-1]
            macd = engine.MACDIndicator(
                close,
                config["IndicatorParameters"]["MACDShort"],
                config["IndicatorParameters"]["MACDLong"],
                config["IndicatorParameters"]["MACDSignal"],
            ).dropna().iloc[-1]
            bb = engine.BollingerBandsIndicator(
                close,
                config["IndicatorParameters"]["BBWindow"],
                config["IndicatorParameters"]["BBStd"],
            ).dropna().iloc[-1]
            adi = engine.ADIIndicator(
                close,
                config["IndicatorParameters"]["ADIWindow"],
            ).dropna().iloc[-1]
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
        risk_adjusted_ranks = momentum_engine.MomentumRanker(data, lookbacks)
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

        backtester = BacktestingEngine(fetcher, config)
        hist_alloc = backtester.AllocationFromHistory(tickers)
        interval = config.get("RebalanceIntervalMonths", 0)
        backtest_return = backtester.PortfolioBacktest(hist_alloc, interval)
        baseline_return = backtester.BuyAndHoldReturn(tickers)
        if interval:
            print(f"Interval Backtest return: {backtest_return:.2%}")
        else:
            print(f"Backtest return: {backtest_return:.2%}")
        print(f"Buy and Hold return: {baseline_return:.2%}")
    except Exception as exc:
        print(f"Failed to fetch data: {exc}")


if __name__ == "__main__":
    main()
