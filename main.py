from MarketDataFetcher import MarketDataFetcher
from IndicatorEngine import IndicatorEngine
from IndicatorNormalizer import IndicatorNormalizer
from MomentumEngine import MomentumEngine
from ScoringEngine import ScoringEngine
from PortfolioEngine import PortfolioEngine
from ConfigManager import ConfigManager
import pandas as pd


def main() -> None:
    manager = ConfigManager()
    config = manager.LoadConfig()
    tickers = config.get("Tickers", [])
    fetcher = MarketDataFetcher()
    engine = IndicatorEngine()
    normalizer = IndicatorNormalizer()
    momentum = MomentumEngine()
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
            rows.append({"Ticker": ticker, "SMA": sma, "EMA": ema, "RSI": rsi, "Volatility": vol})

        df = pd.DataFrame(rows).set_index("Ticker")
        df_clean = normalizer.MissingValueHandler(df, method="ffill")
        df_norm = normalizer.ZScoreNormalizer(df_clean)
        print(df_norm)
        lookbacks = config.get("MomentumLookbacks", [])
        ranks = momentum.MomentumRanker(data, lookbacks)
        lookback_weights = config.get(
            "LookbackWeights", {f"Lookback_{lb}": 1.0 for lb in lookbacks}
        )
        print(ranks)

        weights = config.get("IndicatorWeights", {})
        weighted = scorer.IndicatorWeighter(df_norm, weights)
        combined = scorer.ScoreAggregator(
            weighted,
            ranks,
            momentum_weight=config.get("MomentumWeight", 1.0),
            lookback_weights=lookback_weights,
        )
        scaled = scorer.ScoreScaler(combined)
        print(scaled)

        top = portfolio.PortfolioSelector(scaled, config.get("TopN", 1))
        alloc = portfolio.AllocationCalculator(
            top, method=config.get("AllocationMethod", "equal")
        )
        portfolio.PortfolioExporter(
            top,
            alloc,
            config.get("CsvPath", "portfolio.csv"),
            config.get("JsonPath", "portfolio.json"),
        )
        print(top)
        print(alloc)
    except Exception as exc:
        print(f"Failed to fetch data: {exc}")


if __name__ == "__main__":
    main()
