from MarketDataFetcher import MarketDataFetcher
from IndicatorEngine import IndicatorEngine
from IndicatorNormalizer import IndicatorNormalizer
from MomentumEngine import MomentumEngine
from ScoringEngine import ScoringEngine
import pandas as pd


def main() -> None:
    tickers = ["AAPL", "MSFT"]
    fetcher = MarketDataFetcher()
    engine = IndicatorEngine()
    normalizer = IndicatorNormalizer()
    momentum = MomentumEngine()
    scorer = ScoringEngine()
    try:
        data = fetcher.MarketDataAdapter(tickers)
        rows = []
        for ticker in tickers:
            close = data[ticker].dropna()
            sma = engine.MovingAverageIndicator(close, 5).dropna().iloc[-1]
            ema = engine.MovingAverageIndicator(close, 5, exponential=True).dropna().iloc[-1]
            rsi = engine.RSI_Indicator(close).dropna().iloc[-1]
            vol = engine.VolatilityIndicator(close, 5).dropna().iloc[-1]
            rows.append({"Ticker": ticker, "SMA": sma, "EMA": ema, "RSI": rsi, "Volatility": vol})

        df = pd.DataFrame(rows).set_index("Ticker")
        df_clean = normalizer.MissingValueHandler(df, method="ffill")
        df_norm = normalizer.ZScoreNormalizer(df_clean)
        print(df_norm)
        ranks = momentum.MomentumRanker(data, [5])
        print(ranks)

        weights = {"SMA": 1.0, "EMA": 1.0, "RSI": 1.0, "Volatility": 1.0}
        weighted = scorer.IndicatorWeighter(df_norm, weights)
        combined = scorer.ScoreAggregator(weighted, ranks["Lookback_5"], momentum_weight=1.0)
        scaled = scorer.ScoreScaler(combined)
        print(scaled)
    except Exception as exc:
        print(f"Failed to fetch data: {exc}")


if __name__ == "__main__":
    main()
