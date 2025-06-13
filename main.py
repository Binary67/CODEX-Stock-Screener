from MarketDataFetcher import MarketDataFetcher
from IndicatorEngine import IndicatorEngine
from IndicatorNormalizer import IndicatorNormalizer
import pandas as pd


def main() -> None:
    tickers = ["AAPL", "MSFT"]
    fetcher = MarketDataFetcher()
    engine = IndicatorEngine()
    normalizer = IndicatorNormalizer()
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
    except Exception as exc:
        print(f"Failed to fetch data: {exc}")


if __name__ == "__main__":
    main()
