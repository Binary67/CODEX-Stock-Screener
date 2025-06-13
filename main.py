from MarketDataFetcher import MarketDataFetcher
from IndicatorEngine import IndicatorEngine


def main() -> None:
    tickers = ["AAPL", "MSFT"]
    fetcher = MarketDataFetcher()
    engine = IndicatorEngine()
    try:
        data = fetcher.MarketDataAdapter(tickers)
        close = data[tickers[0]].dropna()
        sma = engine.MovingAverageIndicator(close, 5)
        ema = engine.MovingAverageIndicator(close, 5, exponential=True)
        rsi = engine.RSI_Indicator(close)
        vol = engine.VolatilityIndicator(close, 5)
        print("Latest SMA:", sma.dropna().iloc[-1])
        print("Latest EMA:", ema.dropna().iloc[-1])
        print("Latest RSI:", rsi.dropna().iloc[-1])
        print("Latest Volatility:", vol.dropna().iloc[-1])
    except Exception as exc:
        print(f"Failed to fetch data: {exc}")


if __name__ == "__main__":
    main()
