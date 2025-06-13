from market_data_fetcher import MarketDataFetcher


def main() -> None:
    tickers = ["AAPL", "MSFT"]
    fetcher = MarketDataFetcher()
    try:
        data = fetcher.MarketDataAdapter(tickers)
        print(data.tail())
    except Exception as exc:
        print(f"Failed to fetch data: {exc}")


if __name__ == "__main__":
    main()
