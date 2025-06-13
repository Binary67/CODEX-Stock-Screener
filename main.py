import pandas as pd
import yfinance as yf
from src.MomentumEngine import MomentumEngine


def main():
    tickers = ['AAPL', 'MSFT', 'GOOG']
    data = yf.download(tickers, period='3mo')['Close']
    engine = MomentumEngine([5, 20])
    ranking = engine.MomentumRanker(data)
    print(ranking)


if __name__ == '__main__':
    main()
