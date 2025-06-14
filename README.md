# Stock Screener

A lightweight Python tool for ranking stocks using several technical indicators. Prices are retrieved from Yahoo Finance through the `yfinance` library.

## Features

- **Data Download**: Fetches historical price data for multiple tickers. Hourly data is automatically retrieved in two-week segments to comply with Yahoo Finance limits.
- **Technical Metrics**: Calculates RSI, short and long moving averages, Stochastic Oscillator, ATR, OBV, ADX, and Aroon Up/Down.
- **Weighted Ranking**: Scores each ticker by normalizing the metrics and applying ranking algorithm.
- **Customizable Weights**: Generate a default criteria matrix or supply your own to emphasize specific indicators.
- 
## Data Source

All price data is retrieved from Yahoo Finance via the `yfinance` package.

## Usage

Edit `Parameters.yaml` to configure the screener. A new `TrainingEndDate`
setting specifies the final date used for indicator calculations and momentum
ranking. Example:

```yaml
TrainingEndDate: '2023-12-31'
```

Run the application:

```bash
python main.py
```
