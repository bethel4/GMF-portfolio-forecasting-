# GMF Portfolio Forecasting

This repository provides a modular pipeline for time series forecasting and portfolio backtesting, using both ARIMA/LSTM models and robust backtesting utilities.

## Directory Structure

```
GMF-portfolio-forecasting-/
├── data/                # Raw and processed data
├── notebooks/           # Jupyter notebooks for exploration and visualization
│   └── 02_arima_lstm_modeling.ipynb
├── src/
│   ├── modeling.py      # ARIMA/LSTM reusable modeling functions
│   └── backtest/
│       └── backtesting.py   # Portfolio backtesting utilities
├── main.py              # Main script for running pipelines
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .gitignore
```

## Backtesting Module

The backtesting module (`src/backtest/backtesting.py`) provides functions to:

- **Fetch historical price data** for a list of tickers using Yahoo Finance.
- **Calculate daily returns** from price data.
- **Simulate portfolio returns** given asset weights.
- **Compute performance metrics** such as total return and Sharpe ratio.
- **Visualize cumulative returns** for both strategy and benchmark portfolios.

### Example Usage

```python
from src.backtest.backtesting import backtest_strategy

optimal_weights = {'AAPL': 0.5, 'MSFT': 0.5}
benchmark_weights = {'AAPL': 0.5, 'MSFT': 0.5}
tickers = ['AAPL', 'MSFT']

summary = backtest_strategy(optimal_weights, benchmark_weights, tickers)
print(summary)
```

## Modeling Module

Reusable ARIMA and LSTM modeling functions are provided in `src/modeling.py` for forecasting tasks.

## Notebooks

Exploratory analysis, model comparison, and visualization are performed in `notebooks/02_arima_lstm_modeling.ipynb`.