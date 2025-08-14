# src/backtesting.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def fetch_backtest_data(tickers, start='2024-08-01', end='2025-07-31'):
    """Download historical adjusted close prices for backtesting."""
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

def calculate_daily_returns(prices):
    """Calculate daily percentage returns from price data."""
    return prices.pct_change().dropna()

def simulate_portfolio(daily_returns, weights):
    """Simulate portfolio cumulative returns given daily returns and weights."""
    weights_array = np.array([weights[ticker] for ticker in daily_returns.columns])
    portfolio_returns = daily_returns.dot(weights_array)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return cumulative_returns, portfolio_returns

def portfolio_performance(portfolio_returns):
    """Calculate total return and Sharpe ratio."""
    total_return = portfolio_returns.cumprod().iloc[-1] - 1
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    return total_return, sharpe_ratio

def backtest_strategy(optimal_weights, benchmark_weights, tickers):
    """Run backtesting for strategy and benchmark."""
    # Fetch backtesting prices
    prices = fetch_backtest_data(tickers)
    
    # Daily returns
    daily_returns = calculate_daily_returns(prices)
    
    # Simulate portfolios
    strategy_cum, strategy_daily = simulate_portfolio(daily_returns, optimal_weights)
    benchmark_cum, benchmark_daily = simulate_portfolio(daily_returns, benchmark_weights)
    
    # Performance metrics
    strategy_return, strategy_sharpe = portfolio_performance(strategy_daily)
    benchmark_return, benchmark_sharpe = portfolio_performance(benchmark_daily)
    
    # Plot cumulative returns
    plt.figure(figsize=(12,6))
    plt.plot(strategy_cum, label='Strategy Portfolio')
    plt.plot(benchmark_cum, label='Benchmark Portfolio', linestyle='--')
    plt.title('Backtesting: Strategy vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()
    
    summary = {
        'strategy': {
            'total_return': strategy_return,
            'sharpe_ratio': strategy_sharpe
        },
        'benchmark': {
            'total_return': benchmark_return,
            'sharpe_ratio': benchmark_sharpe
        }
    }
    return summary
