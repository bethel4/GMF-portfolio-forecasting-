# src/portfolio_optimization.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns

def fetch_data(tickers, start='2015-01-01', end='2025-01-01'):
    """Download historical adjusted close prices for given tickers."""
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

def calculate_annualized_returns(data):
    """Compute daily returns and annualized average returns."""
    daily_returns = data.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    return daily_returns, annual_returns

def optimize_portfolio(tsla_forecasted_return, daily_returns):
    """
    Optimize portfolio using MPT with forecasted TSLA return and historical BND & SPY.
    
    Args:
        tsla_forecasted_return (float): expected annual return of TSLA from forecast
        daily_returns (DataFrame): daily returns for TSLA, BND, SPY
    """
    tickers = daily_returns.columns.tolist()
    
    # Replace TSLA's historical return with forecasted return
    mu = expected_returns.mean_historical_return(daily_returns)
    mu['TSLA'] = tsla_forecasted_return
    
    # Covariance matrix
    S = risk_models.sample_cov(daily_returns)
    
    # Efficient Frontier
    ef = EfficientFrontier(mu, S)
    # Maximum Sharpe Ratio Portfolio
    max_sharpe = ef.max_sharpe()
    cleaned_max_sharpe = ef.clean_weights()
    ret_max_sharpe, vol_max_sharpe, sharpe_max = ef.portfolio_performance()
    
    # Minimum Volatility Portfolio
    ef_min = EfficientFrontier(mu, S)
    min_vol = ef_min.min_volatility()
    cleaned_min_vol = ef_min.clean_weights()
    ret_min_vol, vol_min_vol, sharpe_min = ef_min.portfolio_performance()
    
    # Plot Efficient Frontier
    fig, ax = plt.subplots(figsize=(10,6))
    # Simulate random portfolios for visualization
    n_portfolios = 10000
    results = np.zeros((3, n_portfolios))
    
    for i in range(n_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        portfolio_return = np.dot(weights, mu)
        portfolio_vol = np.sqrt(weights.T @ S.values @ weights)
        results[0,i] = portfolio_return
        results[1,i] = portfolio_vol
        results[2,i] = portfolio_return / portfolio_vol  # Sharpe ratio
    
    ax.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', alpha=0.5)
    ax.scatter(vol_max_sharpe, ret_max_sharpe, marker='*', color='r', s=200, label='Max Sharpe')
    ax.scatter(vol_min_vol, ret_min_vol, marker='*', color='b', s=200, label='Min Volatility')
    ax.set_xlabel('Volatility (Risk)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier')
    ax.legend()
    plt.show()
    
    summary = {
        'max_sharpe': {
            'weights': cleaned_max_sharpe,
            'expected_return': ret_max_sharpe,
            'volatility': vol_max_sharpe,
            'sharpe_ratio': sharpe_max
        },
        'min_volatility': {
            'weights': cleaned_min_vol,
            'expected_return': ret_min_vol,
            'volatility': vol_min_vol,
            'sharpe_ratio': sharpe_min
        }
    }
    
    return summary
