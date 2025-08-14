"""
Utility functions for the portfolio forecasting project.

This module contains helper functions for:
- Path management
- Logging configuration
- Performance metrics calculation
- Data validation
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        config: Configuration dictionary containing logging settings
        
    Returns:
        Configured logger instance
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', 'logs/portfolio_forecasting.log')
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple returns from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Returns series
    """
    return prices.pct_change().dropna()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Log returns series
    """
    return np.log(prices / prices.shift(1)).dropna()


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Returns series
        window: Rolling window size
        
    Returns:
        Volatility series
    """
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)
    return excess_returns / volatility if volatility != 0 else 0


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        prices: Price series
        
    Returns:
        Maximum drawdown as a percentage
    """
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min()


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).
    
    Args:
        returns: Returns series
        
    Returns:
        Calmar ratio
    """
    annual_return = returns.mean() * 252
    prices = (1 + returns).cumprod()
    max_dd = abs(calculate_max_drawdown(prices))
    return annual_return / max_dd if max_dd != 0 else 0


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns.mean() * 252 - risk_free_rate
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    return excess_returns / downside_deviation if downside_deviation != 0 else 0


def validate_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that data contains required columns and has no missing values.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if data is valid, False otherwise
    """
    # Check if all required columns are present
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        return False
    
    # Check for missing values
    if data[required_columns].isnull().any().any():
        logging.warning("Data contains missing values")
        return False
    
    return True


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        filepath: Path to save file
    """
    ensure_dir(os.path.dirname(filepath))
    
    if filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    elif filepath.endswith('.json'):
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Results dictionary
    """
    if filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    elif filepath.endswith('.json'):
        import json
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def format_currency(value: float, currency: str = "USD") -> str:
    """
    Format value as currency.
    
    Args:
        value: Numeric value
        currency: Currency symbol
        
    Returns:
        Formatted currency string
    """
    return f"${value:,.2f}" if currency == "USD" else f"{value:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Numeric value (as decimal)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"
