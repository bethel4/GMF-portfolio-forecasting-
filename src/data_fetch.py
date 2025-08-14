"""
Data fetching module for portfolio forecasting.

This module handles:
- Downloading financial data from various sources (primarily yfinance)
- Data validation and cleaning
- Saving data to local storage
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
from .utils import load_config, ensure_dir, validate_data


class DataFetcher:
    """
    Class for fetching financial data from various sources.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataFetcher with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.data_config = self.config['data']
        self.logger = logging.getLogger(__name__)
        
        # Ensure data directories exist
        ensure_dir(self.data_config['raw_data_path'])
        ensure_dir(self.data_config['processed_data_path'])
    
    def fetch_ticker_data(
        self, 
        ticker: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Data frequency
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Use config dates if not provided
            start_date = start_date or self.data_config['date_range']['start_date']
            end_date = end_date or self.data_config['date_range']['end_date']
            
            self.logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
            
            # Download data using yfinance
            stock = yf.Ticker(ticker)
            data = stock.history(
                start=start_date,
                end=end_date,
                period=period,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Clean column names
            data.columns = data.columns.str.lower().str.replace(' ', '_')
            
            # Add ticker column
            data['ticker'] = ticker
            
            # Reset index to make date a column
            data = data.reset_index()
            data['date'] = pd.to_datetime(data['date'])
            
            self.logger.info(f"Successfully fetched {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise
    
    def fetch_multiple_tickers(
        self, 
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with ticker as key and DataFrame as value
        """
        tickers = tickers or self.data_config['tickers']
        data_dict = {}
        
        for ticker in tickers:
            try:
                data_dict[ticker] = self.fetch_ticker_data(
                    ticker, start_date, end_date
                )
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
                continue
        
        return data_dict
    
    def fetch_and_save_data(
        self, 
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_format: str = "csv"
    ) -> Dict[str, str]:
        """
        Fetch data and save to local storage.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save_format: File format ('csv' or 'parquet')
            
        Returns:
            Dictionary with ticker as key and file path as value
        """
        data_dict = self.fetch_multiple_tickers(tickers, start_date, end_date)
        file_paths = {}
        
        for ticker, data in data_dict.items():
            # Create filename
            filename = f"{ticker.lower()}_{start_date or self.data_config['date_range']['start_date']}_to_{end_date or self.data_config['date_range']['end_date']}.{save_format}"
            filepath = os.path.join(self.data_config['raw_data_path'], filename)
            
            # Save data
            if save_format == "csv":
                data.to_csv(filepath, index=False)
            elif save_format == "parquet":
                data.to_parquet(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {save_format}")
            
            file_paths[ticker] = filepath
            self.logger.info(f"Saved {ticker} data to {filepath}")
        
        return file_paths
    
    def load_saved_data(
        self, 
        ticker: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load previously saved data from local storage.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for filename
            end_date: End date for filename
            
        Returns:
            DataFrame with loaded data
        """
        start_date = start_date or self.data_config['date_range']['start_date']
        end_date = end_date or self.data_config['date_range']['end_date']
        
        # Try different file formats
        for ext in ['csv', 'parquet']:
            filename = f"{ticker.lower()}_{start_date}_to_{end_date}.{ext}"
            filepath = os.path.join(self.data_config['raw_data_path'], filename)
            
            if os.path.exists(filepath):
                self.logger.info(f"Loading {ticker} data from {filepath}")
                
                if ext == 'csv':
                    data = pd.read_csv(filepath)
                else:
                    data = pd.read_parquet(filepath)
                
                data['date'] = pd.to_datetime(data['date'])
                return data
        
        raise FileNotFoundError(f"No saved data found for {ticker}")
    
    def get_market_data(
        self, 
        tickers: Optional[List[str]] = None,
        force_download: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Get market data, either from cache or by downloading.
        
        Args:
            tickers: List of ticker symbols
            force_download: Whether to force re-download
            
        Returns:
            Dictionary with ticker data
        """
        tickers = tickers or self.data_config['tickers']
        data_dict = {}
        
        for ticker in tickers:
            try:
                if not force_download:
                    # Try to load from cache first
                    try:
                        data_dict[ticker] = self.load_saved_data(ticker)
                        continue
                    except FileNotFoundError:
                        pass
                
                # Download and save data
                data = self.fetch_ticker_data(ticker)
                
                # Save to cache
                filename = f"{ticker.lower()}_{self.data_config['date_range']['start_date']}_to_{self.data_config['date_range']['end_date']}.csv"
                filepath = os.path.join(self.data_config['raw_data_path'], filename)
                data.to_csv(filepath, index=False)
                
                data_dict[ticker] = data
                
            except Exception as e:
                self.logger.error(f"Error getting data for {ticker}: {str(e)}")
                continue
        
        return data_dict
    
    def validate_data_quality(self, data: pd.DataFrame, ticker: str) -> bool:
        """
        Validate data quality for a ticker.
        
        Args:
            data: DataFrame to validate
            ticker: Ticker symbol
            
        Returns:
            True if data passes quality checks
        """
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        if not validate_data(data, required_columns):
            return False
        
        # Check for reasonable price values
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (data[col] <= 0).any():
                self.logger.warning(f"{ticker}: Found non-positive prices in {col}")
                return False
        
        # Check high >= low
        if (data['high'] < data['low']).any():
            self.logger.warning(f"{ticker}: Found high < low")
            return False
        
        # Check for extreme price movements (>50% in one day)
        returns = data['close'].pct_change().abs()
        if (returns > 0.5).any():
            self.logger.warning(f"{ticker}: Found extreme price movements")
        
        # Check volume
        if (data['volume'] < 0).any():
            self.logger.warning(f"{ticker}: Found negative volume")
            return False
        
        self.logger.info(f"{ticker}: Data quality validation passed")
        return True
    
    def get_fundamental_data(self, ticker: str) -> Dict:
        """
        Get fundamental data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with fundamental data
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key fundamental metrics
            fundamentals = {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
            }
            
            return fundamentals
            
        except Exception as e:
            self.logger.error(f"Error fetching fundamentals for {ticker}: {str(e)}")
            return {}


def main():
    """
    Main function to demonstrate data fetching functionality.
    """
    # Initialize data fetcher
    fetcher = DataFetcher()
    
    # Fetch and save data for configured tickers
    print("Fetching market data...")
    data_dict = fetcher.get_market_data()
    
    # Print summary
    for ticker, data in data_dict.items():
        print(f"\n{ticker}:")
        print(f"  Records: {len(data)}")
        print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"  Latest close: ${data['close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
