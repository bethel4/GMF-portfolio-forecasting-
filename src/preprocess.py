"""
Data preprocessing module for portfolio forecasting.

This module handles:
- Data cleaning and feature engineering
- Stationarity tests
- Technical indicators calculation
- Data transformation for modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

from .utils import load_config, calculate_returns, calculate_log_returns, calculate_volatility


class DataPreprocessor:
    """
    Class for preprocessing financial data.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataPreprocessor with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.preprocess_config = self.config['preprocessing']
        self.logger = logging.getLogger(__name__)
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw financial data.
        
        Args:
            data: Raw financial data
            
        Returns:
            Cleaned data
        """
        df = data.copy()
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Sort by date
        df = df.sort_index()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Forward fill missing values (common in financial data)
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill')
        
        # Drop remaining NaN values
        df = df.dropna()
        
        # Remove outliers using IQR method
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        self.logger.info(f"Data cleaned: {len(df)} records remaining")
        return df
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            data: OHLCV data
            
        Returns:
            Data with technical indicators
        """
        df = data.copy()
        
        # Returns
        df['returns'] = calculate_returns(df['close'])
        df['log_returns'] = calculate_log_returns(df['close'])
        
        # Volatility
        vol_window = self.preprocess_config.get('volatility_window', 20)
        df['volatility'] = calculate_volatility(df['returns'], vol_window)
        
        # Moving averages
        ma_windows = self.preprocess_config.get('ma_windows', [5, 10, 20, 50])
        for window in ma_windows:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
        
        # RSI (Relative Strength Index)
        rsi_period = self.preprocess_config.get('rsi_period', 14)
        df['rsi'] = self._calculate_rsi(df['close'], rsi_period)
        
        # Bollinger Bands
        bb_window = self.preprocess_config.get('bollinger_window', 20)
        bb_std = self.preprocess_config.get('bollinger_std', 2)
        df = self._calculate_bollinger_bands(df, bb_window, bb_std)
        
        # MACD
        df = self._calculate_macd(df)
        
        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            df['price_volume'] = df['close'] * df['volume']
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Intraday returns
        df['intraday_return'] = (df['close'] - df['open']) / df['open']
        
        self.logger.info("Technical indicators calculated")
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(
        self, 
        data: pd.DataFrame, 
        window: int = 20, 
        num_std: float = 2
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: OHLCV data
            window: Moving average window
            num_std: Number of standard deviations
            
        Returns:
            Data with Bollinger Bands
        """
        df = data.copy()
        
        # Calculate moving average and standard deviation
        ma = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        
        # Calculate bands
        df['bb_upper'] = ma + (std * num_std)
        df['bb_lower'] = ma - (std * num_std)
        df['bb_middle'] = ma
        
        # Calculate position within bands
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def _calculate_macd(
        self, 
        data: pd.DataFrame, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: OHLCV data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Data with MACD indicators
        """
        df = data.copy()
        
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=fast_period).mean()
        ema_slow = df['close'].ewm(span=slow_period).mean()
        
        # Calculate MACD line
        df['macd'] = ema_fast - ema_slow
        
        # Calculate signal line
        df['macd_signal'] = df['macd'].ewm(span=signal_period).mean()
        
        # Calculate histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def test_stationarity(self, series: pd.Series, alpha: float = 0.05) -> Dict[str, any]:
        """
        Test for stationarity using ADF and KPSS tests.
        
        Args:
            series: Time series to test
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        series_clean = series.dropna()
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series_clean, autolag='AIC')
        adf_stationary = adf_result[1] < alpha
        
        # KPSS test
        kpss_result = kpss(series_clean, regression='c', nlags="auto")
        kpss_stationary = kpss_result[1] > alpha
        
        # Ljung-Box test for autocorrelation
        lb_result = acorr_ljungbox(series_clean, lags=10, return_df=True)
        lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
        no_autocorr = lb_pvalue < alpha
        
        results = {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_stationary': adf_stationary,
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1],
            'kpss_stationary': kpss_stationary,
            'ljung_box_pvalue': lb_pvalue,
            'has_autocorrelation': no_autocorr,
            'is_stationary': adf_stationary and kpss_stationary
        }
        
        return results
    
    def make_stationary(self, series: pd.Series, method: str = 'diff') -> Tuple[pd.Series, int]:
        """
        Make time series stationary.
        
        Args:
            series: Time series
            method: Method to use ('diff', 'log_diff', 'seasonal_diff')
            
        Returns:
            Tuple of (stationary series, number of differences)
        """
        if method == 'diff':
            # Simple differencing
            diff_series = series.diff().dropna()
            n_diffs = 1
            
            # Test if one difference is enough
            stationarity = self.test_stationarity(diff_series)
            if not stationarity['is_stationary']:
                # Try second difference
                diff_series = diff_series.diff().dropna()
                n_diffs = 2
        
        elif method == 'log_diff':
            # Log transformation then differencing
            log_series = np.log(series.replace(0, np.nan)).dropna()
            diff_series = log_series.diff().dropna()
            n_diffs = 1
        
        elif method == 'seasonal_diff':
            # Seasonal differencing (252 trading days)
            seasonal_period = 252
            diff_series = series.diff(seasonal_period).dropna()
            n_diffs = 1
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return diff_series, n_diffs
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for machine learning models.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Data with engineered features
        """
        df = data.copy()
        
        # Lag features
        lag_periods = [1, 2, 3, 5, 10]
        for lag in lag_periods:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else np.nan
        
        # Rolling statistics
        windows = [5, 10, 20]
        for window in windows:
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
        
        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year
            
            # Market regime indicators
            df['is_monday'] = (df['day_of_week'] == 0).astype(int)
            df['is_friday'] = (df['day_of_week'] == 4).astype(int)
            df['is_month_end'] = df.index.is_month_end.astype(int)
            df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Interaction features
        if 'volume' in df.columns:
            df['price_volume_trend'] = df['close'] * df['volume']
            df['volume_price_ratio'] = df['volume'] / df['close']
        
        # Volatility regime
        df['high_volatility'] = (df['volatility'] > df['volatility'].quantile(0.75)).astype(int)
        df['low_volatility'] = (df['volatility'] < df['volatility'].quantile(0.25)).astype(int)
        
        self.logger.info("Feature engineering completed")
        return df
    
    def prepare_lstm_data(
        self, 
        data: pd.DataFrame, 
        target_col: str = 'close',
        sequence_length: int = 60,
        features: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model.
        
        Args:
            data: Preprocessed data
            target_col: Target column name
            sequence_length: Length of input sequences
            features: List of feature columns to use
            
        Returns:
            Tuple of (X, y) arrays
        """
        df = data.copy()
        
        # Select features
        if features is None:
            # Use price and technical indicators
            feature_cols = [
                'close', 'volume', 'returns', 'volatility',
                'rsi', 'macd', 'bb_position'
            ]
            # Add available MA columns
            ma_cols = [col for col in df.columns if col.startswith('ma_') and col.endswith('_ratio')]
            feature_cols.extend(ma_cols)
            
            # Filter to existing columns
            features = [col for col in feature_cols if col in df.columns]
        
        # Normalize features
        feature_data = df[features].values
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(feature_data)):
            X.append(feature_data[i-sequence_length:i])
            y.append(df[target_col].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        self.logger.info(f"LSTM data prepared: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def split_data(
        self, 
        data: pd.DataFrame, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Data to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        self.logger.info(f"Data split: Train {len(train_data)}, Val {len(val_data)}, Test {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def normalize_data(
        self, 
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        method: str = 'minmax'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Normalize data using training set statistics.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            method: Normalization method ('minmax' or 'zscore')
            
        Returns:
            Tuple of normalized data and scaler parameters
        """
        numeric_columns = train_data.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            # Min-Max scaling
            min_vals = train_data[numeric_columns].min()
            max_vals = train_data[numeric_columns].max()
            
            scaler_params = {'min': min_vals, 'max': max_vals, 'method': 'minmax'}
            
            train_norm = train_data.copy()
            val_norm = val_data.copy()
            test_norm = test_data.copy()
            
            train_norm[numeric_columns] = (train_data[numeric_columns] - min_vals) / (max_vals - min_vals)
            val_norm[numeric_columns] = (val_data[numeric_columns] - min_vals) / (max_vals - min_vals)
            test_norm[numeric_columns] = (test_data[numeric_columns] - min_vals) / (max_vals - min_vals)
        
        elif method == 'zscore':
            # Z-score normalization
            mean_vals = train_data[numeric_columns].mean()
            std_vals = train_data[numeric_columns].std()
            
            scaler_params = {'mean': mean_vals, 'std': std_vals, 'method': 'zscore'}
            
            train_norm = train_data.copy()
            val_norm = val_data.copy()
            test_norm = test_data.copy()
            
            train_norm[numeric_columns] = (train_data[numeric_columns] - mean_vals) / std_vals
            val_norm[numeric_columns] = (val_data[numeric_columns] - mean_vals) / std_vals
            test_norm[numeric_columns] = (test_data[numeric_columns] - mean_vals) / std_vals
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return train_norm, val_norm, test_norm, scaler_params


def main():
    """
    Main function to demonstrate preprocessing functionality.
    """
    from .data_fetch import DataFetcher
    
    # Initialize components
    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()
    
    # Get sample data
    data_dict = fetcher.get_market_data(['TSLA'])
    tsla_data = data_dict['TSLA']
    
    # Preprocess data
    print("Cleaning data...")
    clean_data = preprocessor.clean_data(tsla_data)
    
    print("Calculating technical indicators...")
    processed_data = preprocessor.calculate_technical_indicators(clean_data)
    
    print("Testing stationarity...")
    stationarity = preprocessor.test_stationarity(processed_data['close'])
    print(f"Price series is stationary: {stationarity['is_stationary']}")
    
    stationarity_returns = preprocessor.test_stationarity(processed_data['returns'])
    print(f"Returns series is stationary: {stationarity_returns['is_stationary']}")
    
    print("Creating features...")
    featured_data = preprocessor.create_features(processed_data)
    
    print(f"Final dataset shape: {featured_data.shape}")
    print(f"Features: {list(featured_data.columns)}")


if __name__ == "__main__":
    main()
