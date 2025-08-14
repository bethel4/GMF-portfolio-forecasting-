"""
Exploratory Data Analysis module for portfolio forecasting.

This module handles:
- Price and returns visualization
- Volatility analysis
- Correlation analysis
- Statistical summaries
- Interactive plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .utils import load_config, ensure_dir, calculate_returns, calculate_volatility


class EDAAnalyzer:
    """
    Class for performing exploratory data analysis on financial data.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize EDAAnalyzer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create output directory
        self.output_dir = "reports/figures"
        ensure_dir(self.output_dir)
    
    def plot_price_series(
        self, 
        data_dict: Dict[str, pd.DataFrame], 
        save_fig: bool = True
    ) -> None:
        """
        Plot price series for multiple tickers.
        
        Args:
            data_dict: Dictionary with ticker data
            save_fig: Whether to save the figure
        """
        fig, axes = plt.subplots(len(data_dict), 1, figsize=(15, 4 * len(data_dict)))
        if len(data_dict) == 1:
            axes = [axes]
        
        for i, (ticker, data) in enumerate(data_dict.items()):
            # Ensure date is index
            if 'date' in data.columns:
                data = data.set_index('date')
            
            axes[i].plot(data.index, data['close'], label=f'{ticker} Close Price', linewidth=1.5)
            axes[i].set_title(f'{ticker} Price Series', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Price ($)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_dir}/price_series.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Price series plot saved to {self.output_dir}/price_series.png")
        
        plt.show()
    
    def plot_returns_analysis(
        self, 
        data_dict: Dict[str, pd.DataFrame], 
        save_fig: bool = True
    ) -> None:
        """
        Plot returns analysis including distribution and time series.
        
        Args:
            data_dict: Dictionary with ticker data
            save_fig: Whether to save the figure
        """
        n_tickers = len(data_dict)
        fig, axes = plt.subplots(n_tickers, 3, figsize=(18, 5 * n_tickers))
        if n_tickers == 1:
            axes = axes.reshape(1, -1)
        
        for i, (ticker, data) in enumerate(data_dict.items()):
            # Ensure date is index
            if 'date' in data.columns:
                data = data.set_index('date')
            
            # Calculate returns if not present
            if 'returns' not in data.columns:
                data['returns'] = calculate_returns(data['close'])
            
            returns = data['returns'].dropna()
            
            # Time series plot
            axes[i, 0].plot(returns.index, returns, alpha=0.7, linewidth=0.8)
            axes[i, 0].set_title(f'{ticker} Returns Time Series')
            axes[i, 0].set_xlabel('Date')
            axes[i, 0].set_ylabel('Returns')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Histogram
            axes[i, 1].hist(returns, bins=50, alpha=0.7, density=True, edgecolor='black')
            axes[i, 1].set_title(f'{ticker} Returns Distribution')
            axes[i, 1].set_xlabel('Returns')
            axes[i, 1].set_ylabel('Density')
            
            # Add normal distribution overlay
            mu, sigma = stats.norm.fit(returns)
            x = np.linspace(returns.min(), returns.max(), 100)
            axes[i, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')
            axes[i, 1].legend()
            
            # Q-Q plot
            stats.probplot(returns, dist="norm", plot=axes[i, 2])
            axes[i, 2].set_title(f'{ticker} Q-Q Plot')
            axes[i, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_dir}/returns_analysis.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Returns analysis plot saved to {self.output_dir}/returns_analysis.png")
        
        plt.show()
    
    def plot_volatility_analysis(
        self, 
        data_dict: Dict[str, pd.DataFrame], 
        save_fig: bool = True
    ) -> None:
        """
        Plot volatility analysis.
        
        Args:
            data_dict: Dictionary with ticker data
            save_fig: Whether to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Collect volatility data
        vol_data = {}
        for ticker, data in data_dict.items():
            if 'date' in data.columns:
                data = data.set_index('date')
            
            if 'returns' not in data.columns:
                data['returns'] = calculate_returns(data['close'])
            
            if 'volatility' not in data.columns:
                data['volatility'] = calculate_volatility(data['returns'])
            
            vol_data[ticker] = data['volatility'].dropna()
        
        # Time series of volatility
        for ticker, vol in vol_data.items():
            axes[0, 0].plot(vol.index, vol, label=ticker, alpha=0.8)
        axes[0, 0].set_title('Volatility Time Series')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Annualized Volatility')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volatility distribution
        for ticker, vol in vol_data.items():
            axes[0, 1].hist(vol, bins=30, alpha=0.6, label=ticker, density=True)
        axes[0, 1].set_title('Volatility Distribution')
        axes[0, 1].set_xlabel('Volatility')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        
        # Rolling correlation of volatilities (if multiple tickers)
        if len(vol_data) > 1:
            tickers = list(vol_data.keys())
            vol_df = pd.DataFrame(vol_data)
            rolling_corr = vol_df[tickers[0]].rolling(window=60).corr(vol_df[tickers[1]])
            axes[1, 0].plot(rolling_corr.index, rolling_corr)
            axes[1, 0].set_title(f'Rolling Volatility Correlation ({tickers[0]} vs {tickers[1]})')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Correlation')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Volatility clustering visualization
        ticker = list(vol_data.keys())[0]  # Use first ticker
        vol_series = vol_data[ticker]
        high_vol_threshold = vol_series.quantile(0.75)
        high_vol_periods = vol_series > high_vol_threshold
        
        axes[1, 1].plot(vol_series.index, vol_series, alpha=0.6, color='blue')
        axes[1, 1].fill_between(vol_series.index, 0, vol_series, 
                               where=high_vol_periods, alpha=0.3, color='red', 
                               label='High Volatility Periods')
        axes[1, 1].set_title(f'{ticker} Volatility Clustering')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_dir}/volatility_analysis.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Volatility analysis plot saved to {self.output_dir}/volatility_analysis.png")
        
        plt.show()
    
    def plot_correlation_matrix(
        self, 
        data_dict: Dict[str, pd.DataFrame], 
        save_fig: bool = True
    ) -> None:
        """
        Plot correlation matrix of returns.
        
        Args:
            data_dict: Dictionary with ticker data
            save_fig: Whether to save the figure
        """
        # Prepare returns data
        returns_data = {}
        for ticker, data in data_dict.items():
            if 'date' in data.columns:
                data = data.set_index('date')
            
            if 'returns' not in data.columns:
                data['returns'] = calculate_returns(data['close'])
            
            returns_data[ticker] = data['returns']
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=ax)
        
        ax.set_title('Returns Correlation Matrix', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Correlation matrix plot saved to {self.output_dir}/correlation_matrix.png")
        
        plt.show()
        
        return corr_matrix
    
    def plot_rolling_statistics(
        self, 
        data_dict: Dict[str, pd.DataFrame], 
        window: int = 60,
        save_fig: bool = True
    ) -> None:
        """
        Plot rolling statistics (mean, std, correlation).
        
        Args:
            data_dict: Dictionary with ticker data
            window: Rolling window size
            save_fig: Whether to save the figure
        """
        n_tickers = len(data_dict)
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Prepare data
        returns_data = {}
        for ticker, data in data_dict.items():
            if 'date' in data.columns:
                data = data.set_index('date')
            
            if 'returns' not in data.columns:
                data['returns'] = calculate_returns(data['close'])
            
            returns_data[ticker] = data['returns']
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        # Rolling mean
        rolling_mean = returns_df.rolling(window=window).mean()
        for ticker in returns_df.columns:
            axes[0].plot(rolling_mean.index, rolling_mean[ticker], label=ticker)
        axes[0].set_title(f'Rolling Mean Returns ({window}-day window)')
        axes[0].set_ylabel('Mean Returns')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Rolling standard deviation
        rolling_std = returns_df.rolling(window=window).std()
        for ticker in returns_df.columns:
            axes[1].plot(rolling_std.index, rolling_std[ticker], label=ticker)
        axes[1].set_title(f'Rolling Standard Deviation ({window}-day window)')
        axes[1].set_ylabel('Standard Deviation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Rolling correlation (if multiple tickers)
        if len(returns_df.columns) > 1:
            tickers = list(returns_df.columns)
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    rolling_corr = returns_df[tickers[i]].rolling(window=window).corr(returns_df[tickers[j]])
                    axes[2].plot(rolling_corr.index, rolling_corr, label=f'{tickers[i]} vs {tickers[j]}')
            
            axes[2].set_title(f'Rolling Correlation ({window}-day window)')
            axes[2].set_ylabel('Correlation')
            axes[2].set_xlabel('Date')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'Multiple tickers required for correlation analysis', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Rolling Correlation (Not Available)')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_dir}/rolling_statistics.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Rolling statistics plot saved to {self.output_dir}/rolling_statistics.png")
        
        plt.show()
    
    def create_interactive_dashboard(
        self, 
        data_dict: Dict[str, pd.DataFrame]
    ) -> go.Figure:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            data_dict: Dictionary with ticker data
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Price Series', 'Returns Distribution', 
                          'Volatility', 'Correlation Heatmap',
                          'Volume', 'Technical Indicators'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (ticker, data) in enumerate(data_dict.items()):
            color = colors[i % len(colors)]
            
            # Ensure date is available
            if 'date' in data.columns:
                dates = pd.to_datetime(data['date'])
            else:
                dates = data.index
            
            # Price series
            fig.add_trace(
                go.Scatter(x=dates, y=data['close'], name=f'{ticker} Price',
                          line=dict(color=color), showlegend=True),
                row=1, col=1
            )
            
            # Calculate returns if not present
            if 'returns' not in data.columns:
                returns = calculate_returns(data['close'])
            else:
                returns = data['returns']
            
            # Returns histogram
            fig.add_trace(
                go.Histogram(x=returns.dropna(), name=f'{ticker} Returns',
                           marker_color=color, opacity=0.7, showlegend=False),
                row=1, col=2
            )
            
            # Volatility
            if 'volatility' not in data.columns:
                volatility = calculate_volatility(returns)
            else:
                volatility = data['volatility']
            
            fig.add_trace(
                go.Scatter(x=dates, y=volatility, name=f'{ticker} Volatility',
                          line=dict(color=color), showlegend=False),
                row=2, col=1
            )
            
            # Volume
            if 'volume' in data.columns:
                fig.add_trace(
                    go.Scatter(x=dates, y=data['volume'], name=f'{ticker} Volume',
                              line=dict(color=color), showlegend=False),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Portfolio Analysis Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Volatility", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        return fig
    
    def generate_summary_statistics(
        self, 
        data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Generate summary statistics for all tickers.
        
        Args:
            data_dict: Dictionary with ticker data
            
        Returns:
            DataFrame with summary statistics
        """
        summary_stats = []
        
        for ticker, data in data_dict.items():
            if 'date' in data.columns:
                data = data.set_index('date')
            
            # Calculate returns if not present
            if 'returns' not in data.columns:
                data['returns'] = calculate_returns(data['close'])
            
            returns = data['returns'].dropna()
            prices = data['close']
            
            # Calculate statistics
            stats_dict = {
                'Ticker': ticker,
                'Start Date': data.index.min().strftime('%Y-%m-%d'),
                'End Date': data.index.max().strftime('%Y-%m-%d'),
                'Observations': len(data),
                'Start Price': prices.iloc[0],
                'End Price': prices.iloc[-1],
                'Total Return': (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
                'Annualized Return': returns.mean() * 252 * 100,
                'Volatility': returns.std() * np.sqrt(252) * 100,
                'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                'Max Drawdown': self._calculate_max_drawdown(prices) * 100,
                'Skewness': returns.skew(),
                'Kurtosis': returns.kurtosis(),
                'Min Return': returns.min() * 100,
                'Max Return': returns.max() * 100,
                'VaR (5%)': returns.quantile(0.05) * 100,
                'CVaR (5%)': returns[returns <= returns.quantile(0.05)].mean() * 100
            }
            
            summary_stats.append(stats_dict)
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Round numerical columns
        numeric_columns = summary_df.select_dtypes(include=[np.number]).columns
        summary_df[numeric_columns] = summary_df[numeric_columns].round(4)
        
        return summary_df
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            prices: Price series
            
        Returns:
            Maximum drawdown as decimal
        """
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def plot_technical_indicators(
        self, 
        data: pd.DataFrame, 
        ticker: str,
        save_fig: bool = True
    ) -> None:
        """
        Plot technical indicators for a single ticker.
        
        Args:
            data: Data with technical indicators
            ticker: Ticker symbol
            save_fig: Whether to save the figure
        """
        if 'date' in data.columns:
            data = data.set_index('date')
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # Price and moving averages
        axes[0].plot(data.index, data['close'], label='Close Price', linewidth=1.5)
        
        # Plot moving averages if available
        ma_columns = [col for col in data.columns if col.startswith('ma_') and not col.endswith('_ratio')]
        for ma_col in ma_columns:
            if ma_col in data.columns:
                axes[0].plot(data.index, data[ma_col], label=ma_col.upper(), alpha=0.7)
        
        # Bollinger Bands if available
        if all(col in data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            axes[0].fill_between(data.index, data['bb_upper'], data['bb_lower'], 
                               alpha=0.2, label='Bollinger Bands')
            axes[0].plot(data.index, data['bb_middle'], '--', alpha=0.7, label='BB Middle')
        
        axes[0].set_title(f'{ticker} Price and Moving Averages')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        if 'rsi' in data.columns:
            axes[1].plot(data.index, data['rsi'], color='purple', linewidth=1.5)
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
            axes[1].fill_between(data.index, 30, 70, alpha=0.1, color='gray')
            axes[1].set_title(f'{ticker} RSI')
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim(0, 100)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # MACD
        if all(col in data.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
            axes[2].plot(data.index, data['macd'], label='MACD', linewidth=1.5)
            axes[2].plot(data.index, data['macd_signal'], label='Signal', linewidth=1.5)
            axes[2].bar(data.index, data['macd_histogram'], label='Histogram', alpha=0.6)
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[2].set_title(f'{ticker} MACD')
            axes[2].set_ylabel('MACD')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Volume
        if 'volume' in data.columns:
            axes[3].bar(data.index, data['volume'], alpha=0.6, color='orange')
            if 'volume_ma_20' in data.columns:
                axes[3].plot(data.index, data['volume_ma_20'], color='red', linewidth=2, label='20-day MA')
                axes[3].legend()
            axes[3].set_title(f'{ticker} Volume')
            axes[3].set_ylabel('Volume')
            axes[3].set_xlabel('Date')
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{self.output_dir}/technical_indicators_{ticker.lower()}.png', 
                       dpi=300, bbox_inches='tight')
            self.logger.info(f"Technical indicators plot saved for {ticker}")
        
        plt.show()


def main():
    """
    Main function to demonstrate EDA functionality.
    """
    from .data_fetch import DataFetcher
    from .preprocess import DataPreprocessor
    
    # Initialize components
    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()
    analyzer = EDAAnalyzer()
    
    # Get and preprocess data
    print("Fetching data...")
    data_dict = fetcher.get_market_data()
    
    processed_data = {}
    for ticker, data in data_dict.items():
        print(f"Processing {ticker}...")
        clean_data = preprocessor.clean_data(data)
        processed_data[ticker] = preprocessor.calculate_technical_indicators(clean_data)
    
    # Perform EDA
    print("Generating EDA plots...")
    
    # Basic plots
    analyzer.plot_price_series(processed_data)
    analyzer.plot_returns_analysis(processed_data)
    analyzer.plot_volatility_analysis(processed_data)
    analyzer.plot_correlation_matrix(processed_data)
    analyzer.plot_rolling_statistics(processed_data)
    
    # Technical indicators for first ticker
    first_ticker = list(processed_data.keys())[0]
    analyzer.plot_technical_indicators(processed_data[first_ticker], first_ticker)
    
    # Generate summary statistics
    summary_stats = analyzer.generate_summary_statistics(processed_data)
    print("\nSummary Statistics:")
    print(summary_stats.to_string(index=False))
    
    # Create interactive dashboard
    print("Creating interactive dashboard...")
    interactive_fig = analyzer.create_interactive_dashboard(processed_data)
    interactive_fig.show()


if __name__ == "__main__":
    main()
