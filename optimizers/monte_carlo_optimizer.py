import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Literal
from dataclasses import dataclass
import seaborn as sns
from datetime import datetime, timedelta

@dataclass
class OptimizationResult:
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    ratio_value: float
    efficient_frontier: pd.DataFrame
    all_weights: np.ndarray  # Added to store all weights for analysis

class MonteCarloOptimizer:
    def __init__(
        self, 
        tickers: List[str], 
        start_date: str = None, 
        end_date: str = None,
        estimate_method: Literal["historical", "custom"] = "historical",
        expected_returns: Optional[pd.Series] = None,
        covariance_matrix: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 4.5,
        ratio: str = "sharpe"
    ):
        """
        Initialize the Monte Carlo optimizer.
        
        Args:
            tickers (List[str]): List of stock tickers
            start_date (str, optional): Start date for historical data in 'YYYY-MM-DD' format
            end_date (str, optional): End date for historical data in 'YYYY-MM-DD' format
            estimate_method (str): Method to estimate returns and volatility ("historical" or "custom")
            expected_returns (pd.Series, optional): Custom expected returns for each asset
            covariance_matrix (pd.DataFrame, optional): Custom covariance matrix
            risk_free_rate (float): Risk-free rate for ratio calculation
            ratio (str): Optimization ratio to use ("sharpe", "sortino")
        """
        self.tickers = tickers
        self.start_date = start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.estimate_method = estimate_method
        self.risk_free_rate = risk_free_rate
        self.ratio = ratio.lower()
        
        # Validate inputs for custom estimates
        if estimate_method == "custom":
            if expected_returns is None or covariance_matrix is None:
                raise ValueError("Custom estimates require both expected_returns and covariance_matrix")
            if not all(ticker in expected_returns.index for ticker in tickers):
                raise ValueError("Expected returns must contain all tickers")
            if not all(ticker in covariance_matrix.index for ticker in tickers):
                raise ValueError("Covariance matrix must contain all tickers")
            
            self.mean_returns = expected_returns[tickers]
            self.cov_matrix = covariance_matrix.loc[tickers, tickers]
        else:
            self.returns = None
            self.mean_returns = None
            self.cov_matrix = None
            
        self.optimal_weights = None
        self.efficient_frontier = None
        
    def fetch_data(self) -> None:
        """Fetch historical data for the given tickers."""
        # Download historical data
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        
        # Calculate daily returns
        self.returns = data['Close'].pct_change().dropna()
        
        # Calculate mean returns and covariance matrix
        self.mean_returns = self.returns.mean() * 252  # Annualized returns
        self.cov_matrix = self.returns.cov() * 252    # Annualized covariance
        
    def calculate_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio metrics for given weights.
        
        Args:
            weights (np.ndarray): Portfolio weights
            
        Returns:
            Tuple[float, float, float]: (return, volatility, ratio)
        """
        # Calculate portfolio return
        portfolio_return = np.sum(self.mean_returns * weights)
        
        # Calculate portfolio volatility using covariance matrix
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
        if self.ratio == "sharpe":
            ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        elif self.ratio == "sortino":
            if self.estimate_method == "historical":
                # Calculate downside deviation using historical returns
                returns = np.sum(self.returns * weights, axis=1)
                downside_returns = returns[returns < self.risk_free_rate/252]  # Daily risk-free rate
                downside_volatility = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
            else:
                # Approximate downside deviation as 70% of total volatility
                downside_volatility = portfolio_volatility * 0.7
            ratio = (portfolio_return - self.risk_free_rate) / downside_volatility
        else:
            raise ValueError(f"Unsupported ratio: {self.ratio}")
            
        return portfolio_return, portfolio_volatility, ratio
        
    def run_monte_carlo(self, n_simulations: int = 1000) -> OptimizationResult:
        """
        Run Monte Carlo simulation to find optimal portfolio weights.
        
        Args:
            n_simulations (int): Number of Monte Carlo simulations
            
        Returns:
            OptimizationResult: Object containing optimization results
        """
        if self.estimate_method == "historical" and self.returns is None:
            self.fetch_data()
            
        # Initialize arrays to store results
        portfolio_returns = np.zeros(n_simulations)
        portfolio_volatilities = np.zeros(n_simulations)
        portfolio_ratios = np.zeros(n_simulations)
        all_weights = np.zeros((n_simulations, len(self.tickers)))
        
        # Run Monte Carlo simulations
        for i in range(n_simulations):
            # Generate random weights using uniform distribution
            weights = np.random.uniform(0, 1, len(self.tickers))
            weights = weights / weights.sum()  # Normalize to sum to 1
            all_weights[i, :] = weights
            
            # Calculate portfolio metrics
            portfolio_returns[i], portfolio_volatilities[i], portfolio_ratios[i] = self.calculate_metrics(weights)
        
        # Find optimal portfolio (maximum ratio)
        optimal_idx = np.argmax(portfolio_ratios)
        
        # Store optimal weights
        self.optimal_weights = dict(zip(self.tickers, all_weights[optimal_idx]))
        
        # Create efficient frontier DataFrame
        self.efficient_frontier = pd.DataFrame({
            'Return': portfolio_returns,
            'Volatility': portfolio_volatilities,
            f'{self.ratio.capitalize()} Ratio': portfolio_ratios
        })
        
        return OptimizationResult(
            weights=self.optimal_weights,
            expected_return=portfolio_returns[optimal_idx],
            volatility=portfolio_volatilities[optimal_idx],
            ratio_value=portfolio_ratios[optimal_idx],
            efficient_frontier=self.efficient_frontier,
            all_weights=all_weights
        )
    def plot(self, save_path: Optional[str] = None, live: bool = False) -> None:
        """
        Plot the efficient frontier, optimal portfolio, and Capital Market Line.
        Displays all values in percentages.
        
        Args:
            save_path (str, optional): Path to save the plot
            live (bool): Whether to show the plot live
        """
        if self.efficient_frontier is None:
            raise ValueError("Run Monte Carlo simulation first using run_monte_carlo()")
            
        plt.figure(figsize=(12, 8))
        
        # Convert to percentages
        returns_pct = self.efficient_frontier['Return'] * 100
        volatility_pct = self.efficient_frontier['Volatility'] * 100
        risk_free_pct = self.risk_free_rate * 100
        
        # Plot all portfolios
        scatter = plt.scatter(
            volatility_pct,
            returns_pct,
            c=self.efficient_frontier[f'{self.ratio.capitalize()} Ratio'],
            cmap='viridis',
            alpha=0.5
        )
        
        # Plot optimal portfolio
        optimal_portfolio = self.efficient_frontier.loc[
            self.efficient_frontier[f'{self.ratio.capitalize()} Ratio'].idxmax()
        ]
        optimal_return_pct = optimal_portfolio['Return'] * 100
        optimal_vol_pct = optimal_portfolio['Volatility'] * 100
        
        plt.scatter(
            optimal_vol_pct,
            optimal_return_pct,
            color='red',
            s=200,
            marker='*',
            label='Optimal Portfolio'
        )
        
        # Plot Capital Market Line
        max_vol = volatility_pct.max()
        # Calculate slope (Sharpe ratio of optimal portfolio)
        slope = (optimal_return_pct - risk_free_pct) / optimal_vol_pct
        # Create CML line
        x_cml = np.array([0, max_vol])
        y_cml = risk_free_pct + slope * x_cml
        plt.plot(x_cml, y_cml, 'r--', label='Capital Market Line')
        
        # Plot risk-free rate point
        plt.scatter(0, risk_free_pct, color='green', marker='o', s=100, label='Risk-free Rate')
        
        plt.colorbar(scatter, label=f'{self.ratio.capitalize()} Ratio')
        plt.xlabel('Volatility (%)')
        plt.ylabel('Expected Return (%)')
        plt.title(f'Efficient Frontier with Optimal Portfolio ({self.ratio.capitalize()} Ratio)')
        plt.grid(True, alpha=0.2)
        plt.legend()
        
        # Add annotations
        plt.annotate(
            f'Optimal Portfolio\nReturn: {optimal_return_pct:.1f}%\nVol: {optimal_vol_pct:.1f}%\nSharpe: {optimal_portfolio[f"{self.ratio.capitalize()} Ratio"]:.2f}',
            xy=(optimal_vol_pct, optimal_return_pct),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8)
        )
        
        plt.annotate(
            f'Risk-free Rate: {risk_free_pct:.1f}%',
            xy=(0, risk_free_pct),
            xytext=(10, -10),
            textcoords='offset points',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8)
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if live:
            plt.show()
        else:
            plt.close()

# Example usage:
if __name__ == "__main__":
    # Example tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Example with historical data
    optimizer = MonteCarloOptimizer(
        tickers=tickers,
        start_date='2022-01-01',
        risk_free_rate=0.02,
        ratio="sharpe"
    )
    
    # Run optimization
    result = optimizer.run_monte_carlo(n_simulations=1000)
    
    # Print results
    print("\nOptimal Portfolio Weights:")
    for ticker, weight in result.weights.items():
        print(f"{ticker}: {weight:.4f}")
    
    print("\nPortfolio Metrics:")
    print(f"Expected Return: {result.expected_return:.4f}")
    print(f"Volatility: {result.volatility:.4f}")
    print(f"Sharpe Ratio: {result.ratio_value:.4f}")
    
    # Plot results
    optimizer.plot(save_path='efficient_frontier.png', live=True)
    
    # Example with custom estimates
    # Create sample expected returns and covariance matrix
    expected_returns = pd.Series({
        'AAPL': 0.15,
        'MSFT': 0.12,
        'GOOGL': 0.18
    })
    
    covariance_matrix = pd.DataFrame({
        'AAPL':  [0.04, 0.02, 0.015],
        'MSFT':  [0.02, 0.03, 0.01],
        'GOOGL': [0.015, 0.01, 0.05]
    }, index=['AAPL', 'MSFT', 'GOOGL'])
    
    optimizer_custom = MonteCarloOptimizer(
        tickers=tickers,
        estimate_method="custom",
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.02,
        ratio="sharpe"
    )
    
    # Run optimization with custom estimates
    result_custom = optimizer_custom.run_monte_carlo(n_simulations=1000)
    
    # Print results
    print("\nOptimal Portfolio Weights (Custom Estimates):")
    for ticker, weight in result_custom.weights.items():
        print(f"{ticker}: {weight:.4f}")
    
    print("\nPortfolio Metrics (Custom Estimates):")
    print(f"Expected Return: {result_custom.expected_return:.4f}")
    print(f"Volatility: {result_custom.volatility:.4f}")
    print(f"Sharpe Ratio: {result_custom.ratio_value:.4f}")
    
    # Plot results
    optimizer_custom.plot(save_path='efficient_frontier_custom.png', live=True)