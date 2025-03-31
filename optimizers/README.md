# Portfolio Optimization

This directory contains implementations of portfolio optimization algorithms for asset allocation and risk management.

## Current Implementation

```
optimizers/
├── monte_carlo_optimizer.py    # Monte Carlo simulation-based optimization ✅
├── other_optimizers(ph)        # Additional optimization methods (placeholder)
```

## Monte Carlo Optimizer

The `monte_carlo_optimizer.py` module implements a comprehensive Monte Carlo simulation approach to portfolio optimization. This optimizer:

- Generates thousands of random portfolio allocations
- Evaluates each portfolio's risk and return characteristics
- Identifies optimal portfolios based on Sharpe or Sortino ratios
- Visualizes the efficient frontier for better decision making

### Features

- Historical or custom return/covariance estimates
- Adjustable risk-free rate
- Sharpe and Sortino ratio optimization
- Efficient frontier visualization
- Comprehensive result metrics

### Usage Example

```python
from optimizers.monte_carlo_optimizer import MonteCarloOptimizer

# Initialize optimizer with stock tickers
optimizer = MonteCarloOptimizer(
    tickers=["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
    start_date="2022-01-01",
    end_date="2023-01-01",
    risk_free_rate=4.5,        # 4.5% risk-free rate
    ratio="sharpe"            # Use Sharpe ratio for optimization
)

# Run Monte Carlo simulation with 10,000 portfolio combinations
result = optimizer.run_monte_carlo(n_simulations=10000)

# View optimization results
print(f"Expected Annual Return: {result.expected_return:.2%}")
print(f"Annual Volatility: {result.volatility:.2%}")
print(f"Sharpe Ratio: {result.ratio_value:.2f}")

# Print optimal weights
print("\nOptimal Portfolio Weights:")
for ticker, weight in result.weights.items():
    print(f"  {ticker}: {weight:.2%}")

# Plot the efficient frontier
optimizer.plot()
```

## Planned Optimizers (ph)

- Mean-Variance Optimization (Markowitz)
- Hierarchical Risk Parity
- Black-Litterman Model
- Minimum Variance Portfolio
- Maximum Diversification Portfolio

## Estimators (ph)

Future implementations will include improved estimators for returns and covariance matrices:
- GARCH models
- Shrinkage estimators
- Factor models (CAPM, Fama-French, etc.)
- Machine learning-based forecasting

