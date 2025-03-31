# MOFS (Modern Optimization for Financial Services)

A comprehensive financial research and portfolio optimization framework focused on discovering investment opportunities through deep research and portfolio optimization.

## Current Implementation Status

This project is under active development. Below is the current state of implementation:

```
mofs/
├── services/                           # AI services for research and analysis
│   └── deep_reseach_workflow.py        # LangGraph workflow for company research
├── optimizers/                         # Portfolio optimization algorithms
│   └── monte_carlo_optimizer.py        # Monte Carlo optimization implementation
├── data_source(ph)/                    # Data sources (placeholder)
├── estimators(ph)/                     # Statistical estimators (placeholder)
├── copilot(ph)/                        # Interactive copilot features (placeholder)
├── experiment.ipynb                    # Experiment notebook
├── demo_tariff.ipynb                   # Demo notebook for trade tariff impact analysis
├── demo_low_fertility_rate.ipynb       # Demo notebook for declining fertility rate impact analysis
├── requirements.txt                    # Project dependencies
├── .env                                # Environment variables (API keys)
└── .gitignore                          # Git ignore file
```

## Core Features

### ✅ Deep Research with LangGraph
- Utilizes LangGraph to orchestrate a multi-step research workflow
- Automatically generates relevant search terms for any research query
- Uses Tavily search and extract APIs to gather and process information
- Identifies companies and investment opportunities from research data

### ✅ Portfolio Optimization
- Monte Carlo simulation for portfolio optimization
- Risk/return analysis with covariance matrix
- Sharpe and Sortino ratio optimization
- Efficient frontier visualization

### 🔄 In Progress
- Inter-theme optimization
- Risk analysis improvements
- More market estimators

### 📋 Planned Features (Not Yet Implemented)
- Political, Technology, and Social research agents (ph)
- Price movement clustering (ph)
- Major player tracking (ph)
- Game theoretical analysis on major players(ph)

## Data Sources
- ✅ Tavily API for web search and content extraction(integrated into services)
- ✅ Yahoo Finance for historical price data(integrated into optimizers)
- 📋 News and media analysis (planned)
- 📋 Social media and Reddit (planned)
- 📋 Earnings calls analysis (planned)

## Required API Keys

This project requires the following API keys which should be stored in a `.env` file:

```
# Azure OpenAI Configuration
AZURE_ENDPOINT=your-azure-endpoint
API_KEY=your-api-key
DEPLOYMENT_NAME=your-deployment-name
API_VERSION=your-api-version

# Tavily API Key
TAVILY_API_KEY=your-tavily-api-key
```

**IMPORTANT**: Never commit your `.env` file to version control!

## Setup

### Using Conda (Recommended)

1. Create and activate the conda environment:
```bash
# Create environment with Python 3.11
conda create -n mofs python=3.11

# Activate the environment
conda activate mofs

# Install requirements
pip install -r requirements.txt
```

2. Set up API keys:
- Create a `.env` file with your API keys as shown above
- Keep these keys secure and never share them

### Using Virtual Environment (Alternative)

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Deep Research

Our deep research implementation uses a LangGraph workflow to:

1. Generate focused search terms from your query
2. Search the web for relevant information using Tavily
3. Extract detailed content from the most valuable sources
4. Identify companies that represent potential investment opportunities

Example usage:
```python
from services.deep_reseach_workflow import perform_deep_research

# Run deep research
result = perform_deep_research("Which companies could benefit from trade tariff imposition in the US?")

# View identified companies
for company in result.get("companies", []):
    print(f"{company.get('company_name')} ({company.get('ticker_symbol', 'No ticker')})")
    print(f"  Opportunity: {company.get('opportunity')}")
```

### Portfolio Optimization

The Monte Carlo optimizer allows you to find optimal portfolio allocations:

```python
from optimizers.monte_carlo_optimizer import MonteCarloOptimizer

# Initialize optimizer with stock tickers
optimizer = MonteCarloOptimizer(
    tickers=["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
    start_date="2022-01-01",
    end_date="2023-01-01"
)

# Run optimization
result = optimizer.run_monte_carlo(n_simulations=10000)

# View results
print(f"Expected Return: {result.expected_return:.2%}")
print(f"Volatility: {result.volatility:.2%}")
print(f"Sharpe Ratio: {result.ratio_value:.2f}")
print("\nOptimal Portfolio Weights:")
for ticker, weight in result.weights.items():
    print(f"  {ticker}: {weight:.2%}")

# Plot efficient frontier
optimizer.plot()
```

## Demo Notebooks

We have two demo notebooks available to showcase the capabilities of the MOFS framework:

- [Trade Tariff Impact Analysis](demo_tariff.ipynb): Demonstrates how to research companies that could benefit from trade tariffs and optimize a portfolio based on findings.
- [Declining Fertility Rate Impact Analysis](demo_low_fertility_rate.ipynb): Explores investment opportunities related to declining fertility rates.

These notebooks provide practical examples of how to use the deep research and portfolio optimization features of the framework.

## Development

### Future Modules
We plan to expand the framework with these additional components:
- Theme discovery agents for political, technology, and social research
- Advanced clustering techniques for price movement analysis
- Custom covariance matrix estimators for better risk assessment

