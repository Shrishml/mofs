# Services

This directory contains AI-powered services for financial research and analysis.

## Current Implementation

```
services/
├── deep_reseach_workflow.py    # LangGraph workflow for company research on recent trends ✅
```

## Deep Research Workflow

The `deep_reseach_workflow.py` module implements a sophisticated LangGraph-based workflow for conducting deep research on companies and investment opportunities. This research workflow:

1. Takes a research query as input
2. Generates focused search terms using LLM
3. Retrieves relevant information from the web using Tavily Search API
4. Extracts detailed content from the most promising sources using Tavily Extract API
5. Analyzes the content to identify companies and investment opportunities

### Features

- Multi-step LangGraph workflow for orchestrated research
- Smart search term generation for comprehensive coverage
- Web content search and extraction via Tavily APIs
- Company identification with investment opportunity analysis
- Detailed progress reporting

### Usage Example

```python
from services.deep_reseach_workflow import perform_deep_research

# Run deep research on a specific query
result = perform_deep_research("Which companies could benefit from trade tariff imposition in the US?")

# Access the research results
if result and result.get("companies"):
    print("\n=== IDENTIFIED INVESTMENT OPPORTUNITIES ===\n")
    
    for i, company in enumerate(result["companies"], 1):
        ticker = f" ({company.get('ticker_symbol')})" if company.get('ticker_symbol') else ""
        print(f"{i}. {company.get('company_name')}{ticker}")
        print(f"   Opportunity: {company.get('opportunity')}")
```

## Future Plans (ph)

### Research Enhancement
- Implement checkpoint IDs to restart conversations
- Save intermediate research reports as knowledge base
- Add critique node for feedback and research improvement

### Additional Agents (ph)
- Political research agent for policy impact analysis
- Technology research agent for emerging tech trends
- Social trends agent for demographic and cultural shifts

### Integration (ph)
- Connect with internal data sources
- Support for alternative data analysis
- Enhance with domain-specific knowledge
