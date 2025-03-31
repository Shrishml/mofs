"""
News and deep research data module using LangGraph, LangChain, and Tavily.
Performs deep research on topics and extracts investment insights.
"""

from typing import Dict, List, Any, Tuple, Optional, Annotated, TypedDict, Sequence
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_tavily import TavilySearch, TavilyExtract

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.graph import StateGraph, END
import logging


# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define state types and models
class ResearchState(TypedDict):
    """State for the research process."""
    query: str
    search_terms: List[str]
    urls: List[Dict[str, Any]]
    research_docs: Dict[str, Dict[str, Any]]
    iteration: int
    companies: List[Dict[str, Any]]

class SearchTerm(BaseModel):
    """A search term with explanation"""
    term: str = Field(description="The search term")
    explanation: str = Field(description="Explanation why this term is relevant")

class SearchTerms(BaseModel):
    """List of search terms"""
    terms: List[SearchTerm] = Field(description="List of search terms")


class CompanyInsight(BaseModel):
    """Information about a company that could benefit from tariffs"""
    company_name: str = Field(description="Name of the company")
    ticker_symbol: Optional[str] = Field(description="Stock ticker symbol if publicly traded")
    reason: str = Field(description="Reason for selecting this company")
    confidence: int = Field(description="Confidence score (1-10) of this assessment")

class CompanyInsightList(BaseModel):
    """List of company insights"""
    companies: List[CompanyInsight] = Field(description="List of company insights")

# Initialize necessary clients
tavily_search_tool = TavilySearch(
    max_results=5,
    topic="finance",
    api_key=os.getenv("TAVILY_API_KEY"),
)

tavily_extract_tool = TavilyExtract(
    extract_depth="advanced",
    include_images=False,
    api_key=os.getenv("TAVILY_API_KEY"),
)

# Define the agent LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("API_VERSION"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    temperature=0.1
)

from langchain_core.output_parsers import PydanticOutputParser

# Set up a parser


# Create a chain for generating search terms
def generate_search_terms(query: str) -> List[Dict[str, str]]:
    """Generate search terms using LLM"""
    print(f"Generating search terms for query: {query}")

    parser = PydanticOutputParser(pydantic_object=SearchTerms)
    search_terms_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial researcher specializing deep research, you are known for finding diamond from coals.
        Generate 5-10 specific search terms to answer query asked by user. Create diverse terms that will capture different 
        aspects, industries, and perspectives related to query. 
        
        For each search term:
        1. Make it specific and targeted
        2. Provide a brief explanation of why this term is relevant
        3. Ensure it will help identify specific investment opportunities, not just general information
        
        Format your response as a JSON with a list of terms, each with a 'term' and 'explanation' field.
        Wrap the output in `json` tags {format_instructions}.
        """),
        ("user", "Generate search terms for researching: {query}")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    search_terms_chain = search_terms_prompt | llm.with_structured_output(SearchTerms)
    result = search_terms_chain.invoke({"query": query})
    print(f"Generated search terms: {result.terms}")
    return result


# Helper function to filter unique URLs
def filter_urls(urls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out  URLs with low score"""
    filtered_urls = []
    for url in urls:
        if url["score"] > 0.5:
            filtered_urls.append(url)
    print(f"Filtered URLs: {len(filtered_urls)} out of {len(urls)}")
    return filtered_urls


def wide_research(state: ResearchState) -> ResearchState:
    """
    First node: Generate search terms and collect related URLs.
    
    Uses Tavily to gather a wide range of relevant URLs based on multiple search terms
    derived from the original query.
    """
    query = state["query"]
    print(f"\n--- STARTING WIDE RESEARCH FOR: {query} ---")
    all_urls = []
    
    # Generate search terms and use Tavily Search
    search_terms= generate_search_terms(query).terms

    logger.info(f"Search terms generated: {search_terms}")
    print(f"Search terms generated: {len(search_terms)}")
    
    for term_info in search_terms:
        term = term_info.term
        print(f"Searching for term: {term}")
        # Search Tavily for this term
        try:
            result = tavily_search_tool.invoke({"query": term})
            filtered_urls = filter_urls(result["results"])
            print(f"Found {len(filtered_urls)} relevant URLs for term: {term}")
            for url_info in filtered_urls:
                url_info["search_term"] = term_info.term
                url_info["search_explanation"] = term_info.explanation
                all_urls.append(url_info)

            
        except Exception as e:
            print(f"Error searching for term '{term}': {e}")

    # need to remove duplicate
    print(f"Total URLs collected: {len(all_urls)}")
    logger.info(f"Filtered URLs: {all_urls}")
    return { "search_terms": search_terms, "urls": all_urls}


def extract_content(state: ResearchState) -> ResearchState:
    """
    Second node: Non-LLM node that extracts content from each URL.
    
    Goes through each URL and extracts webpage information, populating
    the research_docs dictionary.
    """
    urls = state["urls"]
    research_docs = state.get("research_docs", {})
    
    print(f"\n--- EXTRACTING CONTENT FROM {len(urls)} URLS ---")
    
    # Extract content using Tavily Extract
    for i, url_info in enumerate(urls):
        url = url_info.get("url", "")
        title = url_info.get("title", "")
        content = url_info.get("content", "")
        search_term = url_info.get("search_term", "")
        search_explanation = url_info.get("search_explanation", "")
        
        print(f"[{i+1}/{len(urls)}] Extracting content from: {title} ({url})")
        try:
            extract_result = tavily_extract_tool.invoke({"urls": [url]})
            print(f"Successfully extracted content from {url}")
            logger.info(f"Extract result: {extract_result}")
            raw_content = extract_result["results"][0].get("raw_content", "")
            content_length = len(raw_content)
            print(f"Content length: {content_length} characters")
            research_docs[url] = {"title": title, "content": content, "raw_content": raw_content, "search_term": search_term, "search_explanation": search_explanation}
        except Exception as e:
            print(f"Error extracting content from '{url}': {e}")
    
    print(f"Successfully extracted content from {len(research_docs)} URLs")
    # Update state
    return {"research_docs": research_docs}

def extract_companies(state: ResearchState) -> ResearchState:
    """
    Final node: Extract company information from research docs.
    
    Process research_docs in batches of 10 URLs and extract companies with
    their ticker symbols and investment opportunities.
    """
    query = state["query"]
    research_docs = state["research_docs"]
    all_companies = []
    
    print(f"\n--- EXTRACTING COMPANIES FROM {len(research_docs)} DOCUMENTS ---")
    
    parser = PydanticOutputParser(pydantic_object=CompanyInsightList)
    # Convert research_docs to a list for batch processing
    docs_list = []
    for url, doc in research_docs.items():
        # Skip if there's no content
        if not doc.get("content") and not doc.get("raw_content"):
            continue
            
        docs_list.append({
            "url": url,
            "title": doc.get("title", "Unknown"),
            "content": doc.get("content", ""),
            "raw_content": doc.get("raw_content", ""),
            "search_term": doc.get("search_term", ""),
            "search_explanation": doc.get("search_explanation", "")
        })
    
    print(f"Processing {len(docs_list)} documents with valid content")
    
    # Process in batches of 10
    batch_size = 5  # Reduced batch size for faster processing
    for i in range(0, len(docs_list), batch_size):
        batch = docs_list[i:min(i+batch_size, len(docs_list))]
        print(f"Processing batch {i//batch_size + 1}/{(len(docs_list) + batch_size - 1)//batch_size}")
        
        # Prepare batch content
        batch_content = ""
        for doc in batch:
            content = doc.get("raw_content") if doc.get("raw_content") else doc.get("content")
            truncated_content = content[:10000] + "..." if content and len(content) > 10000 else content
            batch_content += f"--- {doc['title']} ---\nURL: {doc['url']}\nSearch Term: {doc['search_term']}\n\n{truncated_content}\n\n"
        
        print(f"Batch content length: {len(batch_content)} characters")
        
        # Create prompt for company extraction
        company_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial analyst specializing in identifying investment opportunities.
            Your task is to extract companies mentioned in the provided content that could be potential investment opportunities.
            
            For each company you identify:
            1. Extract the company name
            2. Extract the ticker symbol if available
            3. Provide a brief narrative explaining the investment opportunity or potential benefit
            
            Focus only on finding specific companies, not general industries or sectors.
            Format each company as a JSON object with the following fields:
            - company_name: The name of the company
            - ticker_symbol: The stock ticker symbol (if mentioned, otherwise leave as null)
            - opportunity: A brief explanation of why this company represents an investment opportunity
            
            Return an array of these company objects. 
            Wrap the output in `json` tags {format_instructions}.
            """),
            ("user", "Research query: {query}\n\nAnalyze the following content and extract companies:\n{content}")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        # Extract companies from this batch
        try:
            print("Extracting companies from batch...")
            company_extraction_chain = company_extraction_prompt | llm.with_structured_output(CompanyInsightList)
            companies_batch = company_extraction_chain.invoke({
                "query": query,
                "content": batch_content
            })
            companies_batch_list = companies_batch.companies
            print(f"Found {len(companies_batch_list)} companies in this batch")
            logger.info(f"Companies batch: {companies_batch_list}")
            # Add to all companies list
            if isinstance(companies_batch_list, list):
                all_companies.extend(companies_batch_list)
            else:
                print(f"Unexpected format from company extraction: {type(companies_batch_list)}")
        except Exception as e:
            print(f"Error extracting companies from batch: {e}")
   
    return {"companies": all_companies}

# Initialize the graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("wide_research", wide_research)
workflow.add_node("extract_content", extract_content)
workflow.add_node("extract_companies", extract_companies)

# Add edges
workflow.add_edge("wide_research", "extract_content")
workflow.add_edge("extract_content", "extract_companies")
workflow.add_edge("extract_companies", END)

# Set the entry point
workflow.set_entry_point("wide_research")

research_workflow = workflow.compile()

# Function to run the workflow
def perform_deep_research(query: str) -> Dict[str, Any]:
    """
    Perform deep research on a query and extract company investment opportunities.
    
    Args:
        query (str): The research query
        
    Returns:
        Dict[str, Any]: The research results including extracted companies
    """
    print(f"\n=== STARTING DEEP RESEARCH: {query} ===\n")
    
    # Initialize the state
    initial_state = {
        "query": query,
        "search_terms": [],
        "urls": [],
        "research_docs": {},
        "iteration": 0,
        "companies": []
    }
    
    # Run the workflow
    final_state = None
    for event in research_workflow.stream(initial_state):
        state = event.state
        node_name = event.node_name
        
        # Print progress
        if node_name == "wide_research":
            print(f"Conducting wide research on query: {query}")
        elif node_name == "extract_content":
            print(f"Extracting content from {len(state.get('urls', []))} URLs")
        elif node_name == "extract_companies":
            print(f"Extracting companies from {len(state.get('research_docs', {}))} documents")
        
        final_state = state
    
    print("\n=== RESEARCH COMPLETE ===\n")
    return final_state

# Example usage
if __name__ == "__main__":
    query = "Which companies could benefit from trade tariff imposition in the US?"
    result = perform_deep_research(query)
    
    if result and result.get("companies"):
        print("\n\n=== IDENTIFIED INVESTMENT OPPORTUNITIES ===\n")
        for i, company in enumerate(result["companies"], 1):
            ticker = f" ({company.get('ticker_symbol')})" if company.get('ticker_symbol') else ""
            print(f"{i}. {company.get('company_name')}{ticker}")
            print(f"   Opportunity: {company.get('opportunity')}")
            print()
    else:
        print("Research failed or no companies identified.")