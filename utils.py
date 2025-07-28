"""
utils.py

Provides utility classes for web search and document scraping:
- `SerperClient`: Uses Serper API to perform Google searches.
- `FireCrawlClient`: Uses FireCrawl API to scrape webpage content.

These utilities are used by LangGraph agents for research and document retrieval.
"""

# -------------------- IMPORTS --------------------
# Standard Library
import os

# Environment & Configuration
from dotenv import load_dotenv

# LangChain / LangGraph
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import FireCrawlLoader

# -------------------- ENVIRONMENT SETUP --------------------
load_dotenv()

# -------------------- SERPER CLIENT --------------------
class SerperClient:
    """Client for performing Google searches using the Serper API."""

    def __init__(self, serper_api_key: str = os.environ.get("SERPER_API_KEY")) -> None:
        self.serper_api_key = serper_api_key

    def search(self, query: str, num_results: int = 5) -> dict:
        """Perform a Google search and return results in a standard format."""
        response = GoogleSerperAPIWrapper(k=num_results).results(query=query)

        # Normalize response format
        response["items"] = response.pop("organic", [])
        return response


# -------------------- FIRECRAWL CLIENT --------------------
class FireCrawlClient:
    """Client for scraping webpage content using FireCrawl API."""

    def __init__(self, firecrawl_api_key: str = os.environ.get("FIRECRAWL_API_KEY")) -> None:
        self.firecrawl_api_key = firecrawl_api_key

    def scrape(self, url: str) -> str:
        """Scrape webpage content for a given URL (max 10,000 chars)."""
        docs = FireCrawlLoader(api_key=self.firecrawl_api_key, url=url, mode="scrape").lazy_load()

        page_content = ""
        for doc in docs:
            page_content += doc.page_content

        # limit to 10,000 characters
        return page_content[:10000]
