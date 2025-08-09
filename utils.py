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
        """Perform a Google search and return results in a standardized format."""
        wrapper = GoogleSerperAPIWrapper(serper_api_key=self.serper_api_key, k=num_results)
        response = wrapper.results(query=query)

        # Normalize response format
        response["items"] = response.pop("organic", [])
        return response

# -------------------- FIRECRAWL CLIENT --------------------
class FireCrawlClient:
    """Client for scraping webpage content using FireCrawl API."""

    def __init__(self, firecrawl_api_key: str = os.environ.get("FIRECRAWL_API_KEY")) -> None:
        self.firecrawl_api_key = firecrawl_api_key

    def scrape(self, url: str) -> str:
        """Scrape webpage content for a given URL (up to 10,000 characters)."""
        docs = FireCrawlLoader(
            api_key=self.firecrawl_api_key,
            url=url,
            mode="scrape"
        ).lazy_load()

        page_content = "".join(doc.page_content for doc in docs)
        return page_content[:10000]
