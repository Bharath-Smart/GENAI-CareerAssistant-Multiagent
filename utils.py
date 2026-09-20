import os
import requests
from firecrawl import Firecrawl

from dotenv import load_dotenv

load_dotenv()

SERPER_SEARCH_URL = "https://google.serper.dev/search"


class SerperClient:
    """
    A client for performing Google searches using the Serper API.

    Calls the Serper REST API (https://serper.dev) directly, since Serper has
    no official Python SDK; this replaces the deprecated
    `langchain_community.utilities.GoogleSerperAPIWrapper`.
    """

    def __init__(self, serper_api_key: str = os.environ.get("SERPER_API_KEY")) -> None:
        self.serper_api_key = serper_api_key

    def search(
        self,
        query,
        num_results: int = 5,
    ):
        """
        Perform a Google search for the given query and return the search results.

        Args:
            query (str): The search query.
            num_results (int, optional): The number of search results to retrieve. Defaults to GOOGLE_SEARCH_DEFAULT_RESULT_COUNT.

        Returns:
            dict: The search results as a dictionary.

        """
        response = requests.post(
            SERPER_SEARCH_URL,
            headers={
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": num_results},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        # this is to make the response compatible with the response from the google search client
        items = data.pop("organic", [])
        data["items"] = items
        return data


class FireCrawlClient:

    def __init__(
        self, firecrawl_api_key: str = os.environ.get("FIRECRAWL_API_KEY")
    ) -> None:
        self.firecrawl_api_key = firecrawl_api_key

    def scrape(self, url):
        # Current Firecrawl Python SDK (v2): Firecrawl(...).scrape(url, formats=[...])
        # replaces the deprecated langchain_community FireCrawlLoader.
        doc = Firecrawl(api_key=self.firecrawl_api_key).scrape(url, formats=["markdown"])
        page_content = doc.markdown or ""

        # limit to 10,000 characters
        return page_content[:10000]
