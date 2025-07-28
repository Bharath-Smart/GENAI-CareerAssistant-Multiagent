"""
tools.py

Defines LangChain-compatible tools for job search, resume extraction,
cover letter generation, web scraping, and Google (Serper) search.
"""

# -------------------- IMPORTS --------------------
# Standard Library
import os
import asyncio
from typing import List, Union

# Environment & Configuration
from dotenv import load_dotenv

# LangChain / LangGraph
from langchain.pydantic_v1 import Field
from langchain.tools import BaseTool, tool, StructuredTool

# Internal Modules
from data_loader import load_resume, write_cover_letter_to_doc
from schemas import JobSearchInput
from search import get_job_ids, fetch_all_jobs
from utils import FireCrawlClient, SerperClient

# -------------------- ENVIRONMENT SETUP --------------------
load_dotenv()

# -------------------- JOB SEARCH TOOL --------------------
async def linkedin_job_search(
    keywords: Union[str, List[str]],
    location_name: str = None,
    job_type: Union[str, List[str]] = None,
    limit: int = 5,
    employment_type: Union[str, List[str]] = None,
    listed_at=None,
    experience: Union[str, List[str]] = None,
    distance=None,
) -> dict:
    """
    Search LinkedIn for job postings based on specified criteria.

    Returns:
        dict: Detailed job listings based on filters.
    """
    job_ids = get_job_ids(
        keywords=keywords,
        location_name=location_name,
        employment_type=employment_type,
        limit=limit,
        job_type=job_type,
        listed_at=listed_at,
        experience=experience,
        distance=distance,
    )
    return await fetch_all_jobs(job_ids)


def get_job_search_tool():
    """
    Create a structured async tool for LangChain JobPipeline.

    Returns:
        StructuredTool: LangChain-compatible async tool wrapper.
    """
    return StructuredTool.from_function(
        func=linkedin_job_search,
        name="JobSearchTool",
        description="Search LinkedIn for job postings based on specified criteria. Returns detailed job listings.",
        args_schema=JobSearchInput,
        coroutine=linkedin_job_search,
    )

# -------------------- RESUME EXTRACTION TOOL --------------------
class ResumeExtractorTool(BaseTool):
    """
    Extracts the resume content from the uploaded PDF file.

    Returns:
        dict: Parsed resume content (skills, experience, etc.).
    """
    name: str = "ResumeExtractor"
    description: str = "Extract the content of uploaded resume from a PDF file."

    def extract_resume(self) -> str:
        return load_resume("temp/resume.pdf")

    def _run(self) -> dict:
        return self.extract_resume()

# -------------------- COVER LETTER TOOLS --------------------
@tool
def generate_letter_for_specific_job(resume_details: str, job_details: str) -> dict:
    """
    Generate a tailored cover letter using resume and job details.

    Returns:
        dict: Merged inputs for the cover letter generator.
    """
    return {"job_details": job_details, "resume_details": resume_details}


@tool
def save_cover_letter_for_specific_job(cover_letter_content: str, company_name: str) -> str:
    """
    Save the generated cover letter to a DOCX file and return download path.

    Args:
        cover_letter_content (str): The cover letter content.
        company_name (str): Used in filename.

    Returns:
        str: Local download path to the generated file.
    """
    filename = f"temp/{company_name}_cover_letter.docx"
    file = write_cover_letter_to_doc(cover_letter_content, filename)
    return f"Here is the download link: {os.path.abspath(file)}"

# -------------------- WEB SEARCH TOOLS --------------------
@tool("google_search")
def get_google_search_results(query: str = Field(..., description="Search query for web")) -> str:
    """
    Perform a Google-like search using Serper API.

    Args:
        query (str): Search keywords.

    Returns:
        str: Compiled text of top results (title, link, snippet).
    """
    response = SerperClient().search(query)
    items = response.get("items", [])
    results = []

    for result in items:
        try:
            results.append(
                "\n".join([
                    f"Title: {result['title']}",
                    f"Link: {result['link']}",
                    f"Snippet: {result['snippet']}",
                    "---"
                ])
            )
        except KeyError:
            continue

    return "\n".join(results)


@tool("scrape_website")
def scrape_website(url: str = Field(..., description="Url to be scraped")) -> str:
    """
    Scrape the main text content of a website using FireCrawl API.

    Args:
        url (str): Target webpage URL.

    Returns:
        str: Scraped webpage text (up to 10k characters).
    """
    try:
        return FireCrawlClient().scrape(url)
    except Exception:
        return f"Failed to scrape {url}"
