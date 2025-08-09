"""
tools.py

Defines LangChain-compatible tools for job search, resume extraction,
cover letter generation, web scraping, and Google (Serper) search.
"""

# -------------------- IMPORTS --------------------
# Standard Library
import os
import asyncio
from typing import Literal, List, Union, Optional

# Environment & Configuration
from dotenv import load_dotenv

# LangChain / LangGraph
from pydantic import Field
from langchain_core.tools import tool, StructuredTool

# Internal Modules
from data_loader import load_resume, write_cover_letter_to_doc
from schemas import JobSearchInput, GoogleSearchInput, ScrapeWebsiteInput
from search import get_job_ids, fetch_all_jobs
from utils import FireCrawlClient, SerperClient

# -------------------- ENVIRONMENT SETUP --------------------
load_dotenv()

# -------------------- RESUME EXTRACTION TOOL --------------------
@tool(name_or_callable = "extract_resume_tool")
def extract_resume() -> str:
    """
    Extracts text content from the uploaded resume (PDF) stored at 'temp/resume.pdf'.
    This tool assumes the resume has already been uploaded and saved.
    
    Returns:
        str: Resume text (combined content from all pages) used for LLM analysis.       
    """
        
    return load_resume("temp/resume.pdf")

# -------------------- JOB SEARCH TOOL --------------------
async def linkedin_job_search(
    keywords: Union[str, List[str]],
    location_name: Optional[str] = None,
    employment_type: Optional[
        List[
            Literal[
                "full-time", 
                "contract", 
                "part-time", 
                "temporary",
                "internship", 
                "volunteer", 
                "other"                
            ]
        ]
    ] = None,
    limit: Optional[int] = 5,
    job_type: Optional[
        List[
            Literal[
                "onsite", 
                "remote", 
                "hybrid"
            ]
        ]
    ] = None,
    experience: Optional[
        List[
            Literal[
                "internship", 
                "entry-level", 
                "associate",
                "mid-senior-level", 
                "director", 
                "executive"
            ]
        ]
    ] = None,
    listed_at: Optional[Union[int, str]] = 86400,
    distance: Optional[Union[int, str]] = 25
    
) -> List[dict]:
    """
    Search LinkedIn for job postings based on specified criteria.

    Returns:
        List[dict]: Detailed job listings based on filters.
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

job_search_tool = StructuredTool.from_function(
        name="JobSearchTool",
        description="Search LinkedIn for job postings based on specified criteria. Returns detailed job listings.",
        args_schema=JobSearchInput,
        coroutine=linkedin_job_search,
    )

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
@tool(name_or_callable = "google_search_tool", args_schema = GoogleSearchInput)
def get_google_search_results(query: str) -> str:
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

@tool(name_or_callable = "scrape_website_tool", args_schema = ScrapeWebsiteInput)
def scrape_website(url: str) -> str:
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
