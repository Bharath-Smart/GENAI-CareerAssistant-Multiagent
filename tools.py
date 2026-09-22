# define tools
import os
import asyncio
import re
from dotenv import load_dotenv
from langchain_core.tools import tool
from data_loader import load_resume, write_cover_letter_to_doc
from schemas import JobSearchInput
from search import get_job_ids, fetch_all_jobs
from utils import FirecrawlClient, SerperClient

load_dotenv()


# Job search tool.
@tool("JobSearchTool", args_schema=JobSearchInput)
def linkedin_job_search(
    keywords: str,
    location_name: str = "India",
    job_type: list[str] | None = None,
    limit: int = 10,
    employment_type: list[str] | None = None,
    listed_at: int | str = 86400,
    experience: list[str] | None = None,
    distance: int | str = 100,
) -> list[dict]:
    """
    Search LinkedIn for job postings and return normalized job records.
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
    return asyncio.run(fetch_all_jobs(job_ids))


# Resume Extraction Tool
@tool("ResumeExtractor")
def extract_resume() -> str:
    """
    Extract the content of uploaded resume from a PDF file.

    Extract and structure job-relevant information from an uploaded CV.

    Returns:
    str: The content of the highlight skills, experience, and qualifications relevant to job applications, omitting personal information
    """
    try:
        return load_resume("temp/resume.pdf")
    except (FileNotFoundError, ValueError) as exc:
        return f"Resume unavailable: {exc}"


# Cover Letter Generation Tool
@tool
def generate_letter_for_specific_job(resume_details: str, job_details: str) -> dict:
    """
    Generate a tailored cover letter using the provided CV and job details. This function constructs the letter as plain text.
    returns: A dictionary containing the job and resume details for generating the cover letter.
    """
    if not resume_details or not resume_details.strip():
        return {"error": "Resume details are required to generate a cover letter."}
    if resume_details.startswith("Resume unavailable:"):
        return {"error": resume_details}
    if not job_details or not job_details.strip():
        return {"error": "Job details are required to generate a cover letter."}
    return {
        "resume_details": resume_details,
        "job_details": job_details,
        "ready": True,
    }


@tool
def save_cover_letter_for_specific_job(
    cover_letter_content: str, company_name: str
) -> str:
    """
    Returns a download link for the generated cover letter.
    Params:
    cover_letter_content: The combine information of resume and job details to tailor the cover letter.
    """
    if not cover_letter_content or not cover_letter_content.strip():
        return "Unable to save an empty cover letter."
    safe_company_name = re.sub(r"[^A-Za-z0-9._-]+", "_", company_name).strip("._")
    if not safe_company_name:
        return "Unable to save the cover letter without a company name."
    filename = f"temp/{safe_company_name}_cover_letter.docx"
    file = write_cover_letter_to_doc(cover_letter_content, filename)
    abs_path = os.path.abspath(file)
    return f"Here is the download link: {abs_path}"


# Web Search Tools
@tool("google_search", parse_docstring=True)
def get_google_search_results(query: str) -> str:
    """Search the web for the given query and return the search results.

    Args:
        query: Search query for web.
    """
    response = SerperClient().search(query)
    items = response.get("items")
    string = []
    for result in items:
        try:
            string.append(
                "\n".join(
                    [
                        f"Title: {result['title']}",
                        f"Link: {result['link']}",
                        f"Snippet: {result['snippet']}",
                        "---",
                    ]
                )
            )
        except KeyError:
            continue

    content = "\n".join(string)
    return content


@tool("scrape_website", parse_docstring=True)
def scrape_website(url: str) -> str:
    """Scrape the content of a website and return the text.

    Args:
        url: Url to be scraped.
    """
    try:
        content = FirecrawlClient().scrape(url)
    except Exception as exc:
        return f"Failed to scrape {url}"
    return content
