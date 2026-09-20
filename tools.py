# define tools
import os
import asyncio
from dotenv import load_dotenv
from langchain_core.tools import BaseTool, tool
from data_loader import load_resume, write_cover_letter_to_doc
from schemas import JobSearchInput
from search import get_job_ids, fetch_all_jobs
from utils import FireCrawlClient, SerperClient

load_dotenv()


# Job search tool.
# Uses @tool(args_schema=...) directly (the current documented pattern for a
# plain function with an explicit schema), replacing the previous
# StructuredTool.from_function(...) wrapper/factory.
@tool("JobSearchTool", args_schema=JobSearchInput)
def linkedin_job_search(
    keywords: str,
    location_name: str = None,
    job_type: str = None,
    limit: int = 5,
    employment_type: str = None,
    listed_at=None,
    experience=None,
    distance=None,
) -> dict:  # type: ignore
    """
    Search LinkedIn for job postings based on specified criteria. Returns detailed job listings.
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
    job_desc = asyncio.run(fetch_all_jobs(job_ids))
    return job_desc


# Resume Extraction Tool
class ResumeExtractorTool(BaseTool):
    """
    Extract the content of a resume from a PDF file.
    Returns:
        dict: The extracted content of the resume.
    """
    name: str = "ResumeExtractor"
    description: str = "Extract the content of uploaded resume from a PDF file."

    def extract_resume(self) -> str:
        """
        Extract resume content from a PDF file.
        Extract and structure job-relevant information from an uploaded CV.

        Returns:
        str: The content of the highlight skills, experience, and qualifications relevant to job applications, omitting personal information
        """
        text = load_resume("temp/resume.pdf")
        return text

    def _run(self) -> dict:
        return self.extract_resume()


# Cover Letter Generation Tool
@tool
def generate_letter_for_specific_job(resume_details: str, job_details: str) -> dict:
    """
    Generate a tailored cover letter using the provided CV and job details. This function constructs the letter as plain text.
    returns: A dictionary containing the job and resume details for generating the cover letter.
    """
    return {"job_details": job_details, "resume_details": resume_details}


@tool
def save_cover_letter_for_specific_job(
    cover_letter_content: str, company_name: str
) -> str:
    """
    Returns a download link for the generated cover letter.
    Params:
    cover_letter_content: The combine information of resume and job details to tailor the cover letter.
    """
    filename = f"temp/{company_name}_cover_letter.docx"
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
        content = FireCrawlClient().scrape(url)
    except Exception as exc:
        return f"Failed to scrape {url}"
    return content
