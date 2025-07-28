"""
search.py

This module handles job discovery from LinkedIn, supporting both:
1. Web scraping from public LinkedIn job listings.
2. Unofficial LinkedIn API via `linkedin_api`.

Includes logic for:
- Constructing LinkedIn search URLs based on user filters (keywords, location, experience, etc.).
- Validating job search parameters.
- Fetching job IDs and parsing job details.
- Supporting both synchronous API mode and asynchronous scraping mode.

Used by the JobSearcher agent to retrieve job data in the LangGraph workflow.
"""

# -------------------- IMPORTS --------------------
# Standard Library
import os
import urllib
import asyncio
import requests
import aiohttp
from typing import List, Literal, Union, Optional

# Async Utilities
from asgiref.sync import sync_to_async

# External Libraries
from linkedin_api import Linkedin
from bs4 import BeautifulSoup

# -------------------- CONSTANTS --------------------
employment_type_mapping = {
    "full-time": "F",
    "contract": "C",
    "part-time": "P",
    "temporary": "T",
    "internship": "I",
    "volunteer": "V",
    "other": "O",
}

experience_type_mapping = {
    "internship": "1",
    "entry-level": "2",
    "associate": "3",
    "mid-senior-level": "4",
    "director": "5",
    "executive": "6",
}

job_type_mapping = {
    "onsite": "1",
    "remote": "2",
    "hybrid": "3",
}

# -------------------- JOB URL CONSTRUCTION --------------------
def build_linkedin_job_url(
    keywords: Union[str, List[str]],
    location=None,
    employment_type=None,
    experience_level=None,
    job_type=None,
):
    base_url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search/"
    
    if isinstance(keywords, list):
        keywords = ", ".join(keywords)
    query_params = {"keywords": keywords}

    if location:
        query_params["location"] = location

    if employment_type:
        if isinstance(employment_type, str):
            employment_type = [employment_type]
        query_params["f_WT"] = ",".join(employment_type)

    if experience_level:
        if isinstance(experience_level, str):
            experience_level = [experience_level]
        query_params["f_E"] = ",".join(experience_level)

    if job_type:
        if isinstance(job_type, str):
            job_type = [job_type]
        query_params["f_WT"] = ",".join(job_type)

    # Build the complete URL
    query_string = urllib.parse.urlencode(query_params)
    full_url = f"{base_url}?{query_string}&sortBy=R"

    return full_url

# -------------------- PARAMETER VALIDATION --------------------
def validate_job_search_params(agent_input: Union[str, list], value_dict_mapping: dict):
    if isinstance(agent_input, list):
        return [val for val in agent_input if val in value_dict_mapping]
    elif isinstance(agent_input, str):
        return agent_input if agent_input in value_dict_mapping else None
    return None

# -------------------- JOB SEARCH USING LINKEDIN API --------------------
def get_job_ids_from_linkedin_api(
    keywords: Union[str, List[str]],
    location_name: str,
    employment_type=None,
    limit: Optional[int] = 5,
    job_type=None,
    experience=None,
    listed_at=86400,
    distance=None,
):
    try:
        job_type = validate_job_search_params(job_type, job_type_mapping)
        employment_type = validate_job_search_params(employment_type, employment_type_mapping)
        experience_level = validate_job_search_params(experience, experience_type_mapping)

        if isinstance(keywords, list):
            keywords = ", ".join(keywords)

        api = Linkedin(os.getenv("LINKEDIN_EMAIL"), os.getenv("LINKEDIN_PASS"))
        job_postings = api.search_jobs(
            keywords=keywords,
            job_type=employment_type,
            location_name=location_name,
            remote=job_type,
            limit=limit,
            experience=experience_level,
            listed_at=listed_at,
            distance=distance,
        )

        job_ids = [job["trackingUrn"].split("jobPosting:")[1] for job in job_postings]
        return job_ids
    except Exception as e:
        print(f"Error in fetching job ids from LinkedIn API -> {e}")
        return []

# -------------------- JOB SEARCH CONTROLLER --------------------
def get_job_ids(
    keywords: Union[str, List[str]],
    location_name: str,
    employment_type: Optional[
        List[
            Literal[
                "full-time",
                "contract",
                "part-time",
                "temporary",
                "internship",
                "volunteer",
                "other",
            ]
        ]
    ] = None,
    limit: Optional[int] = 10,
    job_type: Optional[List[Literal["onsite", "remote", "hybrid"]]] = None,
    experience: Optional[
        List[
            Literal[
                "internship",
                "entry level",
                "associate",
                "mid-senior level",
                "director",
                "executive",
            ]
        ]
    ] = None,
    listed_at: Optional[Union[int, str]] = 86400,
    distance=None,
):
    if os.environ.get("LINKEDIN_SEARCH") == "linkedin_api":
        return get_job_ids_from_linkedin_api(
            keywords=keywords,
            location_name=location_name,
            employment_type=employment_type,
            limit=limit,
            job_type=job_type,
            experience=experience,
            listed_at=listed_at,
            distance=distance,
        )

    try:
        # Construct the URL for LinkedIn job search
        job_url = build_linkedin_job_url(
            keywords=keywords,
            location=location_name,
            employment_type=employment_type,
            experience_level=experience,
            job_type=job_type,
        )

        # Send a GET request to the URL and store the response
        response = requests.get(
            job_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}
        )

        # Get the HTML, parse the response and find all list items(jobs postings)
        list_soup = BeautifulSoup(response.text, "html.parser")
        page_jobs = list_soup.find_all("li")

        job_ids = []
        for job in page_jobs:
            base_card_div = job.find("div", {"class": "base-card"})
            job_id = base_card_div.get("data-entity-urn").split(":")[3]
            job_ids.append(job_id)
        return job_ids
    except Exception as e:
        print(f"Error in fetching job ids from LinkedIn -> {e}")
        return []

# -------------------- PUBLIC LINKEDIN SCRAPING --------------------
async def fetch_job_details(session, job_id):
    # Construct the URL for each job using the job ID    
    job_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"

    # Send a GET request to the job URL    
    async with session.get(job_url) as response:
        job_soup = BeautifulSoup(await response.text(), "html.parser")
        
        # Create a dictionary to store job details        
        job_post = {}

        try:
            job_post["job_title"] = job_soup.find("h2", {"class": "top-card-layout__title"}).text.strip()
        except:
            job_post["job_title"] = ""

        try:
            job_post["job_location"] = job_soup.find("span", {"class": "topcard__flavor--bullet"}).text.strip()
        except:
            job_post["job_location"] = ""

        try:
            job_post["company_name"] = job_soup.find("a", {"class": "topcard__org-name-link"}).text.strip()
        except:
            job_post["company_name"] = ""

        try:
            job_post["time_posted"] = job_soup.find("span", {"class": "posted-time-ago__text"}).text.strip()
        except:
            job_post["time_posted"] = ""

        try:
            job_post["num_applicants"] = job_soup.find("span", {"class": "num-applicants__caption"}).text.strip()
        except:
            job_post["num_applicants"] = ""

        try:
            job_post["job_desc_text"] = job_soup.find("div", {"class": "decorated-job-posting__details"}).text.strip()
        except:
            job_post["job_desc_text"] = ""

        try:
            apply_link_tag = job_soup.find("a", class_="topcard__link")
            job_post["apply_link"] = apply_link_tag.get("href") if apply_link_tag else ""
        except:
            job_post["apply_link"] = ""

        return job_post

# -------------------- LINKEDIN API JOB DETAILS --------------------
async def get_job_details_from_linkedin_api(job_id):
    try:
        api = Linkedin(os.getenv("LINKEDIN_EMAIL"), os.getenv("LINKEDIN_PASS"))
        job_data = await sync_to_async(api.get_job)(job_id)

        return {
            "company_name": job_data.get("companyDetails", {})
            .get("com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany", {})
            .get("companyResolutionResult", {})
            .get("name", ""),

            "company_url": job_data.get("companyDetails", {})
            .get("com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany", {})
            .get("companyResolutionResult", {})
            .get("url", ""),

            "job_desc_text": job_data.get("description", {}).get("text", ""),
            "work_remote_allowed": job_data.get("workRemoteAllowed", ""),
            "job_title": job_data.get("title", ""),
            "company_apply_url": job_data.get("applyMethod", {})
            .get("com.linkedin.voyager.jobs.OffsiteApply", {})
            .get("companyApplyUrl", ""),
            "job_location": job_data.get("formattedLocation", ""),
        }
    except:
        return {
            "company_name": "",
            "company_url": "",
            "job_desc_text": "",
            "work_remote_allowed": "",
            "job_title": "",
            "apply_link": "",
            "job_location": "",
        }

# -------------------- FETCH ALL JOBS --------------------
async def fetch_all_jobs(job_ids, batch_size=5):
    try:
        if os.environ.get("LINKEDIN_SEARCH") == "linkedin_api":
            return await asyncio.gather(*[get_job_details_from_linkedin_api(job_id) for job_id in job_ids])

        async with aiohttp.ClientSession() as session:
            tasks = [asyncio.create_task(fetch_job_details(session, job_id)) for job_id in job_ids]
            return await asyncio.gather(*tasks)
    except Exception as exc:
        print(f"Error in fetching job details -> {exc}")
        return []
