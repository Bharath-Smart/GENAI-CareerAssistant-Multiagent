import aiohttp
import os
import urllib
import asyncio
import requests
from typing import List, Literal, Union, Optional
from asgiref.sync import sync_to_async
from linkedin_api import Linkedin
from bs4 import BeautifulSoup

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


def build_linkedin_job_url(
    keywords,
    location=None,
    employment_type=None,
    experience_level=None,
    job_type=None,
    start=10,
):
    base_url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search/"

    # Prepare query parameters
    query_params = {
        "keywords": keywords,
    }

    if location:
        query_params["location"] = location

    if employment_type:
        if isinstance(employment_type, str):
            employment_type = [employment_type]
        employment_type = ",".join(employment_type)
        query_params["f_JT"] = employment_type

    if experience_level:
        if isinstance(experience_level, str):
            experience_level = [experience_level]
        experience_level = ",".join(experience_level)
        query_params["f_E"] = experience_level

    if job_type:
        if isinstance(job_type, str):
            job_type = [job_type]
        query_params["f_WT"] = ",".join(job_type)

    # Build the complete URL
    query_string = urllib.parse.urlencode(query_params)
    full_url = f"{base_url}?{query_string}&sortBy=R"

    return full_url


def validate_job_search_params(agent_input: Union[str, list], value_dict_mapping: dict):
    if isinstance(agent_input, list):
        return [item for item in agent_input if item in value_dict_mapping]
    elif isinstance(agent_input, str) and not value_dict_mapping.get(agent_input):
        agent_input = None
    return agent_input


def get_job_ids_from_linkedin_api(
    keywords: str,
    location_name: str,
    employment_type=None,
    limit: Optional[int] = 10,
    job_type=None,
    experience=None,
    listed_at=86400,
    distance=100,
):
    try:
        api = Linkedin(os.getenv("LINKEDIN_EMAIL"), os.getenv("LINKEDIN_PASS"))
        job_postings = api.search_jobs(
            keywords=keywords,
            job_type=employment_type,
            location_name=location_name,
            remote=job_type,
            limit=limit,
            experience=experience,
            listed_at=listed_at,
            distance=distance,
        )
        job_ids = []
        for job in job_postings:
            tracking_urn = job.get("trackingUrn", "")
            if "jobPosting:" not in tracking_urn:
                continue
            job_ids.append(tracking_urn.split("jobPosting:", 1)[1])
        return job_ids
    except Exception as e:
        print(f"Error in fetching job ids from LinkedIn API -> {e}")

    return []


def get_job_ids(
    keywords: str,
    location_name: str = "India",
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
    ] = ["full-time"],
    limit: Optional[int] = 10,
    job_type: Optional[List[Literal["onsite", "remote", "hybrid"]]] = ["onsite"],
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
    ] = ["internship"],
    listed_at: Optional[Union[int, str]] = 86400,
    distance: Optional[Union[int, str]] = 100,
):
    employment_type = validate_job_search_params(
        employment_type, employment_type_mapping
    )
    job_type = validate_job_search_params(job_type, job_type_mapping)
    experience = validate_job_search_params(experience, experience_type_mapping)

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

        # Parse the HTML and ignore cards that do not contain a usable ID.
        list_data = response.text
        list_soup = BeautifulSoup(list_data, "html.parser")
        page_jobs = list_soup.find_all("li")

        job_ids = []
        for job in page_jobs:
            base_card_div = job.find("div", {"class": "base-card"})
            if base_card_div is None:
                continue
            entity_urn = base_card_div.get("data-entity-urn", "")
            entity_parts = entity_urn.split(":")
            if len(entity_parts) > 3 and entity_parts[3]:
                job_ids.append(entity_parts[3])
        return job_ids
    except Exception as e:
        print(f"Error in fetching job ids from LinkedIn -> {e}")
    return []


async def fetch_job_details(session, job_id):
    # Construct the URL for each job using the job ID
    job_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"

    # Send a GET request to the job URL
    async with session.get(job_url) as response:
        job_soup = BeautifulSoup(await response.text(), "html.parser")

        # Create a dictionary to store job details
        job_post = {
            "job_title": "",
            "company_name": "",
            "job_location": "",
            "job_description": "",
            "apply_url": "",
            "posted_age": "",
            "num_applicants": "",
        }

        # Try to extract and store the job title
        try:
            job_post["job_title"] = job_soup.find(
                "h2",
                {
                    "class": "top-card-layout__title font-sans text-lg papabear:text-xl font-bold leading-open text-color-text mb-0 topcard__title"
                },
            ).text.strip()
        except Exception as exc:
            job_post["job_title"] = ""

        try:
            job_post["job_location"] = job_soup.find(
                "span",
                {"class": "topcard__flavor topcard__flavor--bullet"},
            ).text.strip()
        except Exception as exc:
            job_post["job_location"] = ""

        # Try to extract and store the company name
        try:
            job_post["company_name"] = job_soup.find(
                "a", {"class": "topcard__org-name-link topcard__flavor--black-link"}
            ).text.strip()
        except Exception as exc:
            job_post["company_name"] = ""

        # Try to extract and store the time posted
        try:
            job_post["posted_age"] = job_soup.find(
                "span", {"class": "posted-time-ago__text topcard__flavor--metadata"}
            ).text.strip()
        except Exception as exc:
            job_post["posted_age"] = ""

        # Try to extract and store the number of applicants
        try:
            job_post["num_applicants"] = job_soup.find(
                "span",
                {
                    "class": "num-applicants__caption topcard__flavor--metadata topcard__flavor--bullet"
                },
            ).text.strip()
        except Exception as exc:
            job_post["num_applicants"] = ""

        # Try to extract and store the job description
        try:
            job_description = job_soup.find(
                "div", {"class": "decorated-job-posting__details"}
            ).text.strip()
            job_post["job_description"] = job_description
        except Exception as exc:
            job_post["job_description"] = ""

        try:
            # Try to extract and store the apply link
            apply_link_tag = job_soup.find("a", class_="topcard__link")
            if apply_link_tag:
                job_post["apply_url"] = apply_link_tag.get("href", "")
        except Exception as exc:
            job_post["apply_url"] = ""

        return job_post


async def get_job_details_from_linkedin_api(job_id):
    try:
        api = Linkedin(os.getenv("LINKEDIN_EMAIL"), os.getenv("LINKEDIN_PASS"))
        job_data = await sync_to_async(api.get_job)(
            job_id
        )  # Assuming this function is async and fetches job data

        # Construct the job data dictionary with defaults
        job_data_dict = {
            "company_name": job_data.get("companyDetails", {})
            .get(
                "com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany",
                {},
            )
            .get("companyResolutionResult", {})
            .get("name", ""),
            "company_url": job_data.get("companyDetails", {})
            .get(
                "com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany",
                {},
            )
            .get("companyResolutionResult", {})
            .get("url", ""),
            "job_description": job_data.get("description", {}).get("text", ""),
            "work_remote_allowed": job_data.get("workRemoteAllowed", ""),
            "job_title": job_data.get("title", ""),
            "apply_url": job_data.get("applyMethod", {})
            .get("com.linkedin.voyager.jobs.OffsiteApply", {})
            .get("companyApplyUrl", ""),
            "job_location": job_data.get("formattedLocation", ""),
            "posted_age": "",
            "num_applicants": "",
        }
    except Exception as e:
        # Handle exceptions or errors in fetching or parsing the job data
        job_data_dict = {
            "company_name": "",
            "company_url": "",
            "job_description": "",
            "work_remote_allowed": "",
            "job_title": "",
            "apply_url": "",
            "job_location": "",
            "posted_age": "",
            "num_applicants": "",
        }

    return job_data_dict


async def fetch_all_jobs(job_ids, batch_size=5):
    results = []

    try:
        if os.environ.get("LINKEDIN_SEARCH") == "linkedin_api":
            return await asyncio.gather(
                *[get_job_details_from_linkedin_api(job_id) for job_id in job_ids]
            )

        async with aiohttp.ClientSession() as session:
            tasks = []
            for job_id in job_ids:
                task = asyncio.create_task(fetch_job_details(session, job_id))
                tasks.append(task)

            # Await the completion of all tasks
            results = await asyncio.gather(*tasks)
            return results
    except Exception as exc:
        print(f"Error in fetching job details -> {exc}")

    return results
