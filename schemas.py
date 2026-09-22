from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


class RouteSchema(BaseModel):
    next_action: Literal[
        "ResumeAnalyzer",
        "CoverLetterGenerator",
        "JobSearcher",
        "WebResearcher",
        "ChatBot",
        "Finish",
    ] = Field(
        ...,
        title="Next",
        description="Select the next role",
    )


class JobSearchInput(BaseModel):
    keywords: str = Field(
        description="Keywords describing the job role. (if the user is looking for a role in particular company then pass company with keywords)"
    )
    location_name: str = Field(
        default="India",
        description='Name of the location to search within. Defaults to "India".',
    )
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
    ] = Field(
        default=["full-time"],
        description="Specific type(s) of job to search for.",
    )
    limit: Optional[int] = Field(
        default=10, ge=1, description="Maximum number of jobs to retrieve."
    )
    job_type: List[Literal["onsite", "remote", "hybrid"]] = Field(
        default=["onsite"],
        description="Filter for onsite, remote, or hybrid jobs.",
    )
    experience: Optional[
        List[
            Literal[
                "internship",
                "entry-level",
                "associate",
                "mid-senior-level",
                "director",
                "executive",
            ]
        ]
    ] = Field(
        default=["internship"],
        description='Filter by experience levels. Options are "internship", "entry level", "associate", "mid-senior level", "director", "executive". pass the exact arguments'
    )
    listed_at: Optional[Union[int, str]] = Field(
        default=86400,
        description="Maximum number of seconds passed since job posting. 86400 will filter job postings posted in the last 24 hours.",
    )
    distance: Union[int, str] = Field(
        default=100,
        description="Maximum distance from location in miles. Defaults to 100.",
    )
