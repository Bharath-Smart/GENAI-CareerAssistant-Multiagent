"""
schemas.py

Defines Pydantic data models for validating agent inputs and controlling routing decisions in the LangGraph workflow.
"""

# -------------------- IMPORTS --------------------
# Standard Library
from typing import Literal, Optional, List, Union

# Third-Party Libraries
from pydantic import BaseModel, Field


class RouteSchema(BaseModel):
    """Used by the supervisor to decide which agent to route to next."""
    
    next_action: Literal[
        "ResumeAnalyzer",
        "CoverLetterGenerator",
        "JobSearcher",
        "WebResearcher",
        "ChatBot",
        "Finish",
    ] = Field(
        ...,  # Required field (no default)
        title="Next",
        description="Select the next role",
    )


class JobSearchInput(BaseModel):
    """Schema for user-defined job search parameters."""

    keywords: str = Field(
        description="Keywords describing the job role. If the user is targeting a company, include the company name in keywords."
    )

    location_name: Optional[str] = Field(
        description='Location to search within. Example: "Kyiv City, Ukraine".'
    )

    employment_type: Optional[
        List[
            Literal[
                "full-time", "contract", "part-time", "temporary",
                "internship", "volunteer", "other"
            ]
        ]
    ] = Field(
        description="Types of employment to filter by (can be multiple)."
    )

    limit: Optional[int] = Field(
        default=5,
        description="Max number of jobs to retrieve (default is 5)."
    )

    job_type: Optional[
        List[Literal["onsite", "remote", "hybrid"]]
    ] = Field(
        description="Filter based on job type (remote, onsite, hybrid)."
    )

    experience: Optional[
        List[
            Literal[
                "internship", "entry-level", "associate",
                "mid-senior-level", "director", "executive"
            ]
        ]
    ] = Field(
        description="Experience levels to filter by. Use exact terms listed."
    )

    listed_at: Optional[Union[int, str]] = Field(
        default=86400,
        description="Job postings created within the last N seconds. (e.g. 86400 = last 24 hours)"
    )

    distance: Optional[Union[int, str]] = Field(
        default=25,
        description="Max distance from location in miles. Default is 25 miles."
    )
