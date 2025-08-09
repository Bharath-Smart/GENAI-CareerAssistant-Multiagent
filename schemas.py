"""
schemas.py

Defines Pydantic v2-compatible data models for validating agent inputs
and routing decisions used in the LangGraph multi-agent workflow.
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

    keywords: Union[str, List[str]] = Field(
        ...,  # Required
        description="Keyword(s) describing the job role. If the user is targeting a company, include the company name in keywords."
    )

    location_name: Optional[str] = Field(
        default=None,
        description='Location to search within. Example: "Kyiv City, India".'
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
                "other"                
            ]
        ]
    ] = Field(
        default=None,
        description="Types of employment to filter by (multiple allowed)."
    )

    limit: Optional[int] = Field(
        default=5,
        description="Maximum number of jobs to retrieve (default = 5)."
    )

    job_type: Optional[
        List[
            Literal[
                "onsite", 
                "remote", 
                "hybrid"
            ]
        ]
    ] = Field(
        default=None,
        description="Filter based on job type (remote, onsite, hybrid)."
    )

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
    ] = Field(
        default=None,
        description="Experience levels to filter by. Use exact terms listed."
    )

    listed_at: Optional[Union[int, str]] = Field(
        default=86400,
        description="Job postings created within the last N seconds (e.g., 86400 = 24 hrs)."
    )

    distance: Optional[Union[int, str]] = Field(
        default=25,
        description="Max distance from location in miles. Default is 25 miles."
    )

class GoogleSearchInput(BaseModel):
    query: str = Field(..., description="Search query for Google/Serper API")

class ScrapeWebsiteInput(BaseModel):
    url: str = Field(..., description="URL of the website to scrape")