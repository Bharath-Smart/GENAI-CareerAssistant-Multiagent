"""
members.py

Defines and registers all agent nodes for the LangGraph workflow.

Each agent is represented as a LangChain Runnable and has a specific role:
- ResumeAnalyzer
- JobSearcher
- CoverLetterGenerator
- WebResearcher
- ChatBot

These members are coordinated by the Supervisor agent for task delegation.
"""

# -------------------- IMPORTS --------------------
# Standard Library
from typing import List, Dict

def get_team_members_details() -> List[Dict[str, str]]:
    """
    Returns a list of dictionaries, each containing details of a team member.

    Each dictionary includes:
    - name (str): The name of the team member.
    - description (str): A brief summary of their role and responsibilities.

    Returns:
        List[Dict[str, str]]: List of team member descriptions.
    """
    members_dict = [
        {
            "name": "ResumeAnalyzer",
            "description": "Responsible for analyzing resumes to extract key information.",
        },
        {
            "name": "CoverLetterGenerator",
            "description": "Specializes in creating and optimizing cover letters tailored to job descriptions. Highlights the candidate's strengths and ensures the cover letter aligns with the requirements of the position.",
        },
        {
            "name": "JobSearcher",
            "description": "Conducts job searches based on specified criteria such as industry, location, and job title.",
        },
        {
            "name": "WebResearcher",
            "description": "Conducts online research to gather information from the web.",
        },
        {
            "name": "ChatBot",
            "description": "Answers user queries or retrieves context from prior messages.",
        },
        {
            "name": "Finish",
            "description": "Represents the end of the workflow.",
        },
    ]
    return members_dict
