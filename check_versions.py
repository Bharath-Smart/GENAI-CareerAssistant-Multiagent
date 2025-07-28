"""
check_versions.py

Utility script to verify that required packages are installed at expected versions.
Useful during collaboration or deployment to ensure environment consistency.
"""

# -------------------- IMPORTS --------------------
# Standard Library
import importlib.metadata as metadata

# -------------------- CONFIGURATION --------------------
# Define expected versions for critical dependencies used in the project
expected_versions = {
    "langchain": "0.3.26",
    "langchain-core": "0.3.66",
    "langchain-community": "0.3.26",
    "langchain-openai": "0.3.25",
    "langchain-groq": "0.3.4",
    "langgraph": "0.4.9",
    "langsmith": "0.4.1",
    "streamlit": "1.46.0",
    "streamlit-chat": "0.1.1",
    "streamlit-pills": "0.3.0",
    "streamlit-analytics2": "0.10.5",
    "pypdf": "5.6.1",
    "PyMuPDF": "1.26.1",
    "python-dotenv": "1.1.1",
    "linkedin-api": "2.3.1",
    "firecrawl-py": "2.9.0",
    "python-docx": "1.2.0"
}

# -------------------- VERSION CHECK LOGIC --------------------
# Print out each package's version and compare with expected
for pkg, required in expected_versions.items():
    try:
        installed = metadata.version(pkg)
        if installed == required:
            print(f"{pkg}: {installed}")
        else:
            print(f"{pkg}: {installed} (expected {required})")
    except metadata.PackageNotFoundError:
        print(f"{pkg}: Not installed")
