"""
data_loader.py

Handles resume parsing and cover letter generation in file formats suitable for agent processing.

Functionality includes:
- Extracting plain text from uploaded PDF resumes using PyMuPDF.
- Writing generated cover letters into .docx format using python-docx.

Used by the ResumeAnalyzer and CoverLetterGenerator agents.
"""

# -------------------- IMPORTS --------------------
# LangChain / LangGraph
from langchain_community.document_loaders import PyMuPDFLoader  # For loading PDF resumes

# External Libraries
from docx import Document  # For creating .docx cover letters

# -------------------- RESUME LOADING --------------------
def load_resume(file_path: str) -> str:
    """
    Load and return text content from a PDF resume using PyMuPDF.

    Args:
        file_path (str): Path to the resume PDF file.

    Returns:
        str: Combined text from all pages.
    """
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    return "".join(page.page_content for page in pages)

# -------------------- COVER LETTER WRITING --------------------
def write_cover_letter_to_doc(text: str, filename: str = "temp/cover_letter.docx") -> str:
    """
    Write a cover letter to a Word (.docx) file.

    Args:
        text (str): The content of the cover letter.
        filename (str): Path where the document should be saved.

    Returns:
        str: Path to the saved .docx file.
    """
    doc = Document()
    for para in text.split("\n"):
        doc.add_paragraph(para)
    doc.save(filename)
    return filename
