import os
import tempfile
import unittest

import pymupdf

from data_loader import load_resume
from schemas import JobSearchInput
from search import build_linkedin_job_url, validate_job_search_params
from tools import generate_letter_for_specific_job


class TestResumeLifecycle(unittest.TestCase):
    def test_missing_resume_is_explicit(self):
        with self.assertRaises(FileNotFoundError):
            load_resume(os.path.join(tempfile.gettempdir(), "missing-resume.pdf"))

    def test_invalid_pdf_is_explicit(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as resume_file:
            resume_file.write(b"not a PDF")
            resume_file.flush()
            with self.assertRaises(ValueError):
                load_resume(resume_file.name)

    def test_valid_pdf_is_readable(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as resume_file:
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Candidate experience")
            document.save(resume_file.name)
            document.close()
            self.assertEqual(
                load_resume(resume_file.name).strip(),
                "Candidate experience",
            )


class TestJobSearchContracts(unittest.TestCase):
    def test_defaults_match_phase_one_contract(self):
        fields = JobSearchInput.model_fields
        self.assertEqual(fields["location_name"].default, "India")
        self.assertEqual(fields["employment_type"].default, ["full-time"])
        self.assertEqual(fields["limit"].default, 10)
        self.assertEqual(fields["job_type"].default, ["onsite"])
        self.assertEqual(fields["experience"].default, ["internship"])
        self.assertEqual(fields["listed_at"].default, 86400)
        self.assertEqual(fields["distance"].default, 100)

    def test_explicit_job_search_parameters_are_preserved(self):
        params = JobSearchInput(
            keywords="AI engineer",
            location_name="Berlin",
            employment_type=["contract"],
            limit=3,
            job_type=["remote"],
            experience=["mid-senior-level"],
            listed_at=3600,
            distance=25,
        )
        self.assertEqual(params.location_name, "Berlin")
        self.assertEqual(params.limit, 3)
        self.assertIn("f_JT=contract", build_linkedin_job_url(
            params.keywords,
            params.location_name,
            params.employment_type,
            params.experience,
            params.job_type,
        ))
        self.assertEqual(
            validate_job_search_params(["remote", "invalid"], {"remote": "2"}),
            ["remote"],
        )

    def test_malformed_job_cards_are_ignored(self):
        from unittest.mock import patch
        import search

        response = type(
            "Response",
            (),
            {"text": "<li><div class='base-card'></div></li>", "raise_for_status": lambda self: None},
        )()
        with patch.object(search.requests, "get", return_value=response):
            self.assertEqual(search.get_job_ids("AI engineer"), [])

    def test_job_tool_returns_structured_records(self):
        from unittest.mock import patch
        from tools import linkedin_job_search

        records = [{"job_title": "Engineer", "company_name": "Acme"}]
        async def fake_fetch_all_jobs(_job_ids):
            return records

        with patch("tools.get_job_ids", return_value=["123"]), patch(
            "tools.fetch_all_jobs", new=fake_fetch_all_jobs
        ):
            result = linkedin_job_search.invoke({"keywords": "AI engineer"})
        self.assertEqual(result, records)


class TestCoverLetterContracts(unittest.TestCase):
    def test_cover_letter_requires_resume_and_job_details(self):
        self.assertIn(
            "Resume details are required",
            generate_letter_for_specific_job.invoke(
                {"resume_details": "", "job_details": "Engineer at Acme"}
            )["error"],
        )
        self.assertIn(
            "Job details are required",
            generate_letter_for_specific_job.invoke(
                {"resume_details": "Candidate experience", "job_details": ""}
            )["error"],
        )


if __name__ == "__main__":
    unittest.main()
