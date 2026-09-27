import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from agents import (
    GraphContext,
    chatbot_node,
    cover_letter_generator_node,
    job_search_node,
    resume_analyzer_node,
    web_research_node,
)
from schemas import (
    CoverLetterResult,
    JobRecord,
    JobSearchResponse,
    ResearchResult,
    ResumeAnalysis,
    RouteOutput,
)


class TestStructuredContracts(unittest.TestCase):
    def test_route_output_restricts_routing_values(self):
        self.assertEqual(RouteOutput(next_action="ChatBot").next_action, "ChatBot")
        with self.assertRaises(ValueError):
            RouteOutput(next_action="Unknown")

    def test_resume_analysis_contract_has_application_fields(self):
        analysis = ResumeAnalysis(
            skills=["Python"],
            experience="Five years",
            qualifications="B.Tech",
            recommended_role="AI Engineer",
        )
        self.assertEqual(analysis.recommended_role, "AI Engineer")

    def test_job_search_response_contains_normalized_records(self):
        response = JobSearchResponse(
            jobs=[JobRecord(job_title="Engineer", company_name="Acme")]
        )
        self.assertEqual(response.jobs[0].company_name, "Acme")


class TestStructuredWorkerResults(unittest.TestCase):
    def _runtime(self):
        return SimpleNamespace(context=GraphContext(llm_config={}))

    def test_resume_worker_uses_structured_response(self):
        output = {
            "messages": [AIMessage(content="raw model output")],
            "structured_response": ResumeAnalysis(
                skills=["Python"],
                experience="Five years",
                qualifications="B.Tech",
                recommended_role="AI Engineer",
            ),
        }
        fake_agent = SimpleNamespace(invoke=lambda _input, _config: output)
        with patch("agents._llm"), patch(
            "agents.create_agent", return_value=fake_agent
        ):
            result = resume_analyzer_node(
                {"messages": [HumanMessage(content="analyze my resume")]},
                self._runtime(),
                {},
            )
        self.assertIn("AI Engineer", result["resume_analysis"])
        self.assertEqual(result["completed_workers"], ["ResumeAnalyzer"])

    def test_job_worker_uses_structured_response_and_renders_message(self):
        output = {
            "messages": [AIMessage(content="raw model output")],
            "structured_response": JobSearchResponse(
                jobs=[JobRecord(job_title="Engineer", company_name="Acme")]
            ),
        }
        fake_agent = SimpleNamespace(invoke=lambda _input, _config: output)
        with patch("agents._llm"), patch(
            "agents.create_agent", return_value=fake_agent
        ):
            result = job_search_node(
                {"messages": [HumanMessage(content="find jobs")]},
                self._runtime(),
                {},
            )
        self.assertEqual(result["job_results"][0]["company_name"], "Acme")
        self.assertIn("Engineer", result["messages"][0].content)

    def test_web_worker_uses_structured_summary(self):
        output = {
            "messages": [AIMessage(content="raw model output")],
            "structured_response": ResearchResult(summary="Research summary"),
        }
        fake_agent = SimpleNamespace(invoke=lambda _input, _config: output)
        with patch("agents._llm"), patch(
            "agents.create_agent", return_value=fake_agent
        ):
            result = web_research_node(
                {"messages": [HumanMessage(content="research this")]},
                self._runtime(),
                {},
            )
        self.assertEqual(result["messages"][0].content, "Research summary")

    def test_cover_letter_worker_uses_structured_result(self):
        output = {
            "messages": [AIMessage(content="raw model output")],
            "structured_response": CoverLetterResult(
                cover_letter="Dear hiring manager",
                download_link="/tmp/letter.docx",
            ),
        }
        fake_agent = SimpleNamespace(invoke=lambda _input, _config: output)
        state = {
            "messages": [HumanMessage(content="write a cover letter")],
            "selected_job": {"job_title": "Engineer", "company_name": "Acme"},
        }
        with patch("agents._llm"), patch(
            "agents.create_agent", return_value=fake_agent
        ):
            result = cover_letter_generator_node(state, self._runtime(), {})
        self.assertIn("Dear hiring manager", result["messages"][0].content)
        self.assertIn("/tmp/letter.docx", result["messages"][0].content)


class TestChatBotInvocation(unittest.TestCase):
    def test_chatbot_invokes_llm_with_system_message_and_history(self):
        captured = {}

        class FakeLLM:
            def invoke(self, messages, config):
                captured["messages"] = messages
                captured["config"] = config
                return AIMessage(content="Hello")

        with patch("agents._llm", return_value=FakeLLM()):
            result = chatbot_node(
                {"messages": [HumanMessage(content="Hi")]},
                SimpleNamespace(context=GraphContext(llm_config={})),
                {"tags": ["test"]},
            )
        self.assertIsInstance(captured["messages"][0], SystemMessage)
        self.assertEqual(captured["messages"][1].content, "Hi")
        self.assertEqual(captured["config"]["tags"], ["test"])
        self.assertEqual(result["messages"][0].content, "Hello")


if __name__ == "__main__":
    unittest.main()
