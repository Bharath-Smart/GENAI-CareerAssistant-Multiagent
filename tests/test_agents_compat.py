"""
Focused regression tests covering:
  1. Dependency-compatibility fixes made on the `dev` branch (imports that
     previously broke with the currently installed langchain / langgraph /
     streamlit / pydantic versions).
  2. The subsequent modernization migration (langchain.agents.create_agent,
     MessagesState/add_messages, direct PyMuPDF, the lean callback handler,
     and the @tool(args_schema=...) job-search tool).

These do not require any API keys or network access.

Run with: python -m unittest discover -s tests -v
(uses the stdlib `unittest` runner only -- no new test dependency such as
pytest was added, since the project's installed dependency set does not
currently include one.)
"""
import os
import unittest


class TestCompatibilityImports(unittest.TestCase):
    """
    Guards that every application module still imports cleanly against the
    currently installed dependency versions (langchain 1.x / langgraph 1.x /
    streamlit 1.x / pydantic 2.x), after both the original compatibility
    fixes and the subsequent modernization migration (langchain.agents
    .create_agent, direct pymupdf, direct Firecrawl SDK, direct Serper API).
    """

    def test_agents_module_imports(self):
        import agents  # noqa: F401

    def test_tools_module_imports(self):
        import tools  # noqa: F401

    def test_custom_callback_handler_imports(self):
        import custom_callback_handler  # noqa: F401

    def test_data_loader_and_utils_import(self):
        import data_loader  # noqa: F401
        import utils  # noqa: F401

    def test_schemas_and_chains_import(self):
        import schemas  # noqa: F401
        import chains  # noqa: F401
        import members  # noqa: F401
        import prompts  # noqa: F401


class TestGraphDefinition(unittest.TestCase):
    """
    Verifies the LangGraph workflow still compiles and exposes the expected
    nodes/edges after migrating AgentState to MessagesState/add_messages and
    workers to langchain.agents.create_agent. Does not invoke the graph
    (no LLM/network calls), so it is safe to run without API keys.
    """

    def test_define_graph_compiles_with_expected_nodes(self):
        from agents import define_graph

        graph = define_graph()
        node_names = set(graph.get_graph().nodes.keys())
        expected = {
            "__start__",
            "__end__",
            "Supervisor",
            "ResumeAnalyzer",
            "JobSearcher",
            "CoverLetterGenerator",
            "WebResearcher",
            "ChatBot",
        }
        self.assertEqual(node_names, expected)


class TestCustomStreamlitCallbackHandler(unittest.TestCase):
    """
    Verifies the rewritten CustomStreamlitCallbackHandler (now a plain
    langchain_core BaseCallbackHandler subclass, no langchain_community
    dependency) still exposes a callable write_agent_name(name) that writes
    to the provided container, since every worker node in agents.py depends
    on this method.
    """

    def test_write_agent_name_writes_to_container(self):
        from custom_callback_handler import CustomStreamlitCallbackHandler

        written = []

        class FakeContainer:
            def write(self, value):
                written.append(value)

        handler = CustomStreamlitCallbackHandler(FakeContainer())
        handler.write_agent_name("TestAgent")
        self.assertEqual(written, ["TestAgent"])


class TestJobSearchTool(unittest.TestCase):
    """
    Verifies linkedin_job_search is now a proper BaseTool built via
    @tool(args_schema=JobSearchInput), replacing the old
    get_job_search_tool()/StructuredTool.from_function(...) factory.
    """

    def test_linkedin_job_search_is_a_tool_with_expected_schema(self):
        from langchain_core.tools import BaseTool
        from tools import linkedin_job_search
        from schemas import JobSearchInput

        self.assertIsInstance(linkedin_job_search, BaseTool)
        self.assertEqual(linkedin_job_search.name, "JobSearchTool")
        self.assertIs(linkedin_job_search.args_schema, JobSearchInput)


class TestPdfTextExtraction(unittest.TestCase):
    """
    Functional (no network) regression test for the data_loader.py migration
    from langchain_community's PyMuPDFLoader to direct `pymupdf.open(...)`.
    Uses the repo's own dummy_resume.pdf, so no external resume is required.
    """

    def test_load_resume_extracts_nonempty_text(self):
        from data_loader import load_resume

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dummy_resume_path = os.path.join(repo_root, "dummy_resume.pdf")
        if not os.path.exists(dummy_resume_path):
            self.skipTest("dummy_resume.pdf not present in repo root")

        text = load_resume(dummy_resume_path)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text.strip()), 0)


if __name__ == "__main__":
    unittest.main()
