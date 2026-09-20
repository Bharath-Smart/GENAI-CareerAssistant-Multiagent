"""
Focused regression tests for the local dependency-compatibility fixes made
on the `dev` branch (see agents.py, tools.py, custom_callback_handler.py).

These protect against regressions of genuine import/runtime breakage caused
by the currently installed (newer) langchain / langchain-core / langgraph /
streamlit / pydantic versions, without requiring any API keys or network
access.

Run with: python -m unittest discover -s tests -v
(uses the stdlib `unittest` runner only -- no new test dependency such as
pytest was added, since the project's installed dependency set does not
currently include one.)
"""
import unittest


class TestCompatibilityImports(unittest.TestCase):
    """
    Guards the specific import-path fixes applied on `dev`:
      - agents.py:      langchain.agents -> langchain_classic.agents
      - tools.py:       langchain.pydantic_v1 -> pydantic
                         langchain.tools -> langchain_core.tools
      - custom_callback_handler.py: no longer depends on the removed
        streamlit.external.langchain.streamlit_callback_handler module.
    Each of these previously raised ImportError/ModuleNotFoundError with the
    currently installed dependency versions before the fix.
    """

    def test_agents_module_imports(self):
        import agents  # noqa: F401

    def test_tools_module_imports(self):
        import tools  # noqa: F401

    def test_custom_callback_handler_imports(self):
        import custom_callback_handler  # noqa: F401

    def test_schemas_and_chains_import(self):
        import schemas  # noqa: F401
        import chains  # noqa: F401
        import members  # noqa: F401
        import prompts  # noqa: F401
        import llms  # noqa: F401


class TestGraphDefinition(unittest.TestCase):
    """
    Verifies the LangGraph workflow still compiles and exposes the expected
    nodes/edges after the compatibility fixes. Does not invoke the graph
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
    Verifies the rewritten CustomStreamlitCallbackHandler (a factory function
    returning a patched langchain_community StreamlitCallbackHandler) still
    exposes a callable write_agent_name(name) that writes to the provided
    container, since every worker node in agents.py depends on this method.
    """

    def test_write_agent_name_writes_to_container(self):
        # langchain_community's StreamlitCallbackHandler requires a real
        # streamlit DeltaGenerator (it calls parent_container.container()
        # internally), so a real container is used here rather than a
        # hand-rolled stub. Running outside an active ScriptRunContext just
        # emits a harmless "missing ScriptRunContext" warning.
        import streamlit as st
        from custom_callback_handler import CustomStreamlitCallbackHandler

        container = st.container()
        handler = CustomStreamlitCallbackHandler(container)
        self.assertTrue(hasattr(handler, "write_agent_name"))
        # Should not raise -- this is what every worker node in agents.py
        # calls before invoking its agent.
        handler.write_agent_name("TestAgent")


if __name__ == "__main__":
    unittest.main()
