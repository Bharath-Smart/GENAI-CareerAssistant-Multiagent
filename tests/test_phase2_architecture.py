import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from agents import (
    AgentState,
    GraphContext,
    _resolve_next_action,
    cover_letter_generator_node,
    job_search_node,
    supervisor_node,
    define_graph,
)


class TestRoutingContract(unittest.TestCase):
    def test_finish_before_worker_routes_to_chatbot(self):
        self.assertEqual(
            _resolve_next_action("Finish", {"completed_workers": []}),
            "ChatBot",
        )

    def test_finish_after_worker_is_terminal(self):
        self.assertEqual(
            _resolve_next_action("Finish", {"completed_workers": ["JobSearcher"]}),
            "Finish",
        )

    def test_chatbot_is_terminal_after_one_execution(self):
        self.assertEqual(
            _resolve_next_action(
                "ChatBot",
                {"completed_workers": ["JobSearcher", "ChatBot"]},
            ),
            "Finish",
        )

    def test_invalid_route_is_safe(self):
        self.assertEqual(
            _resolve_next_action("Unexpected", {"completed_workers": []}),
            "ChatBot",
        )
        self.assertEqual(
            _resolve_next_action(
                "Unexpected", {"completed_workers": ["JobSearcher"]}
            ),
            "Finish",
        )

    def test_completed_worker_is_not_selected_again(self):
        self.assertEqual(
            _resolve_next_action(
                "JobSearcher", {"completed_workers": ["JobSearcher"]}
            ),
            "ChatBot",
        )

    def test_supervisor_returns_explicit_command_route(self):
        fake_chain = SimpleNamespace(
            invoke=lambda _input, _config: SimpleNamespace(next_action="JobSearcher")
        )
        state = {"messages": [HumanMessage(content="find jobs")]}
        runtime = SimpleNamespace(context=GraphContext(llm_config={}))
        with patch("agents._llm"), patch(
            "agents.get_supervisor_chain", return_value=fake_chain
        ):
            command = supervisor_node(state, runtime, {})
        self.assertIsInstance(command, Command)
        self.assertEqual(command.goto, "JobSearcher")


class TestWorkerContracts(unittest.TestCase):
    def test_job_worker_keeps_structured_results_separate_from_messages(self):
        records = [{"job_title": "Engineer", "company_name": "Acme"}]
        fake_agent = SimpleNamespace(
            invoke=lambda _input, _config: {
                "messages": [
                    ToolMessage(
                        name="JobSearchTool",
                        content=json.dumps(records),
                        tool_call_id="test-call",
                    ),
                    AIMessage(content="formatted by worker"),
                ]
            }
        )
        state = {"messages": [HumanMessage(content="find jobs")]}
        runtime = SimpleNamespace(context=GraphContext(llm_config={}))
        with patch("agents._llm"), patch(
            "agents.create_agent", return_value=fake_agent
        ):
            result = job_search_node(state, runtime, {})
        self.assertEqual(result["job_results"], records)
        self.assertEqual(result["completed_workers"], ["JobSearcher"])

    def test_cover_letter_requires_explicit_selected_job(self):
        state = {"messages": [HumanMessage(content="write a cover letter")]}
        runtime = SimpleNamespace(context=GraphContext(llm_config={}))
        result = cover_letter_generator_node(state, runtime, {})
        self.assertIn("selected job is required", result["messages"][0].content)
        self.assertEqual(result["completed_workers"], ["CoverLetterGenerator"])

    def test_cover_letter_receives_explicit_selected_job_context(self):
        captured = {}

        def invoke(inputs, _config):
            captured["messages"] = inputs["messages"]
            return {"messages": [AIMessage(content="letter generated")]}

        state = {
            "messages": [HumanMessage(content="write a cover letter")],
            "selected_job": {"job_title": "Engineer", "company_name": "Acme"},
        }
        runtime = SimpleNamespace(context=GraphContext(llm_config={}))
        with patch("agents._llm"), patch(
            "agents.create_agent", return_value=SimpleNamespace(invoke=invoke)
        ):
            result = cover_letter_generator_node(state, runtime, {})
        self.assertIn("Acme", captured["messages"][-1].content)
        self.assertEqual(result["completed_workers"], ["CoverLetterGenerator"])


class TestStateContract(unittest.TestCase):
    def test_state_removes_redundant_and_ui_runtime_fields(self):
        annotations = AgentState.__annotations__
        self.assertNotIn("user_input", annotations)
        self.assertNotIn("next_step", annotations)
        self.assertNotIn("config", annotations)
        self.assertNotIn("callback", annotations)
        self.assertNotIn("last_worker", annotations)
        self.assertNotIn("cover_letter_inputs", annotations)
        self.assertIn("completed_workers", annotations)
        self.assertIn("selected_job", annotations)
        self.assertIn("job_results", annotations)


class TestCompiledGraphRouting(unittest.TestCase):
    def _run_graph(self, routes):
        remaining_routes = list(routes)
        chatbot_calls = []

        def supervisor_chain(_llm):
            return SimpleNamespace(
                invoke=lambda _input, _config: SimpleNamespace(
                    next_action=remaining_routes.pop(0)
                )
            )

        def worker_agent(*, tools, **_kwargs):
            if "ResumeExtractor" in {tool.name for tool in tools}:
                worker_name = "ResumeAnalyzer"
            else:
                worker_name = "WebResearcher"

            return SimpleNamespace(
                invoke=lambda _input, _config: {
                    "messages": [AIMessage(content=f"{worker_name} result")]
                }
            )

        def finish_chain(_llm):
            return SimpleNamespace(
                invoke=lambda _input, _config: (
                    chatbot_calls.append("ChatBot")
                    or AIMessage(content="ChatBot result")
                )
            )

        with patch("agents._llm"), patch(
            "agents.get_supervisor_chain", side_effect=supervisor_chain
        ), patch("agents.create_agent", side_effect=worker_agent), patch(
            "agents.get_finish_chain", side_effect=finish_chain
        ):
            return define_graph().invoke(
                {"messages": [HumanMessage(content="perform the task")]},
                {"recursion_limit": 20},
                context=GraphContext(llm_config={}),
            ), chatbot_calls

    def test_worker_worker_finish_reaches_end(self):
        output, _chatbot_calls = self._run_graph(
            ["ResumeAnalyzer", "WebResearcher", "Finish"]
        )
        self.assertEqual(
            output["completed_workers"],
            ["ResumeAnalyzer", "WebResearcher"],
        )
        self.assertEqual(
            [message.content for message in output["messages"][-2:]],
            ["ResumeAnalyzer result", "WebResearcher result"],
        )

    def test_worker_chatbot_finish_reaches_end(self):
        output, chatbot_calls = self._run_graph(
            ["ResumeAnalyzer", "ChatBot", "ChatBot"]
        )
        self.assertEqual(
            output["completed_workers"],
            ["ResumeAnalyzer", "ChatBot"],
        )
        self.assertEqual(chatbot_calls, ["ChatBot"])

    def test_chatbot_cannot_repeat(self):
        output, chatbot_calls = self._run_graph(["ChatBot", "ChatBot"])
        self.assertEqual(output["completed_workers"], ["ChatBot"])
        self.assertEqual(chatbot_calls, ["ChatBot"])


if __name__ == "__main__":
    unittest.main()
