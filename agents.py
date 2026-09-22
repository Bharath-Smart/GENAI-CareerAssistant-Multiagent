import json
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, MessagesState, END
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired
from dotenv import load_dotenv
from chains import get_finish_chain, get_supervisor_chain
from tools import (
    linkedin_job_search,
    extract_resume,
    generate_letter_for_specific_job,
    get_google_search_results,
    save_cover_letter_for_specific_job,
    scrape_website,
)
from prompts import (
    get_search_agent_prompt_template,
    get_analyzer_agent_prompt_template,
    get_researcher_agent_prompt_template,
    get_generator_agent_prompt_template,
)

load_dotenv()


@dataclass
class GraphContext:
    llm_config: dict


WORKERS = {
    "ResumeAnalyzer",
    "CoverLetterGenerator",
    "JobSearcher",
    "WebResearcher",
    "ChatBot",
}


def _resolve_next_action(next_action: str, state: dict) -> str:
    completed_workers = set(state.get("completed_workers", []))
    if next_action not in WORKERS and next_action != "Finish":
        next_action = "Finish"
    if next_action == "Finish" and not completed_workers:
        return "ChatBot"
    if "ChatBot" in completed_workers:
        return "Finish"
    if next_action in completed_workers:
        return "ChatBot"
    return next_action


def _mark_worker_completed(state: dict, worker_name: str) -> list[str]:
    completed_workers = list(state.get("completed_workers", []))
    if worker_name not in completed_workers:
        completed_workers.append(worker_name)
    return completed_workers


def _write_agent_name(config: RunnableConfig, name: str) -> None:
    callbacks = config.get("callbacks", [])
    if hasattr(callbacks, "handlers"):
        callbacks = callbacks.handlers
    if isinstance(callbacks, list):
        for callback in callbacks:
            if hasattr(callback, "write_agent_name"):
                callback.write_agent_name(name)


def _llm(runtime: Runtime[GraphContext]):
    return init_chat_model(**runtime.context.llm_config)


def supervisor_node(
    state: "AgentState",
    runtime: Runtime[GraphContext],
    config: RunnableConfig,
):
    """
    The supervisor node is the main node in the graph. It is responsible for routing to the correct agent.
    """
    chat_history = state["messages"]
    if not chat_history:
        raise ValueError("The graph requires the current user message in messages.")
    llm = _llm(runtime)
    supervisor_chain = get_supervisor_chain(llm)
    output = supervisor_chain.invoke({"messages": chat_history}, config)
    next_action = _resolve_next_action(output.next_action, state)
    destination = END if next_action == "Finish" else next_action
    return Command(goto=destination)


def job_search_node(
    state: "AgentState",
    runtime: Runtime[GraphContext],
    config: RunnableConfig,
):
    """
    This Node is responsible for searching for jobs from linkedin or any other job search engine.
    Tools: Job Search Tool
    """
    llm = _llm(runtime)
    search_agent = create_agent(
        model=llm, tools=[linkedin_job_search], system_prompt=get_search_agent_prompt_template()
    )
    _write_agent_name(config, "JobSearcher Agent 💼")
    output = search_agent.invoke(
        {"messages": state["messages"]}, config
    )
    result_message = output["messages"][-1]
    job_results = []
    for message in output["messages"]:
        if isinstance(message, ToolMessage) and message.name == "JobSearchTool":
            content = message.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = []
            if isinstance(content, list):
                job_results = content
    return {
        "messages": [
            AIMessage(content=result_message.content, name="JobSearcher")
        ],
        "job_results": job_results,
        "completed_workers": _mark_worker_completed(state, "JobSearcher"),
    }


def resume_analyzer_node(
    state: "AgentState",
    runtime: Runtime[GraphContext],
    config: RunnableConfig,
):
    """
    Resume analyzer node will analyze the resume and return the output.
    Tools: Resume Extractor
    """
    llm = _llm(runtime)
    analyzer_agent = create_agent(
        model=llm, tools=[extract_resume], system_prompt=get_analyzer_agent_prompt_template()
    )
    _write_agent_name(config, "ResumeAnalyzer Agent 📄")
    output = analyzer_agent.invoke(
        {"messages": state["messages"]}, config
    )
    result_message = output["messages"][-1]
    return {
        "messages": [
            AIMessage(content=result_message.content, name="ResumeAnalyzer")
        ],
        "resume_analysis": result_message.content,
        "completed_workers": _mark_worker_completed(state, "ResumeAnalyzer"),
    }


def cover_letter_generator_node(
    state: "AgentState",
    runtime: Runtime[GraphContext],
    config: RunnableConfig,
):
    """
    Node which handles the generation of cover letters.
    Tools: Cover Letter Generator, Cover Letter Saver
    """
    selected_job = state.get("selected_job")
    if not selected_job:
        return {
            "messages": [
                AIMessage(
                    content="A selected job is required before generating a cover letter.",
                    name="CoverLetterGenerator",
                )
            ],
            "completed_workers": _mark_worker_completed(
                state, "CoverLetterGenerator"
            ),
        }

    llm = _llm(runtime)
    generator_agent = create_agent(
        model=llm,
        tools=[
            generate_letter_for_specific_job,
            save_cover_letter_for_specific_job,
            extract_resume,
        ],
        system_prompt=get_generator_agent_prompt_template(),
    )

    _write_agent_name(config, "CoverLetterGenerator Agent ✍️")
    selected_job_message = SystemMessage(
        content=(
            "Use this explicitly selected job data for the cover letter. "
            f"Do not infer or replace it:\n{json.dumps(selected_job)}"
        )
    )
    output = generator_agent.invoke(
        {"messages": [*state["messages"], selected_job_message]}, config
    )
    result_message = output["messages"][-1]
    return {
        "messages": [
            AIMessage(content=result_message.content, name="CoverLetterGenerator")
        ],
        "completed_workers": _mark_worker_completed(
            state, "CoverLetterGenerator"
        ),
    }


def web_research_node(
    state: "AgentState",
    runtime: Runtime[GraphContext],
    config: RunnableConfig,
):
    """
    Node which handles the web research.
    Tools: Google Search, Web Scraper
    """
    llm = _llm(runtime)
    research_agent = create_agent(
        model=llm,
        tools=[get_google_search_results, scrape_website],
        system_prompt=get_researcher_agent_prompt_template(),
    )
    _write_agent_name(config, "WebResearcher Agent 🔍")
    output = research_agent.invoke(
        {"messages": state["messages"]}, config
    )
    result_message = output["messages"][-1]
    return {
        "messages": [
            AIMessage(content=result_message.content, name="WebResearcher")
        ],
        "completed_workers": _mark_worker_completed(state, "WebResearcher"),
    }


def chatbot_node(
    state: "AgentState",
    runtime: Runtime[GraphContext],
    config: RunnableConfig,
):
    llm = _llm(runtime)
    finish_chain = get_finish_chain(llm)
    _write_agent_name(config, "ChatBot Agent 🤖")
    output = finish_chain.invoke({"messages": state["messages"]}, config)
    return {
        "messages": [AIMessage(content=output.content, name="ChatBot")],
        "completed_workers": _mark_worker_completed(state, "ChatBot"),
    }


def define_graph():
    """
    Defines and returns a graph representing the workflow of job search agent.
    Returns:
        graph (StateGraph): The compiled graph representing the workflow.
    """
    workflow = StateGraph(AgentState, context_schema=GraphContext)
    workflow.add_node("ResumeAnalyzer", resume_analyzer_node)
    workflow.add_node("JobSearcher", job_search_node)
    workflow.add_node("CoverLetterGenerator", cover_letter_generator_node)
    workflow.add_node("Supervisor", supervisor_node, destinations=tuple(WORKERS))
    workflow.add_node("WebResearcher", web_research_node)
    workflow.add_node("ChatBot", chatbot_node)

    members = [
        "ResumeAnalyzer",
        "CoverLetterGenerator",
        "JobSearcher",
        "WebResearcher",
        "ChatBot",
    ]
    workflow.set_entry_point("Supervisor")

    for member in members:
        # We want our workers to ALWAYS "report back" to the supervisor when done
        workflow.add_edge(member, "Supervisor")

    graph = workflow.compile()
    return graph


# The agent state is the input to each node in the graph. Subclassing
# MessagesState gives `messages` the standard `add_messages` reducer (append
# by default, replace-by-id for updates), so nodes return partial updates
# (e.g. {"messages": [new_message]}) instead of mutating and returning the
# full state dict each time.
class AgentState(MessagesState):
    completed_workers: NotRequired[list[str]]
    resume_analysis: NotRequired[str]
    job_results: NotRequired[list[dict]]
    selected_job: NotRequired[dict]
