from typing import Any
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, MessagesState, END
from dotenv import load_dotenv
from chains import get_finish_chain, get_supervisor_chain
from tools import (
    linkedin_job_search,
    ResumeExtractorTool,
    generate_letter_for_specific_job,
    get_google_search_results,
    save_cover_letter_for_specific_job,
    scrape_website,
)
from prompts import (
    get_search_agent_prompt_template,
    get_analyzer_agent_prompt_template,
    researcher_agent_prompt_template,
    get_generator_agent_prompt_template,
)

load_dotenv()


def supervisor_node(state):
    """
    The supervisor node is the main node in the graph. It is responsible for routing to the correct agent.
    """
    chat_history = state["messages"] or [HumanMessage(content=state["user_input"])]
    llm = init_chat_model(**state["config"])
    supervisor_chain = get_supervisor_chain(llm)
    output = supervisor_chain.invoke({"messages": chat_history})
    next_action = output.next_action

    # Guard against ending the workflow (Finish -> END) before any worker has
    # ever processed the current user turn. The last message is still the raw,
    # unprocessed HumanMessage (no `name`) that app.py just added for this turn
    # in that case, so Finish would terminate the graph with nothing but the
    # user's own message as output. ChatBot is the designated fallback worker
    # for producing an actual conversational reply, so route there instead.
    last_message = chat_history[-1]
    if (
        next_action == "Finish"
        and isinstance(last_message, HumanMessage)
        and not last_message.name
    ):
        next_action = "ChatBot"

    update = {"next_step": next_action}
    if not state["messages"]:
        # Graph was invoked with no `messages` at all (only `user_input`);
        # seed history with the synthesized HumanMessage above so it's part
        # of the persisted state going forward.
        update["messages"] = chat_history
    return update


def job_search_node(state):
    """
    This Node is responsible for searching for jobs from linkedin or any other job search engine.
    Tools: Job Search Tool
    """
    llm = init_chat_model(**state["config"])
    search_agent = create_agent(
        model=llm, tools=[linkedin_job_search], system_prompt=get_search_agent_prompt_template()
    )
    state["callback"].write_agent_name("JobSearcher Agent 💼")
    output = search_agent.invoke(
        {"messages": state["messages"]}, {"callbacks": [state["callback"]]}
    )
    result_message = output["messages"][-1]
    return {"messages": [HumanMessage(content=result_message.content, name="JobSearcher")]}


def resume_analyzer_node(state):
    """
    Resume analyzer node will analyze the resume and return the output.
    Tools: Resume Extractor
    """
    llm = init_chat_model(**state["config"])
    analyzer_agent = create_agent(
        model=llm, tools=[ResumeExtractorTool()], system_prompt=get_analyzer_agent_prompt_template()
    )
    state["callback"].write_agent_name("ResumeAnalyzer Agent 📄")
    output = analyzer_agent.invoke(
        {"messages": state["messages"]}, {"callbacks": [state["callback"]]}
    )
    result_message = output["messages"][-1]
    return {"messages": [HumanMessage(content=result_message.content, name="ResumeAnalyzer")]}


def cover_letter_generator_node(state):
    """
    Node which handles the generation of cover letters.
    Tools: Cover Letter Generator, Cover Letter Saver
    """
    llm = init_chat_model(**state["config"])
    generator_agent = create_agent(
        model=llm,
        tools=[
            generate_letter_for_specific_job,
            save_cover_letter_for_specific_job,
            ResumeExtractorTool(),
        ],
        system_prompt=get_generator_agent_prompt_template(),
    )

    state["callback"].write_agent_name("CoverLetterGenerator Agent ✍️")
    output = generator_agent.invoke(
        {"messages": state["messages"]}, {"callbacks": [state["callback"]]}
    )
    result_message = output["messages"][-1]
    return {
        "messages": [
            HumanMessage(content=result_message.content, name="CoverLetterGenerator")
        ]
    }


def web_research_node(state):
    """
    Node which handles the web research.
    Tools: Google Search, Web Scraper
    """
    llm = init_chat_model(**state["config"])
    research_agent = create_agent(
        model=llm,
        tools=[get_google_search_results, scrape_website],
        system_prompt=researcher_agent_prompt_template(),
    )
    state["callback"].write_agent_name("WebResearcher Agent 🔍")
    output = research_agent.invoke(
        {"messages": state["messages"]}, {"callbacks": [state["callback"]]}
    )
    result_message = output["messages"][-1]
    return {"messages": [HumanMessage(content=result_message.content, name="WebResearcher")]}


def chatbot_node(state):
    llm = init_chat_model(**state["config"])
    finish_chain = get_finish_chain(llm)
    state["callback"].write_agent_name("ChatBot Agent 🤖")
    output = finish_chain.invoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=output.content, name="ChatBot")]}


def define_graph():
    """
    Defines and returns a graph representing the workflow of job search agent.
    Returns:
        graph (StateGraph): The compiled graph representing the workflow.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("ResumeAnalyzer", resume_analyzer_node)
    workflow.add_node("JobSearcher", job_search_node)
    workflow.add_node("CoverLetterGenerator", cover_letter_generator_node)
    workflow.add_node("Supervisor", supervisor_node)
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

    conditional_map = {k: k for k in members}
    conditional_map["Finish"] = END

    workflow.add_conditional_edges(
        "Supervisor", lambda x: x["next_step"], conditional_map
    )

    graph = workflow.compile()
    return graph


# The agent state is the input to each node in the graph. Subclassing
# MessagesState gives `messages` the standard `add_messages` reducer (append
# by default, replace-by-id for updates), so nodes return partial updates
# (e.g. {"messages": [new_message]}) instead of mutating and returning the
# full state dict each time.
class AgentState(MessagesState):
    user_input: str
    next_step: str
    config: dict
    callback: Any
