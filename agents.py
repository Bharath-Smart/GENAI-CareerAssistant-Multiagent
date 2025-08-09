"""
agents.py

Defines the graph of LangGraph agents for a job search assistant system. Each node represents an agent (e.g., JobSearcher, ResumeAnalyzer),
and this module builds the full workflow pipeline for coordinating them via a Supervisor agent.
"""

# -------------------- IMPORTS --------------------
# Standard Library
from typing import Any, TypedDict
import asyncio

# Environment & Configuration
from dotenv import load_dotenv

# LangChain / LangGraph
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Internal Modules
from chains import get_finish_chain, get_supervisor_chain
from tools import (
    extract_resume,    
    job_search_tool,
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

# -------------------- ENVIRONMENT SETUP --------------------
load_dotenv()

# -------------------- AGENT STATE --------------------
class AgentState(TypedDict):
    user_input: str
    messages: list[BaseMessage]
    next_step: str
    config: dict
    callback: Any
    
# -------------------- AGENT CREATION --------------------
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    """
    Creates an agent using the specified LLM, tools, and system prompt.

    Args:
        llm (ChatOpenAI): The language model to power the agent.
        tools (list): List of LangChain tools available to the agent.
        system_prompt (str): The system-level instructions for behavior control.

    Returns:
        AgentExecutor: An executable agent instance.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

# -------------------- GRAPH NODES --------------------
def supervisor_node(state: AgentState) -> AgentState:
    """
    Supervisor node: Routes input to the next agent based on system decision.
    """
    chat_history = state.get("messages", [])
    llm = init_chat_model(**state["config"])
    supervisor_chain = get_supervisor_chain(llm)

    if not chat_history:
        chat_history.append(HumanMessage(state["user_input"]))

    output = supervisor_chain.invoke({"messages": chat_history})
    state["next_step"] = output.next_action
    state["messages"] = chat_history
    return state

def resume_analyzer_node(state: AgentState) -> AgentState:
    """
    Extracts & analyzes resume content and suggest suitable job role(s).
    Tools: Resume Extractor.
    """
    chat_history = state.get("messages", [])
    llm = init_chat_model(**state["config"])
    analyzer_agent = create_agent(llm, [extract_resume], get_analyzer_agent_prompt_template())
    state["callback"].write_agent_name("ResumeAnalyzer Agent 📄")

    output = analyzer_agent.invoke({"messages": chat_history}, {"callbacks": [state["callback"]]})
    chat_history.append(HumanMessage(content=output.get("output"), name="ResumeAnalyzer"))
    state["messages"] = chat_history
    return state

def job_search_node(state: AgentState) -> AgentState:
    """
    Handles job search using LinkedIn (or similar APIs).
    Tools: Job Search Tool.
    """
    chat_history = state.get("messages", [])
    llm = init_chat_model(**state["config"])
    search_agent = create_agent(llm, [job_search_tool], get_search_agent_prompt_template())
    state["callback"].write_agent_name("JobSearcher Agent 💼")

    # Convert async tool call to sync using asyncio.run
    output = asyncio.run(search_agent.ainvoke({"messages": chat_history}, {"callbacks": [state["callback"]]}))
    
    chat_history.append(HumanMessage(content=output.get("output"), name="JobSearcher"))
    state["messages"] = chat_history
    return state

def cover_letter_generator_node(state: AgentState) -> AgentState:
    """
    Generates personalized cover letters.
    Tools: Cover Letter Generator, Cover Letter Saver, Resume Extractor.
    """
    chat_history = state.get("messages", [])
    llm = init_chat_model(**state["config"])
    generator_agent = create_agent(
        llm,
        [
            extract_resume,                         # Step 1 (if resume not present)
            generate_letter_for_specific_job,       # Step 2 (generate content)
            save_cover_letter_for_specific_job      # Step 3 (store it)
        ],
        get_generator_agent_prompt_template(),
    )
    state["callback"].write_agent_name("CoverLetterGenerator Agent ✍️")

    output = generator_agent.invoke({"messages": chat_history}, {"callbacks": [state["callback"]]})
    chat_history.append(HumanMessage(content=output.get("output"), name="CoverLetterGenerator"))
    state["messages"] = chat_history
    return state

def web_research_node(state: AgentState) -> AgentState:
    """
    Performs online research via Google Search and Firecrawl.
    Tools: Web Search Tool, Web Scraper.
    """
    chat_history = state.get("messages", [])
    llm = init_chat_model(**state["config"])
    research_agent = create_agent(
        llm,
        [get_google_search_results, scrape_website],
        get_researcher_agent_prompt_template(),
    )
    state["callback"].write_agent_name("WebResearcher Agent 🔍")

    output = research_agent.invoke({"messages": chat_history}, {"callbacks": [state["callback"]]})
    chat_history.append(HumanMessage(content=output.get("output"), name="WebResearcher"))
    state["messages"] = chat_history
    return state

def chatbot_node(state: AgentState) -> AgentState:
    """
    Handles fallback/general chatbot conversation.
    Tools: Final fallback LLM (ChatBot Agent 🤖).
    """
    chat_history = state.get("messages", [])
    llm = init_chat_model(**state["config"])
    finish_chain = get_finish_chain(llm)
    state["callback"].write_agent_name("ChatBot Agent 🤖")

    output = finish_chain.invoke({"messages": chat_history})
    chat_history.append(AIMessage(content=output.content, name="ChatBot"))
    state["messages"] = chat_history
    return state

# -------------------- GRAPH CONSTRUCTION --------------------
def define_graph():
    """
    Constructs and compiles the full agent workflow as a LangGraph.

    Returns:
        graph (StateGraph): Compiled LangGraph object.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("ResumeAnalyzer", resume_analyzer_node)
    workflow.add_node("JobSearcher", job_search_node)
    workflow.add_node("CoverLetterGenerator", cover_letter_generator_node)
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("WebResearcher", web_research_node)
    workflow.add_node("ChatBot", chatbot_node)

    members = ["ResumeAnalyzer", "CoverLetterGenerator", "JobSearcher", "WebResearcher", "ChatBot"]
    workflow.set_entry_point("Supervisor")

    for member in members:
        workflow.add_edge(member, "Supervisor")

    conditional_map = {k: k for k in members}
    conditional_map["Finish"] = END
    workflow.add_conditional_edges("Supervisor", lambda x: x["next_step"], conditional_map)

    return workflow.compile()
