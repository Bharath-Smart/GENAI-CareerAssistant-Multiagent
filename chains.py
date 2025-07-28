"""
chains.py

Defines LangChain-based agent orchestration logic using prompt templates and structured routing.
Includes chains for:
- Supervising agent decisions
- Handling task routing across multiple agents
- Finalizing conversations when tasks are complete
"""

# -------------------- IMPORTS --------------------
# Standard Library
from typing import List

# LangChain / LangGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

# Internal Modules
from members import get_team_members_details
from prompts import get_supervisor_prompt_template, get_finish_step_prompt
from schemas import RouteSchema

# -------------------- SUPERVISOR CHAIN --------------------
def get_supervisor_chain(llm: BaseChatModel) -> Runnable:
    """
    Constructs the supervisor decision-making chain.
    """

    team_members = get_team_members_details()

    formatted_member_list = "\n\n".join(
        [f"**{i+1} {member['name']}**\nRole: {member['description']}" for i, member in enumerate(team_members)]
    )

    options = [member["name"] for member in team_members]
    system_prompt = get_supervisor_prompt_template()

    routing_instruction = """
    Few steps to follow:
    - Don't overcomplicate the conversation.
    - If the user asked something to search on web then get the information and show it.
    - If the user asked to analyze resume then just analyze it, don't be oversmart and do something else.
    - Don't call chatbot agent if user is not asking from the above conversation.

    Penalty point will be given if you are not following the above steps.
    Given the conversation above, who should act next?
    Or should we FINISH? Select one of: {options}.
    Do only what is asked, and do not deviate from the instructions.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", routing_instruction),
        ]
    ).partial(options=str(options), members=formatted_member_list)

    supervisor_chain = prompt | llm.with_structured_output(RouteSchema)
    return supervisor_chain


# -------------------- FINISH CHAIN --------------------
def get_finish_chain(llm: BaseChatModel) -> Runnable:
    """
    Chain that generates the final response if the supervisor decides to finish.
    """
    system_prompt = get_finish_step_prompt()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("system", system_prompt),
        ]
    )

    finish_chain = prompt | llm
    return finish_chain
