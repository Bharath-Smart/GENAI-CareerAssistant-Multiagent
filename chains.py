from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel

from members import get_team_members_details
from prompts import get_supervisor_prompt_template
from schemas import RouteOutput


def get_supervisor_chain(llm: BaseChatModel):
    """
    Returns a supervisor chain that manages a conversation between workers.

    The supervisor chain is responsible for managing a conversation between a group
    of workers. It prompts the supervisor to select the next worker to act, and
    each worker performs a task and responds with their results and status. The
    conversation continues until the supervisor decides to finish.

    Returns:
        supervisor_chain: A chain of prompts and functions that handle the conversation
                          between the supervisor and workers.
    """

    team_members = get_team_members_details()

    formatted_members_string = "\n".join(
        f"- {member['name']}: {member['description']}" for member in team_members
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_supervisor_prompt_template()),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ).partial(members=formatted_members_string)

    supervisor_chain = prompt | llm.with_structured_output(RouteOutput)

    return supervisor_chain
