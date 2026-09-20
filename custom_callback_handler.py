from langchain_community.callbacks import StreamlitCallbackHandler


def CustomStreamlitCallbackHandler(parent_container):
    callback_handler = StreamlitCallbackHandler(parent_container)

    def write_agent_name(name: str):
        parent_container.write(name)

    callback_handler.write_agent_name = write_agent_name

    return callback_handler