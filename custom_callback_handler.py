from langchain_core.callbacks.base import BaseCallbackHandler


class CustomStreamlitCallbackHandler(BaseCallbackHandler):
    """
    Minimal callback handler used to display the name of the currently
    active worker agent above its output in the Streamlit UI.

    Note: this intentionally does NOT wrap langchain_community's
    StreamlitCallbackHandler (the "LLM thought" expander UI). That
    integration depends on the deprecated `langchain_community` package,
    relies on `streamlit.external.langchain` internals that no longer exist
    in the current Streamlit version, and was observed at runtime to raise
    `RuntimeError('Current LLMThought is unexpectedly None!')` because our
    graph invokes multiple agents against the same callback/container across
    a single conversation turn, which the thought-tracking state machine
    doesn't support. `BaseCallbackHandler` (langchain_core, the current,
    stable, non-deprecated callback extension point) plus our own
    `write_agent_name` is all the application actually relies on.
    """

    def __init__(self, parent_container):
        self._parent_container = parent_container

    def write_agent_name(self, name: str):
        self._parent_container.write(name)
