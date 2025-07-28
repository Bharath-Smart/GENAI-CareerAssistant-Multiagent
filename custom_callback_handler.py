"""
custom_callback_handler.py

Defines a custom Streamlit callback handler for controlling how agent names
are displayed in the Streamlit UI during LangChain agent execution.
"""

# -------------------- IMPORTS --------------------
# Streamlit & UI Libraries
from streamlit.external.langchain.streamlit_callback_handler import StreamlitCallbackHandler

class CustomStreamlitCallbackHandler(StreamlitCallbackHandler):
    def write_agent_name(self, name: str):
        # Override default display format to use plain text output
        self._parent_container.write(name)
