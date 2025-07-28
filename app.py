"""
app.py

Streamlit frontend for the GenAI Career Assistant. 
This file sets up the UI, handles environment variables, loads the multi-agent LangGraph, 
and manages interactions between the user and agents.

Key features:
- Resume upload and fallback to dummy resume
- Query options via UI pills and text input
- Multi-agent routing powered by LangGraph (ResumeAnalyzer, JobSearcher, etc.)
- Support for both OpenAI and Groq APIs
- Callback streaming and interaction history tracking

Run this file using `streamlit run app.py` to launch the assistant.
"""

# -------------------- IMPORTS --------------------
# Standard Library
import os
import shutil
import inspect
from typing import Callable, TypeVar

# Environment & Configuration
from dotenv import load_dotenv 

# Streamlit & UI Libraries
import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from streamlit_chat import message
from streamlit_pills import pills
from streamlit.runtime.scriptrunner import (
    add_script_run_ctx,
    get_script_run_ctx
)
from streamlit.delta_generator import DeltaGenerator

# LangChain / LangGraph
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Internal Modules
from agents import define_graph
from custom_callback_handler import CustomStreamlitCallbackHandler

# -------------------- ENVIRONMENT SETUP --------------------
load_dotenv()

# Use secrets or fallback to .env values
def set_env_var(key):
    os.environ[key] = st.secrets.get(key, os.getenv(key, ""))

env_keys = [
    "LINKEDIN_EMAIL", "LINKEDIN_PASS", "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "GROQ_API_KEY",
    "SERPER_API_KEY", "FIRECRAWL_API_KEY", "LINKEDIN_JOB_SEARCH"
]
for key in env_keys:
    set_env_var(key)

# -------------------- STREAMLIT PAGE SETUP --------------------
st.set_page_config(layout="wide")
st.title("GenAI Career Assistant - 👨‍💼")

streamlit_analytics.start_tracking()

# -------------------- FILE MANAGEMENT --------------------
# Setup directories and paths
temp_dir = "temp"
dummy_resume_path = os.path.abspath("dummy_resume.pdf")
ORIGINAL_DUMMY_LINK = "https://drive.google.com/file/d/1vTdtIPXEjqGyVgUgCO6HLiG9TSPcJ5eM/view?usp=sharing"

# Track whether original dummy is being used
using_original_dummy = True

# Add dummy resume if it does not exist i.e, you (developer) deleted it
if not os.path.exists(dummy_resume_path):
    DEFAULT_DUMMY_PATH = "path/to/your/dummy_resume.pdf"  # Use your own resume file (local path)
    CUSTOM_DUMMY_LINK = ""  # Optional: If using custom dummy resume, you may provide a viewable link
    
    for pdf in os.listdir():
        if pdf.endswith(".pdf") and pdf != "dummy_resume.pdf":
            shutil.copy(pdf, dummy_resume_path)  # Replace dummy with user's PDF
            os.remove(pdf)
            break
    else:
        shutil.copy(DEFAULT_DUMMY_PATH, dummy_resume_path)

# Sidebar - Resume Upload
uploaded_document = st.sidebar.file_uploader("Upload Your Resume", type="pdf")

# If not uploaded, use dummy
if not uploaded_document:
    uploaded_document = open(dummy_resume_path, "rb")
    st.sidebar.write("Using a dummy resume for demonstration purposes.")
    
    # Show appropriate link based on what dummy is being used
    if using_original_dummy:
        st.sidebar.markdown(f"[View Dummy Resume]({ORIGINAL_DUMMY_LINK})", unsafe_allow_html=True)
    elif CUSTOM_DUMMY_LINK:
        st.sidebar.markdown(f"[View Custom Resume]({CUSTOM_DUMMY_LINK})", unsafe_allow_html=True)

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)    

# Save uploaded or dummy resume to temp
filepath = os.path.join(temp_dir, "resume.pdf")
with open(filepath, "wb") as f:
    f.write(uploaded_document.read())

st.markdown("**Resume uploaded successfully!**")

# -------------------- MODEL CONFIGURATION --------------------
# Sidebar: Choose between OpenAI or Groq
service_provider = st.sidebar.selectbox(
    "Service Provider",
    ("groq (llama-3.1-70b-versatile)", "openai"),
)

streamlit_analytics.stop_tracking()  # For not tracking the key

if service_provider == "openai":
    # OpenAI API Key input (persisted using session_state)
    api_key_openai = st.sidebar.text_input(
        "OpenAI API Key",
        st.session_state.get("OPENAI_API_KEY", ""),
        type="password",
    )

    # Choose OpenAI model
    model_openai = st.sidebar.selectbox(
        "OpenAI Model",
        ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"),
    )

    # Store key in session and env for downstream use
    st.session_state["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"] = api_key_openai

    settings = {
        "model": model_openai,
        "model_provider": "openai",
        "temperature": 0.3,
    }

else:
    # Toggle: Reveal Groq API key input field
    if "groq_key_visible" not in st.session_state:
        st.session_state["groq_key_visible"] = False

    if st.sidebar.button("Enter Groq API Key (optional)"):
        st.session_state["groq_key_visible"] = True

    if st.session_state["groq_key_visible"]:
        api_key_groq = st.sidebar.text_input("Groq API Key", type="password")
        st.session_state["GROQ_API_KEY"] = api_key_groq
        os.environ["GROQ_API_KEY"] = api_key_groq

    settings = {
        "model": "llama-3.1-70b-versatile",
        "model_provider": "groq",
        "temperature": 0.3,
    }

# Info Note: Best experience with OpenAI
st.sidebar.markdown(
    """
    **Note:** \n
    This multi-agent system works best with OpenAI.\n
    llama 3.1 may not always produce optimal results.\n
    API keys are not stored — used only in your current session.
    """
)

# -------------------- SETUP AGENT GRAPH + SESSION STATE --------------------
flow_graph = define_graph()
message_history = StreamlitChatMessageHistory()

# Initialize session state variables
st.session_state.setdefault("active_option_index", None)
st.session_state.setdefault("interaction_history", [])
st.session_state.setdefault("response_history", ["Hello! How can I assist you today?"])
st.session_state.setdefault("user_query_history", ["Hi there! 👋"])

# Containers for the chat interface
conversation_container = st.container()
input_section = st.container()

# -------------------- CALLBACK WRAPPER --------------------
def initialize_callback_handler(main_container: DeltaGenerator):
    V = TypeVar("V")

    def wrap_function(func: Callable[..., V]) -> Callable[..., V]:
        ctx = get_script_run_ctx()

        def wrapped(*args, **kwargs):
            add_script_run_ctx(ctx)
            return func(*args, **kwargs)
        
        return wrapped

    streamlit_callback_instance = CustomStreamlitCallbackHandler(
        parent_container=main_container
    )
    
    for method_name, method in inspect.getmembers(
        streamlit_callback_instance, predicate=inspect.ismethod
    ):
        setattr(streamlit_callback_instance, method_name, wrap_function(method))

    return streamlit_callback_instance

# -------------------- QUERY HANDLER --------------------
def execute_chat_conversation(user_input, graph):
    callback = initialize_callback_handler(st.container())

    try:
        result = graph.invoke(
            {
                "messages": list(message_history.messages) + [user_input],
                "user_input": user_input,
                "config": settings,
                "callback": callback,
            },
            {"recursion_limit": 30}
        )
        messages_list = result.get("messages")
        message_output = messages_list[-1]
        message_history.clear()
        message_history.add_messages(messages_list)
    except Exception:
        return ":( Sorry, Some error occurred. Can you please try again?"
    
    return message_output.content

# -------------------- CHAT CLEAR BUTTON --------------------
if st.button("Clear Chat"):
    st.session_state["user_query_history"] = []
    st.session_state["response_history"] = []
    message_history.clear()
    st.rerun() # Refresh the app to reflect the cleared chat

# -------------------- CHAT INPUT SECTION --------------------
streamlit_analytics.start_tracking() # For tracking the query

# Display chat interface
with input_section:
    options = [
        "Identify top trends in the tech industry relevant to gen ai",
        "Find emerging technologies and their potential impact on job opportunities",
        "Summarize my resume",
        "Create a career path visualization based on my skills and interests from my resume",
        "GenAI Jobs at Microsoft",
        "Job Search GenAI jobs in India.",
        "Analyze my resume and suggest a suitable job role and search for relevant job listings",
        "Generate a cover letter for my resume."
    ]
    icons = ["🔍", "🌐", "📝", "📈", "💼", "🌟", "✉️", "🧠"]

    selected_query = pills(
        "Pick a question for query:",
        options,
        clearable=None,
        icons=icons,
        index=st.session_state["active_option_index"],
        key="pills",
    )

    if selected_query:
        st.session_state["active_option_index"] = options.index(selected_query)

    # Display text input form
    with st.form(key="query_form", clear_on_submit=True):
        user_input_query = st.text_input(
            "Query:",
            value=selected_query or "Detail analysis of latest layoff news India?",
            placeholder="📝 Write your query or select from the above",
            key="input",
        )
        submit_query_button = st.form_submit_button(label="Send")

    if submit_query_button:
        if not uploaded_document:
            st.error("Please upload your resume before submitting a query.")
        elif service_provider == "openai" and not st.session_state["OPENAI_API_KEY"]:
            st.error("Please enter your OpenAI API key before submitting a query.")
        elif user_input_query:
            chat_output = execute_chat_conversation(user_input_query, flow_graph)
            st.session_state["user_query_history"].append(user_input_query)
            st.session_state["response_history"].append(chat_output)
            st.session_state["last_input"] = user_input_query
            st.session_state["active_option_index"] = None

# -------------------- CHAT DISPLAY SECTION --------------------
if st.session_state["response_history"]:
    with conversation_container:
        for i, response in enumerate(st.session_state["response_history"]):
            message(
                st.session_state["user_query_history"][i], 
                is_user=True, 
                key=f"{i}_user", 
                avatar_style="fun-emoji")
            message(response, key=str(i), avatar_style="bottts")

streamlit_analytics.stop_tracking()
