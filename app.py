import os
import re
import tempfile
import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from dotenv import load_dotenv
from streamlit.delta_generator import DeltaGenerator
from langchain.messages import HumanMessage
from custom_callback_handler import CustomStreamlitCallbackHandler
from agents import GraphContext, define_graph
from data_loader import load_resume

load_dotenv()

# Load configuration from .env
os.environ["LINKEDIN_EMAIL"] = os.getenv("LINKEDIN_EMAIL", "")
os.environ["LINKEDIN_PASS"] = os.getenv("LINKEDIN_PASS", "")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")
os.environ["FIRECRAWL_API_KEY"] = os.getenv("FIRECRAWL_API_KEY", "")
os.environ["LINKEDIN_SEARCH"] = os.getenv("LINKEDIN_JOB_SEARCH", "")

# Page configuration
st.set_page_config(layout="wide")
st.title("GenAI Career Assistant - 👨‍💼")

streamlit_analytics.start_tracking()

# Setup directories and paths
temp_dir = "temp"

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Sidebar - File Upload
uploaded_document = st.sidebar.file_uploader("Upload Your Resume", type="pdf")

resume_path = os.path.join(temp_dir, "resume.pdf")
if uploaded_document is not None:
    candidate_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=temp_dir, suffix=".pdf", delete=False
        ) as candidate_file:
            candidate_file.write(uploaded_document.getvalue())
            candidate_path = candidate_file.name
        load_resume(candidate_path)
        os.replace(candidate_path, resume_path)
        st.session_state["active_resume_path"] = resume_path
        st.sidebar.success("Resume uploaded.")
    except (FileNotFoundError, ValueError) as exc:
        if candidate_path and os.path.exists(candidate_path):
            os.remove(candidate_path)
        st.sidebar.error(str(exc))

active_resume_path = st.session_state.get("active_resume_path")
if active_resume_path == resume_path and os.path.exists(resume_path):
    st.sidebar.info("Using the resume uploaded earlier in this session.")
elif not active_resume_path:
    st.sidebar.info("Upload a PDF resume to use resume analysis and cover letters.")

# Sidebar - Service Provider Selection
service_provider = st.sidebar.selectbox(
    "Service Provider",
    ("groq (llama-3.1-70b-versatile)", "openai"),
)
streamlit_analytics.stop_tracking()

if service_provider == "openai":
    # Sidebar - OpenAI Configuration
    api_key_openai = st.sidebar.text_input(
        "OpenAI API Key",
        st.session_state.get("OPENAI_API_KEY", ""),
        type="password",
    )
    model_openai = st.sidebar.selectbox(
        "OpenAI Model",
        ("gpt-4o-mini"),
    )
    settings = {
        "model": model_openai,
        "model_provider": "openai",
        "temperature": 0.3,
    }
    st.session_state["OPENAI_API_KEY"] = api_key_openai
    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]

else:
    # Toggle visibility for Groq API Key input
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

# Sidebar - Service Provider Note
st.sidebar.markdown(
    """
    **Note:** \n
    This multi-agent system works best with OpenAI. llama 3.1 may not always produce optimal results.\n
    Any key provided will not be stored or shared it will be used only for the current session.
    """
)

# Create the agent flow
flow_graph = define_graph()

# Initialize session state variables
if "chat_messages" not in st.session_state:
    # Full LangChain message history (list[BaseMessage]) that is fed into
    # and returned from the graph.
    st.session_state["chat_messages"] = []
if "pills_reset_counter" not in st.session_state:
    # st.pills has no imperative "clear selection" method; giving the widget
    # a new `key` on the next rerun is the documented way to reset it, so we
    # bump this counter after each submitted query.
    st.session_state["pills_reset_counter"] = 0
if "response_history" not in st.session_state:
    st.session_state["response_history"] = ["Hello! How can I assist you today?"]
if "user_query_history" not in st.session_state:
    st.session_state["user_query_history"] = ["Hi there! 👋"]

# Containers for the chat interface
conversation_container = st.container()
input_section = st.container()

# Define functions used above
def initialize_callback_handler(main_container: DeltaGenerator):
    # write_agent_name is called synchronously from within each graph node
    # (on the main script thread), so no ScriptRunContext propagation is
    # needed here.
    return CustomStreamlitCallbackHandler(parent_container=main_container)

def execute_chat_conversation(user_input, graph):
    callback_handler = initialize_callback_handler(st.container())
    try:
        output = graph.invoke(
            {
                "messages": st.session_state["chat_messages"] + [HumanMessage(content=user_input)],
            },
            {"callbacks": [callback_handler], "recursion_limit": 30},
            context=GraphContext(llm_config=settings),
        )
        messages_list = output.get("messages")
        message_output = messages_list[-1]
        st.session_state["chat_messages"] = messages_list

        return message_output.content
    except Exception as exc:
        return ":( Sorry, Some error occurred. Can you please try again?"


def query_requires_resume(user_input: str) -> bool:
    return bool(re.search(r"\b(resume|cv|cover letter)\b", user_input, re.IGNORECASE))


# Clear Chat functionality
if st.button("Clear Chat"):
    st.session_state["user_query_history"] = []
    st.session_state["response_history"] = []
    st.session_state["chat_messages"] = []
    st.rerun()  # Refresh the app to reflect the cleared chat

# for tracking the query.
streamlit_analytics.start_tracking()

# Display chat interface
with input_section:
    options = [
        "🔍 Identify top trends in the tech industry relevant to gen ai",
        "🌐 Find emerging technologies and their potential impact on job opportunities",
        "📝 Summarize my resume",
        "📈 Create a career path visualization based on my skills and interests from my resume",
        "💼 GenAI Jobs at Microsoft",
        "🌟 Job Search GenAI jobs in India.",
        "✉️ Analyze my resume and suggest a suitable job role and search for relevant job listings",
        "🧠 Generate a cover letter for my resume.",
    ]
    # st.pills auto-detects a leading emoji as an icon and returns the
    # remaining text as the selected value, so no manual icon stripping is
    # needed here.
    selected_query = st.pills(
        "Pick a question for query:",
        options,
        key=f"pills_{st.session_state['pills_reset_counter']}",
    )

    # Display text input form
    with st.form(key="query_form", clear_on_submit=True):
        user_input_query = st.text_input(
            "Query:",
            value=(selected_query if selected_query else "Detail analysis of latest layoff news India?"),
            placeholder="📝 Write your query or select from the above",
            key="input",
        )
        submit_query_button = st.form_submit_button(label="Send")

    if submit_query_button:
        if query_requires_resume(user_input_query) and not (
            active_resume_path == resume_path and os.path.isfile(resume_path)
        ):
            st.error("Please upload a resume before submitting a query.")

        elif service_provider == "openai" and not st.session_state["OPENAI_API_KEY"]:
            st.error("Please enter your OpenAI API key before submitting a query.")

        elif user_input_query:
            chat_output = execute_chat_conversation(user_input_query, flow_graph)
            st.session_state["user_query_history"].append(user_input_query)
            st.session_state["response_history"].append(chat_output)
            st.session_state["pills_reset_counter"] += 1

# Display chat history
if st.session_state["response_history"]:
    with conversation_container:
        for i in range(len(st.session_state["response_history"])):
            with st.chat_message("user"):
                st.write(st.session_state["user_query_history"][i])
            with st.chat_message("assistant"):
                st.write(st.session_state["response_history"][i])

streamlit_analytics.stop_tracking()
