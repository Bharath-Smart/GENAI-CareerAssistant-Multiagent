import os
import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from dotenv import load_dotenv
from streamlit.delta_generator import DeltaGenerator
from langchain.messages import HumanMessage
from custom_callback_handler import CustomStreamlitCallbackHandler
from agents import define_graph
import shutil

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
st.markdown("[Connect with me on LinkedIn](https://www.linkedin.com/in/aman-varyani-885725181/)")

streamlit_analytics.start_tracking()

# Setup directories and paths
temp_dir = "temp"
dummy_resume_path = os.path.abspath("dummy_resume.pdf")

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Add dummy resume if it does not exist
if not os.path.exists(dummy_resume_path):
    default_resume_path = "path/to/your/dummy_resume.pdf"
    shutil.copy(default_resume_path, dummy_resume_path)

# Sidebar - File Upload
uploaded_document = st.sidebar.file_uploader("Upload Your Resume", type="pdf")

if not uploaded_document:
    uploaded_document = open(dummy_resume_path, "rb")
    st.sidebar.write("Using a dummy resume for demonstration purposes. ")
    st.sidebar.markdown(f"[View Dummy Resume]({'https://drive.google.com/file/d/1vTdtIPXEjqGyVgUgCO6HLiG9TSPcJ5eM/view?usp=sharing'})", unsafe_allow_html=True)
    
bytes_data = uploaded_document.read()

filepath = os.path.join(temp_dir, "resume.pdf")
with open(filepath, "wb") as f:
    f.write(bytes_data)

st.markdown("**Resume uploaded successfully!**")

# Sidebar - Service Provider Selection
service_provider = st.sidebar.selectbox(
    "Service Provider",
    ("groq (llama-3.1-70b-versatile)", "openai"),
)
streamlit_analytics.stop_tracking()

# Not to track the key
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
st.sidebar.markdown(
    """
    <div style="padding:10px 0;">
        If you like the project, give a 
        <a href="https://github.com/amanv1906/GENAI-CareerAssistant-Multiagent" target="_blank" style="text-decoration:none;">
            ⭐ on GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True,
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
                "user_input": user_input,
                "config": settings,
                "callback": callback_handler,
            },
            {"recursion_limit": 30},
        )
        messages_list = output.get("messages")
        message_output = messages_list[-1]
        st.session_state["chat_messages"] = messages_list

        return message_output.content
    except Exception as exc:
        return ":( Sorry, Some error occurred. Can you please try again?"

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
        if not uploaded_document:
            st.error("Please upload your resume before submitting a query.")

        elif service_provider == "openai" and not st.session_state["OPENAI_API_KEY"]:
            st.error("Please enter your OpenAI API key before submitting a query.")

        elif user_input_query:
            # Process the query as usual if resume is uploaded
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
