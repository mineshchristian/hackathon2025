from api.api_client import APIClient
import streamlit as st
from os import getenv
from dotenv import load_dotenv
from config.settings import MESSAGE_HISTORY_KEY, ADK_SESSION_KEY
load_dotenv()

client : APIClient = None

service_url = getenv("SERVICE_URL", "http://127.0.0.1:8181")
client = APIClient(base_url=service_url, timeout=30)

def setup_ui():
    """
    Sets up and runs the Streamlit web application for the ADK chat assistant.
    """
    st.set_page_config(page_title="Mercurious Chat Assistant", layout="wide") 
    st.title("👋Chat Assistant (Mercurious)")
    st.markdown("[Developed by team Mercurious for CCIBT Hackathon 2025]") 
    st.divider() 

def setup_session():
    if ADK_SESSION_KEY not in st.session_state:
        session_id = client.get("createsession")["message"]
        st.session_state[ADK_SESSION_KEY] = session_id
        st.markdown(session_id) 

def get_response( user_input: str) -> str:
    params = {
        "session_id": st.session_state[ADK_SESSION_KEY],
        "user_input": user_input
    }
    try:
        response = client.post("query", json=params)
        return response 
    except Exception as e:
        st.error(f"POST request failed: {e}")
        return {"message": f"Error: Unable to get response from the server. {e}"}


def run_app():
   
    setup_ui()
    setup_session()


    with st.sidebar:
        st.title("Setup")
        # Radio buttons with values 1, 2 and 3 stored in session state at key 'selected_level'
        selected_level = st.radio("Choose a level", (1, 2, 3), index=0, key="selected_level")
        st.caption(f"Selected level: {selected_level}")

    st.subheader("Provide a country name for travel planning!") # Subheading for the chat section.
    # show currently chosen radio value in the main area so users can see it outside the sidebar
    #st.markdown(f"**Current selection:** {st.session_state.get('selected_level', 'Not selected')}")
    
    # Initialize chat message history in Streamlit's session state if it doesn't exist.
    if MESSAGE_HISTORY_KEY not in st.session_state:
        st.session_state[MESSAGE_HISTORY_KEY] = []

    # Display existing chat messages from the session state.
    for message in st.session_state[MESSAGE_HISTORY_KEY]:
        with st.chat_message(message["role"]): # Use Streamlit's chat message container for styling.
            st.markdown(message["content"])

    # Handle new user input.
    if prompt := st.chat_input("Ask for a greeting (e.g., 'greet me'), or just chat...", accept_file="multiple"):
    #if prompt := st.chat_input("Ask for a greeting (e.g., 'greet me'), or just chat..."):
        # Append user's message to history and display it.
        inputtext = prompt.text
        st.session_state[MESSAGE_HISTORY_KEY].append({"role": "user", "content": inputtext})
        with st.chat_message("user"):
            st.markdown(inputtext)
        # Process the user's message with the ADK agent and display the response.
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # Create an empty placeholder to update with the assistant's response.
            with st.spinner("Assistant is thinking..."): # Show a spinner while the agent processes the request.
                agent_response = get_response(inputtext) # Call the synchronous ADK runner.
                message_placeholder.markdown(agent_response["message"]) # Update the placeholder with the final response.
        
        # Append assistant's response to history.
        st.session_state[MESSAGE_HISTORY_KEY].append({"role": "assistant", "content": agent_response["message"]})

run_app()




