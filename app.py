import streamlit as st
import os
from core.agent import (
    ValuationState, initial_state, extract_info_node, 
    ask_next_question_node, summary_confirmation_node, 
    process_confirmation_node, calculate_node, 
    should_calculate, missing_slots
)
import datetime
import time

# Configure page
st.set_page_config(
    page_title="Property Valuation Assistant",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Main app styling */
    .stApp {
        background: #f8fafc;
    }
    
    .main .block-container {
        padding-top: 0;
        padding-bottom: 0;
        max-width: 800px;
    }
    
    /* Header */
    .chat-header {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px 12px 0 0;
        margin-bottom: 0;
        text-align: center;
        box-shadow: 0 2px 10px rgba(14, 165, 233, 0.2);
    }
    
    .chat-header h1 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 600;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 0 0 12px 12px;
        max-height: 450px;
        overflow-y: auto;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Messages */
    .message {
        margin: 0.75rem 0;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin-left: auto;
        max-width: 75%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3);
    }
    
    .bot-message {
        background: #f1f5f9;
        color: #334155;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin-right: auto;
        max-width: 75%;
        word-wrap: break-word;
        border-left: 3px solid #0ea5e9;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .typing-indicator {
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin-right: auto;
        max-width: 75%;
        word-wrap: break-word;
        border-left: 3px solid #0ea5e9;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .typing-dots {
        display: flex;
        gap: 0.25rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #0ea5e9;
        border-radius: 50%;
        animation: typing 1.5s infinite;
    }
    
    @keyframes typing {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    /* Input area */
    .input-section {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="stTextInput"] > div > div > input,
    .stTextInput input,
    input[type="text"] {
        border-radius: 24px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        background-color: #f8fafc !important;
        color: #334155 !important;
    }
    
    div[data-testid="stTextInput"] > div > div > input:focus,
    .stTextInput input:focus,
    input[type="text"]:focus {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1) !important;
        outline: none !important;
        background-color: white !important;
    }
    
    div[data-testid="stTextInput"] > div > div > input::placeholder,
    .stTextInput input::placeholder,
    input[type="text"]::placeholder {
        color: #94a3b8 !important;
        font-style: italic !important;
    }
    
    div[data-testid="stButton"] > button,
    .stButton button,
    button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 24px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3) !important;
    }
    
    div[data-testid="stButton"] > button:hover,
    .stButton button:hover,
    button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4) !important;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    }
    
    /* Reset button */
    .reset-btn {
        text-align: center;
        margin-top: 0.5rem;
    }
    
    .reset-btn button {
        background: #64748b !important;
        font-size: 0.8rem !important;
        padding: 0.5rem 1rem !important;
    }
    
    .reset-btn button:hover {
        background: #475569 !important;
    }
    
    /* Scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    .copy-btn {
        background: none;
        border: none;
        padding: 0;
        margin-left: 0.5rem;
        cursor: pointer;
    }
    
    .message-timestamp {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session():
    """Initialize session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = initial_state()
        st.session_state.agent_state = ask_next_question_node(st.session_state.agent_state)
        
        # Add initial bot message
        if st.session_state.agent_state.get("messages"):
            initial_msg = st.session_state.agent_state["messages"][-1]["content"]
            st.session_state.messages.append({"role": "assistant", "content": initial_msg})

def display_messages():
    """Display chat messages"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            timestamp = message.get("timestamp", "")
            st.markdown(f'''
            <div class="message">
                <div class="user-message">
                    {message["content"]}
                    <div class="message-timestamp">{timestamp}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            content = message["content"].replace("\n", "<br>")
            timestamp = message.get("timestamp", "")
            message_id = f"msg_{hash(content)}"
            st.markdown(f'''
            <div class="message">
                <div class="bot-message">
                    <span id="{message_id}">{content}</span>
                    <button class="copy-btn" onclick="copyToClipboard('{message_id}')">📋</button>
                    <div class="message-timestamp">{timestamp}</div>
                </div>
            </div>
            <script>
                function copyToClipboard(messageId) {{
                    const text = document.querySelector(`#${{messageId}}`).innerText;
                    navigator.clipboard.writeText(text);
                }}
            </script>
            ''', unsafe_allow_html=True)

def show_typing_indicator():
    """Show typing indicator"""
    st.markdown('''
    <div class="message">
        <div class="typing-indicator">
            <span>Assistant is typing</span>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def process_input(user_input):
    """Process user input through the agent"""
    # Add user message with timestamp
    timestamp = datetime.datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": timestamp})
    st.session_state.agent_state["messages"].append({"role": "user", "content": user_input})
    
    # Show typing indicator
    typing_placeholder = st.empty()
    with typing_placeholder.container():
        show_typing_indicator()
    
    # Small delay for UX
    time.sleep(0.5)
    
    # Process through agent
    if "_confirmation" in st.session_state.agent_state.get("asked", []):
        st.session_state.agent_state = process_confirmation_node(st.session_state.agent_state)
        
        if st.session_state.agent_state["slots"].get("_confirmed", False):
            st.session_state.agent_state = calculate_node(st.session_state.agent_state)
            st.session_state.agent_state["asked"] = []
            st.session_state.agent_state["slots"]["_confirmed"] = False
        elif st.session_state.agent_state["slots"].get("_confirmed") == False:
            remaining = missing_slots(st.session_state.agent_state.get("slots", {}))
            if remaining:
                st.session_state.agent_state = ask_next_question_node(st.session_state.agent_state)
    else:
        st.session_state.agent_state = extract_info_node(st.session_state.agent_state)
        result = should_calculate(st.session_state.agent_state)
        
        if result == "ASK":
            st.session_state.agent_state = ask_next_question_node(st.session_state.agent_state)
        elif result == "CONFIRM":
            st.session_state.agent_state = summary_confirmation_node(st.session_state.agent_state)
        elif result == "CALC":
            st.session_state.agent_state = calculate_node(st.session_state.agent_state)
            st.session_state.agent_state["asked"] = []
    
    # Clear typing indicator
    typing_placeholder.empty()
    
    # Add bot response with timestamp
    if st.session_state.agent_state.get("messages"):
        bot_response = st.session_state.agent_state["messages"][-1]["content"]
        timestamp = datetime.datetime.now().strftime("%H:%M")
        st.session_state.messages.append({"role": "assistant", "content": bot_response, "timestamp": timestamp})

def reset_chat():
    """Reset the chat session"""
    st.session_state.messages = []
    st.session_state.agent_state = initial_state()
    st.session_state.agent_state = ask_next_question_node(st.session_state.agent_state)
    
    if st.session_state.agent_state.get("messages"):
        initial_msg = st.session_state.agent_state["messages"][-1]["content"]
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})

def main():
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("🔑 Please set your GOOGLE_API_KEY environment variable")
        st.stop()
    
    # Initialize
    initialize_session()
    
    # Header
    st.markdown("""
    <div class="chat-header">
        <h1>🏠 Property Valuation Assistant</h1>
        <div class="status-badge">
            <div class="status-dot"></div>
            <span>Online & Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    display_messages()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Message",
                placeholder="Type your response here...",
                label_visibility="collapsed"
            )
        
        with col2:
            send_clicked = st.form_submit_button("Send", use_container_width=True)
        
        if send_clicked and user_input.strip():
            process_input(user_input.strip())
            st.rerun()
    
    # Reset button
    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    if st.button("🔄 Start New Valuation", key="reset"):
        reset_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
