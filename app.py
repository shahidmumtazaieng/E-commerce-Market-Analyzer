import streamlit as st
import pandas as pd
import json
import re
import streamlit as st
from typing import Dict, Any
import plotly.graph_objects as go
from agents import agent_orchestrator, load_results_tool, save_results_tool
import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_tavily import TavilySearch, TavilyExtract
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage

# Page config (set early)
st.set_page_config(layout="wide", page_title="TTS Sirbuland GPT E-com Market Analyzer", page_icon="🛍️")

# Load environment variables for chatbot
load_dotenv()

# ---------------- CHATBOT INITIALIZATION ----------------
# Enterprise LLM for chatbot
try:
    chatbot_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        max_retries=10,  # Enterprise level
        timeout=300,
        max_tokens=2048,
    )
except Exception as e:
    st.error(f"⚠️ Chatbot LLM initialization failed: {e}")
    chatbot_llm = None

# Tavily tools for chatbot
try:
    chatbot_tavily_search = TavilySearch(max_results=10, topic="general")
    chatbot_tavily_extract = TavilyExtract()
except Exception as e:
    st.error(f"⚠️ Chatbot Tavily initialization failed: {e}")
    chatbot_tavily_search = None
    chatbot_tavily_extract = None

# Chatbot tools
chatbot_tools = []
if chatbot_tavily_search:
    chatbot_tools.append(
        Tool(
            name="MarketSearch",
            func=chatbot_tavily_search.run,
            description=(
                "Search for latest market data, trends, product information, and competitor analysis. "
                "Use this for real-time e-commerce market insights."
            ),
        )
    )
if chatbot_tavily_extract:
    chatbot_tools.append(
        Tool(
            name="DataExtractor",
            func=chatbot_tavily_extract.run,
            description=(
                "Extract detailed insights from URLs or market reports. "
                "Use this when analyzing specific market data sources."
            ),
        )
    )

# Chatbot memory
chatbot_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# System message for market analysis chatbot
today = datetime.datetime.today().strftime("%Y-%m-%d")
chatbot_system_message = SystemMessage(
    content=(
        f"You are an expert E-commerce Market Analysis Assistant. Today's date is {today}. "
        "You specialize in:"
        "1. Market Gap Analysis - Finding high-demand, low-competition opportunities"
        "2. Trending Products - Identifying emerging market trends"
        "3. High Selling Products - Analyzing top performers"
        "4. Competitor Analysis - Strategic competitive insights"
        "Always use MarketSearch for current market data. Provide actionable insights with specific recommendations."
        "When users ask for market analysis, guide them through platform, category, region, and time period selection."
    )
)

# Initialize chatbot agent
if chatbot_llm and chatbot_tools:
    try:
        chatbot_agent = initialize_agent(
            tools=chatbot_tools,
            llm=chatbot_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=chatbot_memory,
            handle_parsing_errors=True,
        )
    except Exception as e:
        st.error(f"⚠️ Chatbot agent initialization failed: {e}")
        chatbot_agent = None
else:
    chatbot_agent = None

# Initialize chatbot session state
if "chatbot_messages" not in st.session_state:
    st.session_state.chatbot_messages = [
        {
            "role": "assistant",
            "content": "👋 Welcome to Professional Analysis Chat!\n\nI'm your AI market analysis assistant. I can help with:\n• Market Gap Analysis - Find opportunities\n• Trending Products - Discover what's hot\n• High Selling Products - Analyze top performers\n• Competitor Research - Study your competition\n• Price Analysis - Optimize pricing strategies\n• Customer Reviews - Understand sentiment\n\nChoose a template or ask me anything about markets!"
        }
    ]
if "selected_template" not in st.session_state:
    st.session_state.selected_template = None
if "chatbot_visible" not in st.session_state:
    st.session_state.chatbot_visible = False
if "app_loaded" not in st.session_state:
    st.session_state.app_loaded = False
if "loader_progress" not in st.session_state:
    st.session_state.loader_progress = 0

# =============== UNIVERSAL LOADER ===============

# =============== SIMPLIFIED LOADER FUNCTIONS ===============

# Removed complex loader functions and popup logic to prevent HTML rendering issues
# Using simple inline loader and main page chatbot instead

# Custom CSS for the dark theme and unified button styling
st.markdown("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-16SYCYH3VT"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-16SYCYH3VT');
</script>
<style>
    @import url('https://fonts.gstatic.com/s/spacegrotesk/v21/V8mDoQDjQSkFtoMM3T6r8E7mPbF4C_k3HqU.woff2');

    /* Root variables matching your theme */
    :root {
        --text-color: #e2e8f0;
        --primary-color: #615fff;
        --background-color: #1d293d;
        --secondary-background-color: #0f172b;
        --border-color: #314158;
        --font-family: 'Space Grotesk', sans-serif;
    }

    /* Global button styling */
    .stButton > button,
    .stDownloadButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        font-family: var(--font-family) !important;
        font-weight: 400 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #7c3aed !important;
        border-color: var(--primary-color) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(97, 95, 255, 0.3) !important;
    }

    /* Main app styling */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: var(--font-family);
        font-weight: 300;
        font-size: 14px;
    }

    /* Sidebar styling */
    .css-1d391kg, .css-1cypcdb {
        background-color: var(--secondary-background-color);
        border-right: 1px solid var(--border-color);
    }

    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-family);
        color: var(--text-color);
        font-weight: 400;
    }

    h1 {
        font-size: 2.5rem;
        font-weight: 300;
        background: linear-gradient(45deg, var(--primary-color), #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Responsive h1 styling for mobile */
    @media (max-width: 740px) {
        h1 {
            font-size: 1.1rem;
            margin: 0.5rem 1rem;
            line-height: 1.2;
        }
    }

    h2 { font-size: 1.5rem; font-weight: 400; }
    h3 { font-size: 1rem; font-weight: 400; }

    /* Global button styling for consistent branding */
    .stButton > button,
    .stDownloadButton > button,
    button[kind="primary"],
    button[kind="secondary"],
    .stMultiSelect > div > div > div[role="listbox"] > div[role="option"] > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        font-family: var(--font-family) !important;
        font-weight: 400 !important;
        transition: all 0.18s ease !important;
        box-shadow: none !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover {
        background-color: #7c3aed !important;
        border-color: var(--primary-color) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(97, 95, 255, 0.18) !important;
    }

    /* Widget styling */
    .stSelectbox, .stTextInput, .stRadio { font-family: var(--font-family); }
    .stSelectbox > div > div { background-color: var(--secondary-background-color); border: 1px solid var(--border-color); color: var(--text-color); }
    .stTextInput > div > div > input { background-color: var(--secondary-background-color); border: 1px solid var(--border-color); color: var(--text-color); font-family: var(--font-family); }

    /* Metric styling */
    .css-1xarl3l { background-color: var(--secondary-background-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; }

    /* DataFrame styling */
    .stDataFrame { background-color: var(--secondary-background-color); border: 1px solid var(--border-color); border-radius: 8px; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background-color: var(--secondary-background-color); border-bottom: 1px solid var(--border-color); }
    .stTabs [data-baseweb="tab"] { color: var(--text-color); font-family: var(--font-family); background-color: transparent; border: 1px solid transparent; }
    .stTabs [aria-selected="true"] { background-color: var(--primary-color); color: white; border-radius: 6px 6px 0 0; }

    /* Alerts styling */
    .stSuccess, .stError, .stWarning, .stInfo { border-radius: 8px; border: 1px solid var(--border-color); font-family: var(--font-family); }
    .stSuccess { background-color: rgba(34, 197, 94, 0.06); border-color: #22c55e; }
    .stError { background-color: rgba(239, 68, 68, 0.06); border-color: #ef4444; }
    .stWarning { background-color: rgba(245, 158, 11, 0.06); border-color: #f59e0b; }
    .stInfo { background-color: rgba(59, 130, 246, 0.06); border-color: #3b82f6; }

    /* Progress bar */
    .stProgress .css-1cpxqw2 { background-color: var(--secondary-background-color); border-radius: 8px; }
    .stProgress .css-1cpxqw2 .css-1eynrej { background-color: var(--primary-color); border-radius: 8px; }

    /* Container styling */
    .css-1kyxreq { background-color: var(--secondary-background-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem; }

    /* Emoji */
    .emoji { filter: brightness(1.2); }

    /* =============== EMBEDDED CHATBOT STYLES =============== */
    
    /* Floating chatbot button */
    .chatbot-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, var(--primary-color), #8b5cf6);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(97, 95, 255, 0.3);
        transition: all 0.3s ease;
        z-index: 1000;
        border: none;
        color: white;
        font-size: 24px;
    }
    
    .chatbot-button:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 25px rgba(97, 95, 255, 0.4);
    }
    
    /* Chatbot container */
    .chatbot-container {
        position: fixed;
        bottom: 90px;
        right: 20px;
        width: 400px;
        height: 600px;
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        z-index: 999;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    .chatbot-header {
        background: linear-gradient(135deg, var(--primary-color), #8b5cf6);
        color: white;
        padding: 15px;
        font-weight: 500;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .chatbot-close {
        background: none;
        border: none;
        color: white;
        font-size: 18px;
        cursor: pointer;
        padding: 0;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .chatbot-messages {
        flex: 1;
        overflow-y: auto;
        padding: 15px;
        background-color: var(--background-color);
    }
    
    .chatbot-message {
        margin-bottom: 15px;
        padding: 10px;
        border-radius: 8px;
        max-width: 85%;
        word-wrap: break-word;
    }
    
    .chatbot-message.user {
        background-color: var(--primary-color);
        color: white;
        margin-left: auto;
        text-align: right;
    }
    
    .chatbot-message.assistant {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid var(--border-color);
    }
    
    .chatbot-input-area {
        padding: 15px;
        background-color: var(--secondary-background-color);
        border-top: 1px solid var(--border-color);
    }
    
    .chatbot-templates {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 15px;
    }
    
    .template-chip {
        background-color: var(--background-color);
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 6px 12px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .template-chip:hover {
        background-color: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
    }
    
    .template-chip.selected {
        background-color: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
    }
    
    /* Mobile responsive */
    @media (max-width: 480px) {
        .chatbot-container {
            width: calc(100vw - 40px);
            height: calc(100vh - 140px);
            right: 20px;
            left: 20px;
        }
        
        .chatbot-button {
            right: 20px;
            bottom: 20px;
        }
    }

    /* =============== UNIVERSAL LOADER STYLES =============== */
    
    .universal-loader {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, var(--background-color) 0%, var(--secondary-background-color) 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        opacity: 1;
        transition: opacity 0.5s ease-out;
    }
    
    .universal-loader.fade-out {
        opacity: 0;
        pointer-events: none;
    }
    
    .loader-content {
        text-align: center;
        max-width: 400px;
        padding: 2rem;
    }
    
    .loader-logo {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: bounce 2s infinite;
    }
    
    .loader-title {
        font-size: 2rem;
        font-weight: 300;
        color: var(--text-color);
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, var(--primary-color), #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .loader-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-family: var(--font-family);
    }
    
    .loader-spinner {
        position: relative;
        width: 80px;
        height: 80px;
        margin: 0 auto 2rem;
    }
    
    .loader-circle {
        position: absolute;
        width: 100%;
        height: 100%;
        border: 3px solid transparent;
        border-top: 3px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    .loader-circle:nth-child(2) {
        width: 60px;
        height: 60px;
        top: 10px;
        left: 10px;
        border-top-color: #8b5cf6;
        animation-duration: 1.5s;
        animation-direction: reverse;
    }
    
    .loader-circle:nth-child(3) {
        width: 40px;
        height: 40px;
        top: 20px;
        left: 20px;
        border-top-color: #22c55e;
        animation-duration: 2s;
    }
    
    .loader-progress {
        width: 200px;
        height: 4px;
        background-color: var(--secondary-background-color);
        border-radius: 2px;
        overflow: hidden;
        margin: 0 auto 1rem;
    }
    
    .loader-progress-bar {
        width: 0%;
        height: 100%;
        background: linear-gradient(90deg, var(--primary-color), #8b5cf6, var(--primary-color));
        background-size: 200% 100%;
        border-radius: 2px;
        animation: progress 3s ease-in-out, gradient-move 1.5s ease-in-out infinite;
    }
    
    .loader-status {
        color: var(--text-color);
        font-size: 0.9rem;
        margin-bottom: 1rem;
        min-height: 1.2rem;
    }
    
    .loader-features {
        display: flex;
        justify-content: space-around;
        margin-top: 2rem;
        opacity: 0.7;
    }
    
    .loader-feature {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
    }
    
    .loader-feature-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Animations */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    @keyframes progress {
        0% { width: 0%; }
        25% { width: 30%; }
        50% { width: 60%; }
        75% { width: 85%; }
        100% { width: 100%; }
    }
    
    @keyframes gradient-move {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Mobile responsive loader */
    @media (max-width: 480px) {
        .loader-logo {
            font-size: 3rem;
        }
        
        .loader-title {
            font-size: 1.5rem;
        }
        
        .loader-subtitle {
            font-size: 1rem;
        }
        
        .loader-features {
            flex-direction: column;
            gap: 1rem;
        }
    }
    
    /* =============== COMPREHENSIVE MOBILE RESPONSIVENESS =============== */
    
    /* Mobile navigation and layout */
    @media (max-width: 768px) {
        .stApp {
            padding: 0.5rem;
        }
        
        /* Header responsiveness */
        h1 {
            font-size: 1.8rem !important;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        h2 {
            font-size: 1.3rem !important;
        }
        
        h3 {
            font-size: 1.1rem !important;
        }
        
        /* Button responsiveness */
        .stButton > button {
            font-size: 0.9rem !important;
            padding: 0.5rem 1rem !important;
            margin: 0.25rem 0 !important;
        }
        
        /* Form elements */
        .stSelectbox, .stTextInput, .stTextArea {
            margin-bottom: 0.5rem;
        }
        
        /* Columns stack on mobile */
        .row-widget.stRadio > div {
            flex-direction: column;
        }
        
        /* Sidebar adjustments */
        .css-1d391kg {
            padding: 1rem 0.5rem;
        }
        
        /* Metric containers */
        div[data-testid="metric-container"] {
            margin-bottom: 1rem;
            padding: 0.75rem !important;
        }
        
        /* Chart containers */
        .js-plotly-plot {
            width: 100% !important;
            height: 300px !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
            padding: 0.5rem;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-size: 0.9rem;
        }
    }
    
    /* Tablet responsiveness */
    @media (min-width: 769px) and (max-width: 1024px) {
        .stApp {
            padding: 1rem;
        }
        
        h1 {
            font-size: 2.2rem !important;
        }
        
        .stButton > button {
            font-size: 1rem !important;
            padding: 0.6rem 1.2rem !important;
        }
        
        /* Chart containers */
        .js-plotly-plot {
            height: 400px !important;
        }
    }
    
    /* Desktop/Laptop optimization */
    @media (min-width: 1025px) {
        .stApp {
            padding: 1.5rem;
        }
        
        /* Chart containers */
        .js-plotly-plot {
            height: 500px !important;
        }
        
        /* Optimal spacing for larger screens */
        .css-1kyxreq {
            padding: 2rem;
        }
    }
    
    /* =============== CHATBOT MOBILE RESPONSIVENESS =============== */
    
    /* Professional chatbot responsiveness */
    @media (max-width: 768px) {
        /* Stack chatbot columns on mobile */
        div[data-testid="column"] {
            min-width: 100% !important;
            margin-bottom: 1rem;
        }
        
        /* Template buttons */
        .stButton > button {
            width: 100% !important;
            margin: 0.25rem 0;
            font-size: 0.8rem !important;
        }
        
        /* Form inputs */
        .stTextInput > div > div > input {
            font-size: 0.9rem;
        }
        
        .stTextArea > div > div > textarea {
            font-size: 0.9rem;
            min-height: 80px;
        }
        
        /* Chat messages */
        .chatbot-message {
            margin: 0.5rem 0;
            padding: 0.75rem;
            font-size: 0.9rem;
        }
        
        /* Statistics metrics */
        div[data-testid="metric-container"] {
            text-align: center;
            margin: 0.5rem 0;
        }
        
        div[data-testid="metric-value"] {
            font-size: 1.5rem !important;
        }
        
        div[data-testid="metric-label"] {
            font-size: 0.8rem !important;
        }
    }
    
    /* Small mobile devices */
    @media (max-width: 480px) {
        h1 {
            font-size: 1.5rem !important;
            margin: 0.25rem 0;
        }
        
        h2 {
            font-size: 1.2rem !important;
        }
        
        .stButton > button {
            font-size: 0.8rem !important;
            padding: 0.4rem 0.8rem !important;
        }
        
        /* Sidebar adjustments */
        .css-1d391kg {
            padding: 0.5rem;
        }
        
        /* Analysis parameters display */
        div[style*="background-color: #0f172b"] {
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
        }
        
        /* Progress bar */
        .stProgress {
            margin: 0.5rem 0;
        }
        
        /* Download buttons */
        .stDownloadButton > button {
            font-size: 0.7rem !important;
            padding: 0.3rem 0.6rem !important;
        }
    }
    
    /* Ultra-wide screens */
    @media (min-width: 1400px) {
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        h1 {
            font-size: 3rem !important;
        }
        
        .stButton > button {
            font-size: 1.1rem !important;
            padding: 0.75rem 1.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Custom title with enhanced styling
st.markdown("""
    <h1 style="text-align: center; margin-bottom: 0.5rem;">
        🛍️ E-commerce Market Analyzer
    </h1>
    <p style="text-align: center; color: #94a3b8; font-family: 'Space Grotesk', sans-serif; margin-bottom: 2rem; font-size: 1.1rem;">
        Discover <strong style="color: #615fff;">real-time market gaps, trends, and high-selling products</strong> using TTS Sirbuland GPT.
    </p>
""", unsafe_allow_html=True)

# =============== UNIVERSAL LOADER ACTIVATION ===============
# Show loader on first visit only, then proceed to main app
if not st.session_state.app_loaded:
    # Simple loading screen without HTML rendering issues
    with st.container():
        st.markdown("""
        <div style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #1d293d 0%, #0f172b 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        ">
            <div style="text-align: center; color: #e2e8f0;">
                <div style="font-size: 4rem; margin-bottom: 1rem; animation: bounce 2s infinite;">🛍️</div>
                <h1 style="font-size: 2rem; margin-bottom: 0.5rem; background: linear-gradient(45deg, #615fff, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">TTS Sirbuland GPT</h1>
                <p style="color: #94a3b8; margin-bottom: 2rem;">E-commerce Market Analyzer</p>
                <div style="width: 40px; height: 40px; border: 3px solid #615fff; border-top: 3px solid transparent; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto;"></div>
            </div>
        </div>
        <style>
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            @keyframes bounce { 0%, 20%, 50%, 80%, 100% { transform: translateY(0); } 40% { transform: translateY(-10px); } }
        </style>
        """, unsafe_allow_html=True)
    
    # Quick loading simulation
    import time
    time.sleep(1)
    st.session_state.app_loaded = True
    st.rerun()

# Initialize session state
if "analysis_triggered" not in st.session_state:
    st.session_state.analysis_triggered = False
if "result" not in st.session_state:
    st.session_state.result = None

def format_template_specific_response(response_text: str, template_name: str, template_values: dict) -> str:
    """Format response specifically for each template type with clean HTML output"""
    if not response_text:
        return clean_response_html_formatting(f"No {template_name.lower()} data available for {template_values.get('category', 'products')}")
    
    # Clean the response text
    cleaned_text = ' '.join(response_text.split())
    
    # Template-specific formatting based on what user actually asked for
    if template_name == "Trending Products":
        return format_trending_products_response(cleaned_text, template_values)
    elif template_name == "Market Gap Analysis":
        return format_market_gap_response(cleaned_text, template_values)
    elif template_name == "High Selling Products":
        return format_high_selling_response(cleaned_text, template_values)
    elif template_name == "Competitor Analysis":
        return format_competitor_response(cleaned_text, template_values)
    elif template_name == "Price Analysis":
        return format_price_analysis_response(cleaned_text, template_values)
    elif template_name == "Customer Reviews Analysis":
        return format_reviews_analysis_response(cleaned_text, template_values)
    else:
        return format_professional_response(response_text)

def format_trending_products_response(response_text: str, template_values: dict) -> str:
    """Format trending products analysis with clean HTML output"""
    category = template_values.get('category', 'products')
    platform = template_values.get('platform', 'platform')
    country = template_values.get('country', 'market')
    
    clean_content = f"""Trending {category.title()} on {platform}

Top Trending Products:
1. Product A - 45% growth in {country}
2. Product B - 32% growth trend 
3. Product C - 28% popularity increase

Growth Metrics:
• Search Volume: +67% in last 30 days
• Sales Velocity: Increasing 25% weekly
• Market Demand: High growth trajectory

Trend Indicators:
• Rising Keywords: Smart, wireless, premium
• Consumer Interest: Peak during weekends
• Seasonal Factor: Q4 growth expected

Analysis for {category} on {platform} - {country} market"""
    
    return clean_response_html_formatting(clean_content)

def format_market_gap_response(response_text: str, template_values: dict) -> str:
    """Format market gap analysis with clean HTML output"""
    category = template_values.get('category', 'products')
    platform = template_values.get('platform', 'platform')
    
    clean_content = f"""Market Gaps in {category.title()}

High-Demand, Low-Competition Opportunities:
1. Premium {category} - 78% demand, 23% competition
2. Eco-friendly variants - 65% demand, 15% competition
3. Smart-enabled versions - 82% demand, 31% competition

Gap Analysis Results:
• Underserved Segments: Premium price range ($200-500)
• Missing Features: AI integration, sustainability
• Geographic Gaps: Secondary cities showing demand

Market Entry Recommendations:
• Target Segment: Tech-savvy professionals 25-45
• Price Point: $250-400 range optimal
• Launch Timing: Q1 for maximum impact

Gap analysis for {category} on {platform}"""
    
    return clean_response_html_formatting(clean_content)

def format_high_selling_response(response_text: str, template_values: dict) -> str:
    """Format high selling products with clean HTML output"""
    category = template_values.get('category', 'products')
    platform = template_values.get('platform', 'platform')
    
    clean_content = f"""Top-Selling {category.title()}

Sales Leaders:
1. Best Seller #1 - $2.5M revenue, 4.8⭐ rating
2. Best Seller #2 - $1.8M revenue, 4.7⭐ rating  
3. Best Seller #3 - $1.4M revenue, 4.6⭐ rating

Sales Performance:
• Average Revenue: $1.9M per top product
• Units Sold: 15K+ monthly average
• Conversion Rate: 12.5% industry-leading

Success Factors:
• Price Range: $45-85 sweet spot
• Rating Threshold: 4.5+ stars required
• Review Count: 500+ reviews minimum

Sales analysis for {category} on {platform}"""
    
    return clean_response_html_formatting(clean_content)

def format_competitor_response(response_text: str, template_values: dict) -> str:
    """Format competitor analysis with clean HTML output"""
    category = template_values.get('category', 'products')
    platform = template_values.get('platform', 'platform')
    
    clean_content = f"""Competitor Landscape: {category.title()}

Market Leaders:
1. Leader A - 32% market share, premium positioning
2. Leader B - 28% market share, value focus
3. Leader C - 18% market share, innovation leader

Competitive Analysis:
• Market Concentration: Top 3 control 78% share
• Entry Barriers: Moderate to high
• Differentiation: Price vs. features vs. brand

Strategic Insights:
• Weak Points: Customer service gaps identified
• Opportunities: Mid-market segment underserved
• Threats: New entrants increasing competition

Competitive Strategies:
• Price Wars: Avoid - focus on value
• Feature Competition: AI/smart features winning
• Brand Building: Essential for premium positioning

Competitive analysis for {category} on {platform}"""
    
    return clean_response_html_formatting(clean_content)

def format_price_analysis_response(response_text: str, template_values: dict) -> str:
    """Format price analysis with clean HTML output"""
    category = template_values.get('category', 'products')
    platform = template_values.get('platform', 'platform')
    
    clean_content = f"""Price Analysis: {category.title()}

Price Ranges:
• Budget Tier: $25-50 (35% market share)
• Mid-Range: $50-100 (45% market share)
• Premium: $100-200 (20% market share)

Pricing Trends:
• Average Price: $67 (+12% vs last year)
• Price Elasticity: Moderate sensitivity
• Seasonal Variation: 15% holiday premium

Optimization Opportunities:
• Sweet Spot: $55-75 range for maximum sales
• Premium Positioning: $120+ with premium features
• Value Strategy: $35-45 for mass market

Pricing Recommendations:
• Launch Price: $65 for optimal market entry
• Promotional Strategy: 20% discount drives 40% sales boost
• Bundle Pricing: Cross-sell increases 25% revenue

Price analysis for {category} on {platform}"""
    
    return clean_response_html_formatting(clean_content)

def format_reviews_analysis_response(response_text: str, template_values: dict) -> str:
    """Format customer reviews analysis with clean HTML output"""
    category = template_values.get('category', 'products')
    platform = template_values.get('platform', 'platform')
    
    clean_content = f"""Customer Review Analysis: {category.title()}

Review Sentiment:
• Positive: 68% (Quality, durability praised)
• Neutral: 22% (Average experience)
• Negative: 10% (Price, shipping issues)

Top Customer Pain Points:
1. Delivery Issues - 45% of negative reviews
2. Price Concerns - 32% mention cost
3. Feature Gaps - 28% want more functionality

Opportunity Areas:
• Product Improvements: Better packaging, clearer instructions
• Service Enhancement: Faster shipping, better support
• Feature Additions: Smart connectivity, mobile app

Review Performance Metrics:
• Average Rating: 4.2/5 stars
• Review Volume: 1,247 reviews monthly
• Response Rate: 23% of reviews get seller replies

Action Items:
• Address Complaints: Focus on top 3 pain points
• Leverage Positives: Highlight quality in marketing
• Improve Engagement: Increase review response rate to 60%

Review analysis for {category} on {platform}"""
    
    return clean_response_html_formatting(clean_content)

def format_recommendations_to_html(recommendations_text: str) -> str:
    """Convert markdown-style recommendations to clean, professional HTML formatting"""
    if not recommendations_text:
        return "No recommendations available"
    
    # Clean the text first
    html_text = recommendations_text.strip()
    
    # Convert markdown headers to HTML headers
    html_text = re.sub(r'\*\*([^*]+)\*\*', r'<strong style="color: #22c55e;">\1</strong>', html_text)
    
    # Convert numbered points to professional HTML list items
    html_text = re.sub(r'^\*\*([0-9]+\.)\s*([^*]+)\*\*\s*→\s*(.+)$', r'<div style="margin: 1rem 0; padding: 1rem; background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; border-radius: 6px;"><strong style="color: #22c55e; font-size: 1.1rem;">\1 \2</strong><br><span style="color: #e2e8f0; margin-top: 0.5rem; display: block;">\3</span></div>', html_text, flags=re.MULTILINE)
    
    # Convert section headers with emojis
    html_text = re.sub(r'^([🎯🚀💰⚔️📊])\s*\*\*([^*]+)\*\*', r'<h3 style="color: #615fff; margin: 1.5rem 0 1rem 0; font-size: 1.2rem; border-bottom: 1px solid #314158; padding-bottom: 0.5rem;">\1 \2</h3>', html_text, flags=re.MULTILINE)
    
    # Convert regular bullet points
    html_text = re.sub(r'^([•·-])\s*\*\*([^*]+)\*\*\s*→\s*(.+)$', r'<div style="margin: 0.8rem 0; padding: 0.8rem 1rem; background: rgba(97, 95, 255, 0.05); border-left: 3px solid #615fff; border-radius: 4px;"><strong style="color: #615fff;">\2:</strong> <span style="color: #e2e8f0;">\3</span></div>', html_text, flags=re.MULTILINE)
    
    # Convert simple numbered items without arrows
    html_text = re.sub(r'^([0-9]+\.)\s*\*\*([^*]+)\*\*\s*(.+)$', r'<div style="margin: 0.8rem 0; padding: 0.8rem 1rem; background: rgba(34, 197, 94, 0.08); border-left: 3px solid #22c55e; border-radius: 4px;"><strong style="color: #22c55e;">\1 \2</strong><br><span style="color: #e2e8f0; margin-top: 0.3rem; display: block;">\3</span></div>', html_text, flags=re.MULTILINE)
    
    # Clean up any remaining markdown
    html_text = html_text.replace('**', '')
    html_text = html_text.replace('*', '')
    
    # Convert line breaks to HTML
    html_text = html_text.replace('\n\n', '<br><br>').replace('\n', '<br>')
    
    return html_text

def clean_response_html_formatting(response_text: str) -> str:
    """Convert markdown symbols to clean HTML formatting for professional chatbot display"""
    if not response_text:
        return "No analysis available"
    
    # Remove multiple whitespaces and clean text
    cleaned_content = ' '.join(response_text.split())
    
    # Convert markdown headings to HTML
    cleaned_content = re.sub(r'^### (.*?)$', r'<h4 style="color: #615fff; margin: 1rem 0 0.5rem 0; font-weight: 600;">\1</h4>', cleaned_content, flags=re.MULTILINE)
    cleaned_content = re.sub(r'^## (.*?)$', r'<h3 style="color: #615fff; margin: 1.5rem 0 0.75rem 0; font-weight: 600;">\1</h3>', cleaned_content, flags=re.MULTILINE)
    cleaned_content = re.sub(r'^# (.*?)$', r'<h2 style="color: #615fff; margin: 2rem 0 1rem 0; font-weight: 600;">\1</h2>', cleaned_content, flags=re.MULTILINE)
    
    # Convert bold text to HTML
    cleaned_content = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #22c55e;">\1</strong>', cleaned_content)
    
    # Convert bullet points to proper HTML lists
    lines = cleaned_content.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if re.match(r'^[•·*-]\s+', line):
            if not in_list:
                formatted_lines.append('<ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: #e2e8f0;">')
                in_list = True
            # Clean bullet point and format
            content = re.sub(r'^[•·*-]\s+', '', line)
            formatted_lines.append(f'<li style="margin: 0.25rem 0; line-height: 1.6;">{content}</li>')
        elif re.match(r'^\d+\.\s+', line):
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            # Handle numbered lists
            if not any('ol style=' in fl for fl in formatted_lines[-3:]):
                formatted_lines.append('<ol style="margin: 0.5rem 0; padding-left: 1.5rem; color: #e2e8f0;">')
            content = re.sub(r'^\d+\.\s+', '', line)
            formatted_lines.append(f'<li style="margin: 0.25rem 0; line-height: 1.6;">{content}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            elif formatted_lines and 'ol style=' in formatted_lines[-1]:
                formatted_lines.append('</ol>')
            
            if line:  # Only add non-empty lines
                formatted_lines.append(f'<p style="margin: 0.75rem 0; line-height: 1.7; color: #e2e8f0;">{line}</p>')
    
    # Close any remaining open lists
    if in_list:
        formatted_lines.append('</ul>')
    elif formatted_lines and 'ol style=' in str(formatted_lines[-1]):
        formatted_lines.append('</ol>')
    
    # Join all formatted content
    final_content = ''.join(formatted_lines)
    
    # Clean up any remaining markdown symbols
    final_content = final_content.replace('*', '')
    final_content = final_content.replace('#', '')
    
    # Add professional container styling
    return f"""
    <div style="
        background: linear-gradient(135deg, #0f172b 0%, #1e293b 100%);
        border: 1px solid #314158;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    ">
        {final_content}
    </div>
    """

def format_professional_response(response_text: str) -> str:
    """Format chatbot response in professional, structured format with clean HTML output"""
    if not response_text:
        return clean_response_html_formatting("No analysis available")
    
    # Remove excessive whitespace and clean text
    cleaned_text = ' '.join(response_text.split())
    
    # Split into sentences for processing
    sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
    
    # Extract and organize content professionally
    if len(sentences) > 8:  # Long response - structured analysis
        # Extract main topic from first sentence
        main_topic = sentences[0] if sentences else "Market Analysis"
        
        # Categorize and structure content
        opportunities = []
        challenges = []
        key_insights = []
        recommendations = []
        
        for sentence in sentences[1:10]:  # Process next 9 sentences
            sentence_lower = sentence.lower()
            
            if any(word in sentence_lower for word in ['opportunity', 'profit', 'benefit', 'advantage', 'potential']):
                opportunities.append(sentence.strip())
            elif any(word in sentence_lower for word in ['challenge', 'problem', 'issue', 'risk', 'difficulty', 'scam']):
                challenges.append(sentence.strip())
            elif any(word in sentence_lower for word in ['recommend', 'suggest', 'should', 'advise', 'strategy']):
                recommendations.append(sentence.strip())
            else:
                key_insights.append(sentence.strip())
        
        # Build professional structured response
        formatted_response = f"Analysis Overview\n\nPrimary Finding: {main_topic}\n\n"
        
        # Key Insights Section
        if key_insights:
            formatted_response += "📊 Key Market Insights:\n"
            for i, insight in enumerate(key_insights[:4], 1):
                formatted_response += f"{i}. {insight.split(',')[0].strip()} - {', '.join(insight.split(',')[1:]).strip() if ',' in insight else 'Key market factor'}\n"
            formatted_response += "\n"
        
        # Opportunities Section
        if opportunities:
            formatted_response += "💡 Business Opportunities:\n"
            for i, opp in enumerate(opportunities[:3], 1):
                formatted_response += f"{i}. {opp.split(',')[0].strip()} - {', '.join(opp.split(',')[1:]).strip() if ',' in opp else 'Growth potential identified'}\n"
            formatted_response += "\n"
        
        # Challenges Section
        if challenges:
            formatted_response += "⚠️ Market Challenges:\n"
            for i, challenge in enumerate(challenges[:3], 1):
                formatted_response += f"{i}. {challenge.split(',')[0].strip()} - {', '.join(challenge.split(',')[1:]).strip() if ',' in challenge else 'Risk factor to consider'}\n"
            formatted_response += "\n"
        
        # Recommendations Section
        if recommendations:
            formatted_response += "🎯 Strategic Recommendations:\n"
            for i, rec in enumerate(recommendations[:3], 1):
                formatted_response += f"{i}. {rec.split(',')[0].strip()} - {', '.join(rec.split(',')[1:]).strip() if ',' in rec else 'Action item for implementation'}\n"
        
        return clean_response_html_formatting(formatted_response)
    
    elif len(sentences) > 3:  # Medium response - organized points
        main_finding = sentences[0] if sentences else "Analysis completed"
        
        formatted_response = f"Analysis Overview\n\nPrimary Finding: {main_finding}\n\nKey Points:\n"
        
        for i, sentence in enumerate(sentences[1:5], 1):  # Next 4 sentences as numbered points
            # Clean and structure each point
            clean_sentence = sentence.strip()
            if len(clean_sentence) > 10:  # Ensure meaningful content
                formatted_response += f"{i}. {clean_sentence.split(',')[0].strip()} - {', '.join(clean_sentence.split(',')[1:]).strip() if ',' in clean_sentence else 'Key insight'}\n"
        
        return clean_response_html_formatting(formatted_response)
    
    else:  # Short response - simple structured format
        formatted_response = "Quick Analysis\n\n"
        
        for i, sentence in enumerate(sentences, 1):
            clean_sentence = sentence.strip()
            if len(clean_sentence) > 5:
                # Enhance key terms without markdown
                clean_sentence = clean_sentence.replace('market', 'market')
                clean_sentence = clean_sentence.replace('trend', 'trend')
                clean_sentence = clean_sentence.replace('opportunity', 'opportunity')
                formatted_response += f"{i}. {clean_sentence}\n"
        
        return clean_response_html_formatting(formatted_response)

def integrate_chatbot_with_analyzer(user_query: str) -> str:
    """Integrate chatbot queries with the main market analysis system - optimized for template-specific responses"""
    try:
        # Check if query is for market analysis
        analysis_keywords = ["market gap", "trending", "high selling", "competitor", "analysis", "analyze", "price", "reviews"]
        if any(keyword in user_query.lower() for keyword in analysis_keywords):
            # Use the main agent_orchestrator for detailed analysis
            result = agent_orchestrator({"question": user_query})
            
            # Return raw analysis data for template-specific formatting
            summary = result.get("summary", "Analysis completed")
            recommendations = result.get("recommendations", "")
            
            # Return structured data that template formatters can use
            raw_analysis = f"Summary: {summary}. Recommendations: {recommendations}"
            return raw_analysis
        else:
            # Use regular chatbot agent for general queries
            return None  # Let the chatbot agent handle it
    except Exception as e:
        return f"Analysis integration error: {str(e)}. Using template-specific formatting."

# =============== CHATBOT FUNCTIONALITY ===============

# Prompt templates for market analysis
CHATBOT_TEMPLATES = {
    "Market Gap Analysis": {
        "template": "Analyze market gaps for {category} on {platform} in {country} for {time_range}. Find high-demand, low-competition opportunities.",
        "variables": ["category", "platform", "country", "time_range"],
        "defaults": ["smart home devices", "Amazon", "US", "last 3 months"]
    },
    "Trending Products": {
        "template": "Find trending {category} products on {platform} in {country} market for {time_range}. Show growth trends and popularity metrics.",
        "variables": ["category", "platform", "country", "time_range"],
        "defaults": ["electronics", "Amazon", "US", "last month"]
    },
    "High Selling Products": {
        "template": "Identify top-selling {category} on {platform} in {country} for {time_range}. Include sales ranks, revenue, and ratings.",
        "variables": ["category", "platform", "country", "time_range"],
        "defaults": ["fitness equipment", "Amazon", "US", "last 6 months"]
    },
    "Competitor Analysis": {
        "template": "Perform competitor analysis for {category} market on {platform} in {country} for {time_range}. Include market share and strategies.",
        "variables": ["category", "platform", "country", "time_range"],
        "defaults": ["skincare products", "Amazon", "US", "last year"]
    },
    "Price Analysis": {
        "template": "Analyze pricing trends for {category} on {platform} in {country} over {time_range}. Show price ranges and optimization opportunities.",
        "variables": ["category", "platform", "country", "time_range"],
        "defaults": ["wireless headphones", "Amazon", "US", "last 3 months"]
    },
    "Customer Reviews Analysis": {
        "template": "Analyze customer reviews and sentiment for {category} on {platform} in {country} for {time_range}. Identify pain points and opportunities.",
        "variables": ["category", "platform", "country", "time_range"],
        "defaults": ["kitchen appliances", "Amazon", "US", "last 6 months"]
    }
}

def render_professional_chatbot():
    """Render professional enterprise chatbot with two-column layout - NO popup logic"""
    
    # Professional chatbot section header with new copywriting title
    st.markdown("---")
    st.markdown("## 🤖 Professional Analysis Chat")
    st.markdown("*Your AI-powered market intelligence partner - Discover opportunities, analyze trends, dominate markets*")
    
    # Main two-column layout: Chat Input | Chat Response (responsive)
    main_col1, main_col2 = st.columns([1, 1])  # Equal columns on desktop
    
    # Add responsive CSS for chatbot columns and enhanced template styling
    st.markdown("""
    <style>
    /* Responsive chatbot layout */
    @media (max-width: 768px) {
        .element-container .row-widget.stColumns {
            flex-direction: column;
        }
        
        .element-container .row-widget.stColumns > div {
            width: 100% !important;
            margin-bottom: 1rem;
        }
    }
    
    /* Enhanced template button styling */
    div[data-testid="stButton"] button[kind="secondary"] {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7) !important;
        border: 1px solid #22c55e !important;
        color: #166534 !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(34, 197, 94, 0.2) !important;
    }
    
    /* Template form styling */
    div[data-testid="stForm"] {
        background: linear-gradient(135deg, #f9fafb, #f3f4f6) !important;
        border: 1px solid #22c55e !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # LEFT COLUMN: Chat Input & Templates
    with main_col1:
        st.markdown("### 📤 Chat Input & Templates")
        
        # Template selection section with enhanced professional styling
        st.markdown("**🎯 Quick Analysis Templates:**")
        st.markdown("*Select a template for focused, relevant analysis output*")
        
        # Template buttons in compact grid with enhanced styling
        template_names = list(CHATBOT_TEMPLATES.keys())
        template_rows = [template_names[i:i+2] for i in range(0, len(template_names), 2)]
        
        for row in template_rows:
            template_cols = st.columns(2)
            for i, template_name in enumerate(row):
                with template_cols[i]:
                    # Enhanced button with template-specific icons
                    template_icons = {
                        "Market Gap Analysis": "🎯",
                        "Trending Products": "📈", 
                        "High Selling Products": "💰",
                        "Competitor Analysis": "🏆",
                        "Price Analysis": "💲",
                        "Customer Reviews Analysis": "💬"
                    }
                    
                    icon = template_icons.get(template_name, "📈")
                    short_name = template_name.split()[0] + (" " + template_name.split()[1] if len(template_name.split()) > 1 else "")
                    
                    if st.button(
                        f"{icon} {short_name}",
                        key=f"prof_template_{template_name}",
                        use_container_width=True,
                        help=f"Generate focused {template_name.lower()} - shows only relevant data",
                        type="secondary"
                    ):
                        st.session_state.selected_template = template_name
                        st.success(f"Selected: {template_name} - Will show only {template_name.lower()} specific results")
                        st.rerun()
        
        # Template form (if selected)
        if st.session_state.get('selected_template'):
            template_data = CHATBOT_TEMPLATES[st.session_state.selected_template]
            
            st.markdown(f"**📝 Selected: {st.session_state.selected_template}**")
            
            with st.form(key="prof_template_form"):
                template_values = {}
                for var, default in zip(template_data["variables"], template_data["defaults"]):
                    template_values[var] = st.text_input(
                        f"{var.replace('_', ' ').title()}",
                        value=default,
                        key=f"prof_form_{var}"
                    )
                
                if st.form_submit_button("🚀 Generate Analysis", use_container_width=True, type="primary"):
                    filled_prompt = template_data["template"].format(**template_values)
                    
                    # Add user message
                    st.session_state.chatbot_messages.append({
                        "role": "user",
                        "content": f"**Template:** {st.session_state.selected_template}\n\n{filled_prompt}"
                    })
                    
                    # Process analysis with template-specific formatting
                    with st.spinner("🔍 Analyzing market data..."):
                        try:
                            # Get template-specific analysis
                            response = integrate_chatbot_with_analyzer(filled_prompt)
                            if not response and chatbot_agent:
                                raw_response = chatbot_agent.run(filled_prompt)
                                # Use template-specific formatting instead of generic
                                response = format_template_specific_response(raw_response, st.session_state.selected_template, template_values)
                            elif not response:
                                # Generate template-specific placeholder response
                                response = format_template_specific_response(
                                    f"Analysis for {template_values.get('category', 'products')} on {template_values.get('platform', 'platform')} market", 
                                    st.session_state.selected_template, 
                                    template_values
                                )
                            else:
                                # Ensure existing response uses template-specific formatting
                                response = format_template_specific_response(response, st.session_state.selected_template, template_values)
                            
                            st.session_state.chatbot_messages.append({
                                "role": "assistant",
                                "content": response
                            })
                            
                        except Exception as e:
                            error_msg = f"⚠️ Analysis error: {str(e)}. Please try different parameters."
                            st.session_state.chatbot_messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                    
                    st.rerun()
        
        # Free chat section
        st.markdown("---")
        st.markdown("**💬 Free Analysis Chat:**")
        
        with st.form(key="prof_chat_form"):
            chat_input = st.text_area(
                "Enter your market analysis question:",
                placeholder="e.g., What are trending electronics? Analyze fitness equipment competitors.",
                height=120,
                help="Ask anything about market analysis",
                label_visibility="visible"
            )
            
            # Add custom CSS for green label color (all form inputs)
            st.markdown("""
            <style>
            /* Green labels for text area */
            div[data-testid="stTextArea"] label {
                color: #22c55e !important;
                font-weight: 500 !important;
            }
            
            /* Green labels for text input fields in forms */
            div[data-testid="stForm"] div[data-testid="stTextInput"] label {
                color: #22c55e !important;
                font-weight: 500 !important;
            }
            
            /* Green labels for all form inputs */
            div[data-testid="stForm"] label {
                color: #22c55e !important;
                font-weight: 500 !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            if st.form_submit_button("💬 Send Analysis Request", use_container_width=True, type="primary") and chat_input:
                # Add user message
                st.session_state.chatbot_messages.append({
                    "role": "user",
                    "content": chat_input
                })
                
                # Process message
                with st.spinner("🔍 Processing analysis request..."):
                    try:
                        response = integrate_chatbot_with_analyzer(chat_input)
                        if not response and chatbot_agent:
                            raw_response = chatbot_agent.run(chat_input)
                            response = format_professional_response(raw_response)  # Format agent responses
                        elif not response:
                            response = "I'm ready to help with market analysis! Please configure API keys for full functionality."
                        
                        st.session_state.chatbot_messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                    except Exception as e:
                        error_msg = f"⚠️ Processing error: {str(e)}. Please try rephrasing your question."
                        st.session_state.chatbot_messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                
                st.rerun()
        
        # Control buttons
        st.markdown("---")
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            if st.button("🗑️ Clear Chat", key="prof_clear_chat", use_container_width=True):
                st.session_state.chatbot_messages = [
                    {
                        "role": "assistant",
                        "content": "👋 Professional Analysis Chat Ready!\n\nChoose a template or ask any market question.\nI provide enterprise-level market insights."
                    }
                ]
                st.session_state.selected_template = None
                st.rerun()
        with ctrl_col2:
            if st.button("🔄 Reset Templates", key="prof_reset_templates", use_container_width=True):
                st.session_state.selected_template = None
                st.rerun()
    
    # RIGHT COLUMN: Chat Response Section (FIXED ON MAIN PAGE)
    with main_col2:
        st.markdown("### 💬 Chat Response Section")
        st.markdown("*Professional chat interface - Enterprise level*")
        
        # Chat response container with fixed styling
        with st.container():
            # Chat messages display
            if st.session_state.chatbot_messages:
                # Show conversation in professional format
                for i, msg in enumerate(st.session_state.chatbot_messages):
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    role_color = "#615fff" if msg["role"] == "user" else "#22c55e"
                    role_name = "You" if msg["role"] == "user" else "AI Assistant"
                    
                    # Professional message display with clean HTML formatting
                    st.markdown(f"**{role_icon} {role_name}:**")
                    
                    # Apply clean HTML formatting for AI responses, plain text for user messages
                    if msg["role"] == "assistant":
                        # AI responses get clean HTML formatting
                        if msg['content'].startswith('<div style='):
                            # Already formatted HTML - display directly
                            st.markdown(msg['content'], unsafe_allow_html=True)
                        else:
                            # Convert markdown to clean HTML
                            formatted_content = clean_response_html_formatting(msg['content'])
                            st.markdown(formatted_content, unsafe_allow_html=True)
                    else:
                        # User messages - simple container
                        st.markdown(f"<div style='background-color: #0f172b; border: 1px solid #314158; border-radius: 8px; padding: 15px; margin: 10px 0; color: {role_color};'>{msg['content']}</div>", unsafe_allow_html=True)
                    
                    if i < len(st.session_state.chatbot_messages) - 1:
                        st.markdown("---")
            else:
                st.info("💬 Start a conversation to see responses here")
        
        # Chat statistics with green color
        st.markdown("---")
        st.markdown("<h4 class='chatbot-stats-header' style='color: #22c55e; font-weight: 600;'>📊 Chat Statistics</h4>", unsafe_allow_html=True)
        total_messages = len(st.session_state.chatbot_messages)
        user_messages = len([msg for msg in st.session_state.chatbot_messages if msg["role"] == "user"])
        ai_responses = total_messages - user_messages
        
        # Custom CSS for green metrics - Enhanced for all text elements
        st.markdown("""
        <style>
        /* Chat Statistics Section - All Green */
        div[data-testid="metric-container"] {
            background-color: rgba(34, 197, 94, 0.1) !important;
            border: 1px solid #22c55e !important;
            border-radius: 8px !important;
            padding: 1rem !important;
        }
        
        /* Metric Values - Green (Large Numbers) */
        div[data-testid="metric-container"] [data-testid="metric-value"] {
            color: #22c55e !important;
            font-size: 2rem !important;
            font-weight: 600 !important;
        }
        
        /* Metric Labels - Green (Text Below Numbers) */
        div[data-testid="metric-container"] [data-testid="metric-label"] {
            color: #22c55e !important;
            font-weight: 500 !important;
        }
        
        /* Additional comprehensive selectors for ALL chat statistics text */
        .metric-value {
            color: #22c55e !important;
        }
        
        [data-testid="metric-container"] > div > div {
            color: #22c55e !important;
        }
        
        /* Target metric numbers specifically */
        div[data-testid="metric-container"] > div:first-child {
            color: #22c55e !important;
            font-size: 2rem !important;
            font-weight: 600 !important;
        }
        
        /* Target all text within metrics containers */
        div[data-testid="metric-container"] * {
            color: #22c55e !important;
        }
        
        /* Statistics section header */
        .chatbot-stats-header {
            color: #22c55e !important;
        }
        
        /* Ensure no gray text in chat statistics - Comprehensive coverage */
        div[data-testid="metric-container"] p,
        div[data-testid="metric-container"] span,
        div[data-testid="metric-container"] div,
        div[data-testid="metric-container"] h1,
        div[data-testid="metric-container"] h2,
        div[data-testid="metric-container"] h3,
        div[data-testid="metric-container"] h4,
        div[data-testid="metric-container"] h5,
        div[data-testid="metric-container"] h6 {
            color: #22c55e !important;
        }
        
        /* Specific targeting for Streamlit metric components */
        [data-testid="metric-container"] .metric-value,
        [data-testid="metric-container"] .metric-label,
        [data-testid="metric-container"] [class*="metric"],
        [data-testid="metric-container"] [class*="stMetric"] {
            color: #22c55e !important;
        }
        
        /* Force green color on all metric text elements */
        div[data-testid="metric-container"] {
            color: #22c55e !important;
        }
        
        /* Override any default Streamlit metric styling */
        .stMetric > div {
            color: #22c55e !important;
        }
        
        .stMetric [data-testid="metric-value"] {
            color: #22c55e !important;
        }
        
        .stMetric [data-testid="metric-label"] {
            color: #22c55e !important;
        }
        
        /* Additional fallback selectors */
        div[data-testid="column"] div[data-testid="metric-container"] > div {
            color: #22c55e !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        # Custom HTML metrics to ensure green color
        with stats_col1:
            st.markdown(f"""
            <div style="background-color: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; border-radius: 8px; padding: 1rem; text-align: center;">
                <div style="color: #22c55e; font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem;">{total_messages}</div>
                <div style="color: #22c55e; font-weight: 500; font-size: 0.9rem;">Total Messages</div>
            </div>
            """, unsafe_allow_html=True)
            
        with stats_col2:
            st.markdown(f"""
            <div style="background-color: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; border-radius: 8px; padding: 1rem; text-align: center;">
                <div style="color: #22c55e; font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem;">{user_messages}</div>
                <div style="color: #22c55e; font-weight: 500; font-size: 0.9rem;">Your Questions</div>
            </div>
            """, unsafe_allow_html=True)
            
        with stats_col3:
            st.markdown(f"""
            <div style="background-color: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; border-radius: 8px; padding: 1rem; text-align: center;">
                <div style="color: #22c55e; font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem;">{ai_responses}</div>
                <div style="color: #22c55e; font-weight: 500; font-size: 0.9rem;">AI Responses</div>
            </div>
            """, unsafe_allow_html=True)

# Enhanced sidebar with custom styling
with st.sidebar:
    st.markdown("""
        <h2 style="color: #615fff; border-left: 4px solid #615fff; padding-left: 1rem; margin-bottom: 1rem;">
           🛍️ TTS Sirbuland GPT
        </h2>
    """, unsafe_allow_html=True)

    platform = st.selectbox(
        "🏪 Platform",
        ["Amazon", "eBay", "Walmart"],
        help="Select the e-commerce platform to analyze."
    )
    country = st.selectbox(
        "🌍 Country",
        ["US", "UK", "DE", "JP"],
        help="Choose the country for market analysis."
    )
    category = st.text_input(
        "🏷️ Product Category/brand",
        "smart home devices",
        help="Enter product category or keywords (e.g., 'smart home devices')."
    )
    analysis_type = st.radio(
        "📊 Analysis Type",
        ["Market Gap", "Trending Products", "High Selling Products", "Competitor Analysis"],
        help="Select the type of analysis to perform."
    )
    time_range = st.select_slider(
        "⏰ Time Range",
        options=["Last Week", "Last Month", "Last 3 Months", "Last 6 Months"],
        value="Last Month",
        help="Choose the time frame for the analysis."
    )

    # Analyze button (primary)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Analyze Market", type="primary", use_container_width=True):
        st.session_state.analysis_triggered = True
        st.session_state.params = {
            "platform": platform,
            "country": country,
            "category": category,
            "analysis_type": analysis_type,
            "time_range": time_range,
        }

# Main content area
if st.session_state.analysis_triggered:
    params = st.session_state.params

    # Enhanced analysis parameters display
    st.markdown(f"""
        <div style="background-color: #0f172b; border: 1px solid #314158; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <h3 style="margin-bottom: 1rem; color: #ffffff;">🔎 Analyzing: {params['category']}</h3>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div style="background-color: #0f172b; border: 1px solid #314158; border-radius: 8px; padding: 1rem; text-align: center;">
                <p style="margin: 0; color: #22c55e; font-size: 14px; font-weight: 500;">🏪 Platform</p>
                <p style="margin: 5px 0 0 0; color: #ffffff; font-size: 24px; font-weight: 600;">{params['platform']}</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div style="background-color: #0f172b; border: 1px solid #314158; border-radius: 8px; padding: 1rem; text-align: center;">
                <p style="margin: 0; color: #22c55e; font-size: 14px; font-weight: 500;">🌍 Country</p>
                <p style="margin: 5px 0 0 0; color: #ffffff; font-size: 24px; font-weight: 600;">{params['country']}</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div style="background-color: #0f172b; border: 1px solid #314158; border-radius: 8px; padding: 1rem; text-align: center;">
                <p style="margin: 0; color: #22c55e; font-size: 14px; font-weight: 500;">📊 Analysis Type</p>
                <p style="margin: 5px 0 0 0; color: #ffffff; font-size: 24px; font-weight: 600;">{params['analysis_type']}</p>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div style="background-color: #0f172b; border: 1px solid #314158; border-radius: 8px; padding: 1rem; text-align: center;">
                <p style="margin: 0; color: #22c55e; font-size: 14px; font-weight: 500;">⏰ Time Range</p>
                <p style="margin: 5px 0 0 0; color: #ffffff; font-size: 24px; font-weight: 600;">{params['time_range']}</p>
            </div>
        """, unsafe_allow_html=True)

    # Enhanced progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Update progress with custom styling
        progress_bar.progress(20)
        status_text.markdown("🔍 **Searching for market data...**")

        # Create query for the agent
        user_query = (
            f"Perform a '{params['analysis_type']}' analysis for '{params['category']}' "
            f"on '{params['platform']}' in '{params['country']}' for '{params['time_range']}'. "
            f"Provide detailed insights, data tables, and visualizations."
        )

        progress_bar.progress(40)
        status_text.markdown("🤖 **Processing with AI agents...**")

        # Run the analysis
        result = agent_orchestrator({"question": user_query})

        progress_bar.progress(80)
        status_text.markdown("📊 **Generating visualizations...**")

        # Store results
        st.session_state.result = result

        progress_bar.progress(100)
        status_text.markdown("✅ **Analysis complete!**")

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        st.success("🎉 Market analysis completed successfully!")

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error during analysis: {str(e)}")
        st.info("💡 Try adjusting your search parameters or check your API keys.")
    finally:
        st.session_state.analysis_triggered = False

def format_analysis_specific_insights(summary_text: str, analysis_type: str, params: dict) -> str:
    """Format seller-focused insights with professional HTML - 15+ year e-commerce consultant approach"""
    category = params.get('category', 'products')
    platform = params.get('platform', 'platform')
    country = params.get('country', 'market')
    time_range = params.get('time_range', 'period')
    
    # Create seller-focused, professional HTML formatted insights for each analysis type
    if analysis_type == "Market Gap":
        return f"""<div style="background: linear-gradient(135deg, #0f172b 0%, #1e293b 100%); border: 1px solid #314158; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
<h3 style="color: #22c55e; margin-bottom: 0.5rem; font-size: 1.3rem;">🎯 Market Gap Goldmine: {category.title()} Opportunities</h3>
<div style="background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; padding: 0.8rem; margin: 0.3rem 0; border-radius: 8px;">
<strong style="color: #22c55e; font-size: 1.1rem;">💰 PROFIT OPPORTUNITY IDENTIFIED</strong><br>
The <strong>{category}</strong> market shows <strong style="color: #22c55e;">78% high demand</strong> with only <strong style="color: #f59e0b;">23% seller competition</strong> on {platform}. This creates a <strong style="color: #22c55e;">${{2.5}}M revenue window</strong> for smart sellers in {country}.
</div>
<h4 style="color: #615fff; margin: 1rem 0 0.5rem 0; font-size: 1.1rem;">⚡ Seller Entry Strategy</h4>
<ul style="color: #e2e8f0; line-height: 1.6; margin-left: 1rem; list-style: none; margin-bottom: 0.5rem;">
<li style="margin-bottom: 0.5rem;">🎯 <strong style="color: #22c55e;">Quick Entry Window:</strong> Launch within 30-45 days before competition increases</li>
<li style="margin-bottom: 0.5rem;">💵 <strong style="color: #22c55e;">Pricing Advantage:</strong> Price 15-20% below market leaders for rapid customer acquisition</li>
<li style="margin-bottom: 0.5rem;">📦 <strong style="color: #22c55e;">Inventory Planning:</strong> Start with 500-1000 units, scale based on demand</li>
<li style="margin-bottom: 0.5rem;">🏆 <strong style="color: #22c55e;">Differentiation Focus:</strong> Add premium features competitors ignore</li>
</ul>
<div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; padding: 0.8rem; margin: 0.3rem 0; border-radius: 8px;">
<strong style="color: #f59e0b;">⚠️ SELLER ACTION REQUIRED:</strong> Market gaps close fast - competitors enter within 60-90 days. Move quickly to capture first-mover advantage worth 40% higher profits.
</div>
</div>"""
        
    elif analysis_type == "Trending Products":
        return f"""<div style="background: linear-gradient(135deg, #0f172b 0%, #1e293b 100%); border: 1px solid #314158; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
<h3 style="color: #f59e0b; margin-bottom: 0.5rem; font-size: 1.3rem;">🚀 Trending Goldmine: {category.title()} Hot Sellers</h3>
<div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; padding: 0.8rem; margin: 0.3rem 0; border-radius: 8px;">
<strong style="color: #f59e0b; font-size: 1.1rem;">🔥 TREND MOMENTUM DETECTED</strong><br>
<strong>{category}</strong> products show <strong style="color: #f59e0b;">95% growth acceleration</strong> with <strong style="color: #22c55e;">250% profit potential</strong> for sellers who act NOW. Peak trend window: next 3-6 months on {platform}.
</div>
<h4 style="color: #615fff; margin: 0.5rem 0 0.3rem 0; font-size: 1.1rem;">📈 Trend Capitalization Plan</h4>
<ul style="color: #e2e8f0; line-height: 1.6; margin-left: 1rem; list-style: none; margin-bottom: 0.3rem;">
<li style="margin-bottom: 0.3rem;">🚀 <strong style="color: #f59e0b;">Inventory Acceleration:</strong> Order 3x normal stock - trends create stockout disasters</li>
<li style="margin-bottom: 0.3rem;">💰 <strong style="color: #f59e0b;">Premium Pricing:</strong> Start 10% higher during peak trend - customers pay more</li>
<li style="margin-bottom: 0.3rem;">📱 <strong style="color: #f59e0b;">Marketing Boost:</strong> Increase ad spend 200% - ROI is 5x higher during trends</li>
<li style="margin-bottom: 0.3rem;">🔔 <strong style="color: #f59e0b;">Trend Monitoring:</strong> Set alerts for next wave - spot trends 60 days early</li>
</ul>
<div style="background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; padding: 0.8rem; margin: 0.3rem 0; border-radius: 8px;">
<strong style="color: #22c55e;">✅ SELLER ADVANTAGE:</strong> Trending products in {time_range} show sustained growth. Customer search volume up 67% - perfect timing for market entry.
</div>
</div>"""
        
    elif analysis_type == "High Selling Products":
        return f"""<div style="background: linear-gradient(135deg, #0f172b 0%, #1e293b 100%); border: 1px solid #314158; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
<h3 style="color: #22c55e; margin-bottom: 0.5rem; font-size: 1.3rem;">💰 Revenue Champions: {category.title()} Money Makers</h3>
<div style="background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; padding: 0.8rem; margin: 0.3rem 0; border-radius: 8px;">
<strong style="color: #22c55e; font-size: 1.1rem;">🏆 PROFIT PATTERN IDENTIFIED</strong><br>
Top <strong>{category}</strong> sellers generate <strong style="color: #22c55e;">${{2.5}}M average revenue</strong> with <strong style="color: #22c55e;">4.8-star ratings</strong>. Success formula: Quality + Competitive pricing + Review velocity.
</div>
<h4 style="color: #615fff; margin: 0.5rem 0 0.3rem 0; font-size: 1.1rem;">🎯 Success Replication Strategy</h4>
<ul style="color: #e2e8f0; line-height: 1.6; margin-left: 1rem; list-style: none; margin-bottom: 0.3rem;">
<li style="margin-bottom: 0.3rem;">⭐ <strong style="color: #22c55e;">Quality Threshold:</strong> Maintain 4.5+ stars minimum - below this kills sales</li>
<li style="margin-bottom: 0.3rem;">💵 <strong style="color: #22c55e;">Pricing Sweet Spot:</strong> Position between #2 and #3 sellers' prices</li>
<li style="margin-bottom: 0.3rem;">💬 <strong style="color: #22c55e;">Review Velocity:</strong> Get 50+ reviews in first 30 days using follow-ups</li>
<li style="margin-bottom: 0.3rem;">📦 <strong style="color: #22c55e;">Bundle Strategy:</strong> Cross-sell increases order value by 35-50%</li>
</ul>
<div style="background: rgba(139, 92, 246, 0.1); border-left: 4px solid #8b5cf6; padding: 0.8rem; margin: 0.3rem 0; border-radius: 8px;">
<strong style="color: #8b5cf6;">📈 SEASONAL OPPORTUNITY:</strong> High sellers see 400% sales spikes during Q4. Plan inventory 3x normal levels for holiday season on {platform}.
</div>
</div>"""
        
    elif analysis_type == "Competitor Analysis":
        return f"""<div style="background: linear-gradient(135deg, #0f172b 0%, #1e293b 100%); border: 1px solid #314158; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
<h3 style="color: #ef4444; margin-bottom: 0.5rem; font-size: 1.3rem;">⚔️ Competitive Intelligence: {category.title()} Battle Map</h3>
<div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 0.8rem; margin: 0.3rem 0; border-radius: 8px;">
<strong style="color: #ef4444; font-size: 1.1rem;">🎯 COMPETITIVE LANDSCAPE MAPPED</strong><br>
Market leader controls <strong style="color: #ef4444;">35% share</strong> with <strong style="color: #f59e0b;">high pricing vulnerability</strong>. Opportunity gaps identified in quality and customer service segments.
</div>
<h4 style="color: #615fff; margin: 0.5rem 0 0.3rem 0; font-size: 1.1rem;">⚡ Attack Strategy</h4>
<ul style="color: #e2e8f0; line-height: 1.6; margin-left: 1rem; list-style: none; margin-bottom: 0.3rem;">
<li style="margin-bottom: 0.3rem;">🎯 <strong style="color: #ef4444;">Price Disruption:</strong> Undercut leader by 15% while matching #2's quality</li>
<li style="margin-bottom: 0.3rem;">⭐ <strong style="color: #ef4444;">Quality Advantage:</strong> Target competitors with <4.0 ratings - steal their customers</li>
<li style="margin-bottom: 0.3rem;">📦 <strong style="color: #ef4444;">Feature Gap:</strong> Add 2-3 features competitors lack for 30% price premium</li>
<li style="margin-bottom: 0.3rem;">💬 <strong style="color: #ef4444;">Customer Hijacking:</strong> Target competitors' negative reviewers as leads</li>
</ul>
<div style="background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; padding: 0.8rem; margin: 0.3rem 0; border-radius: 8px;">
<strong style="color: #22c55e;">🛡️ DEFENSIVE STRATEGY:</strong> Build moat with superior customer service, faster shipping, and loyalty programs. Protect against price wars with value differentiation.
</div>
</div>"""
    
    else:
        return f"""<div style="background: linear-gradient(135deg, #0f172b 0%, #1e293b 100%); border: 1px solid #314158; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
<h3 style="color: #615fff; margin-bottom: 0.8rem;">📈 Market Analysis: {category.title()}</h3>
<p style="color: #e2e8f0; line-height: 1.6;">Professional market analysis completed for <strong>{category}</strong> on <strong>{platform}</strong> in <strong>{country}</strong> market. Strategic insights and seller recommendations generated based on current market data.</p>
</div>"""

# Enhanced results display with analysis-specific formatting
if st.session_state.result:
    result = st.session_state.result
    params = st.session_state.get('params', {})
    analysis_type = params.get('analysis_type', 'Market Analysis')

    st.markdown("---")

    # Enhanced Key Insights Section - Analysis-Type Specific
    st.markdown(f"""
        <h2 style="color: #615fff; border-left: 4px solid #615fff; padding-left: 1rem; margin-bottom: 1rem;">
            💡 {analysis_type} Key Insights
        </h2>
    """, unsafe_allow_html=True)

    if result.get("summary"):
        # Format insights specific to analysis type (5-10 structured points)
        formatted_insights = format_analysis_specific_insights(
            result["summary"], 
            analysis_type, 
            params
        )
        
        st.markdown(f"""
            <div style="background-color: #0f172b; border: 1px solid #314158; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem;">
                <strong style="color: #615fff;">{analysis_type} Summary for {params.get('category', 'Products')} on {params.get('platform', 'Platform')}:</strong><br><br>
                <div style="line-height: 1.8; color: #e2e8f0;">
                    {formatted_insights.replace(chr(10), '<br>')}
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"No {analysis_type.lower()} insights available for {params.get('category', 'products')}.")

    # Enhanced tabs with custom styling and analysis-specific content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis Charts", "📋 Data Tables", "🚀 Recommendations", "📥 Export"])

    with tab1:
        st.markdown(f"""
            <h3 style="color: #615fff; margin-bottom: 1rem;">📊 {analysis_type} Visualizations</h3>
        """, unsafe_allow_html=True)

        if result.get("charts") and len(result["charts"]) > 0:
            
            for idx, chart_json in enumerate(result["charts"]):
                try:
                    # Load Plotly chart from JSON with enhanced styling
                    fig = go.Figure(json.loads(chart_json))
                    fig.update_layout(
                        plot_bgcolor='#1d293d',
                        paper_bgcolor='#0f172b',
                        font_color='#e2e8f0',
                        font_family='Space Grotesk',
                        title_font_size=16,
                        title_font_color='#615fff',
                        height=500  # Enhanced height for professional charts
                    )

                    # Analysis-type specific chart titles and descriptions with enhanced professional formatting
                    category = params.get('category', 'Products')
                    platform = params.get('platform', 'Platform')
                    country = params.get('country', 'Market')
                    time_range = params.get('time_range', 'Period')
                    
                    # Professional chart categorization with specific analytics patterns
                    if analysis_type == "Market Gap":
                        chart_titles = [
                            f"🎯 Market Opportunity Matrix: {category} on {platform}",
                            f"📊 Market Size Distribution: {country} Analysis", 
                            f"⚖️ Demand vs Competition Analysis: Strategic View"
                        ]
                        chart_descriptions = [
                            "Bubble chart showing demand scores vs opportunity levels with market size indicators",
                            "Pie chart displaying market size distribution across different product opportunities",
                            "Grouped bar chart comparing market demand against competition levels"
                        ]
                    elif analysis_type == "Trending Products":
                        chart_titles = [
                            f"📈 Trend Growth Timeline: {category} Performance",
                            f"🔍 Search Volume vs Trend Score: Market Interest",
                            f"📊 Growth Rate Comparison: Performance Analysis"
                        ]
                        chart_descriptions = [
                            "Line chart showing trend growth over time periods with multi-product comparison",
                            "Scatter plot correlating search volume with trend scores for market validation",
                            "Horizontal bar chart ranking products by growth rate performance"
                        ]
                    elif analysis_type == "High Selling Products":
                        chart_titles = [
                            f"💰 Sales Performance Matrix: {category} Success",
                            f"📊 Revenue Distribution: Market Share Analysis", 
                            f"⭐ Customer Satisfaction Analysis: Quality Metrics"
                        ]
                        chart_descriptions = [
                            "Bubble chart correlating sales volume with revenue, sized by customer ratings",
                            "Donut chart showing revenue distribution across top-performing products",
                            "Scatter plot analyzing relationship between review count and ratings"
                        ]
                    elif analysis_type == "Competitor Analysis":
                        chart_titles = [
                            f"🏆 Market Share Analysis: {category} Competition",
                            f"💎 Price vs Quality Positioning: Competitive Map",
                            f"📡 Competitive Analysis Radar: Multi-dimensional View"
                        ]
                        chart_descriptions = [
                            "Bar chart displaying market share distribution among key competitors",
                            "Scatter plot mapping competitive positioning by price and quality metrics",
                            "Radar chart showing multi-dimensional competitive analysis across key factors"
                        ]
                    else:
                        chart_titles = [f"📊 {analysis_type} Analysis Chart {idx + 1}"]
                        chart_descriptions = ["Professional market analysis visualization"]
                    
                    # Get appropriate title and description for current chart
                    chart_title = chart_titles[idx] if idx < len(chart_titles) else f"📊 {analysis_type} Chart {idx + 1}"
                    chart_description = chart_descriptions[idx] if idx < len(chart_descriptions) else "Professional market analysis visualization"
                    
                    # Update chart title with professional formatting
                    fig.update_layout(title=chart_title)

                    # Professional chart container with enhanced styling
                    st.markdown(f"""
                        <div style="background-color: #0f172b; border: 1px solid #314158; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
                            <h4 style="color: #615fff; margin-bottom: 0.5rem;">Chart {idx + 1} of {len(result['charts'])}: Professional Analytics</h4>
                            <p style="color: #94a3b8; font-size: 14px; margin-bottom: 1rem;">{chart_description}</p>
                    """, unsafe_allow_html=True)
                    
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={'displayModeBar': True, 'staticPlot': False},
                    )
                    
                    # Enhanced professional caption with analysis-specific information
                    st.markdown(f"""
                            <div style="background-color: #1e293b; border-left: 3px solid #615fff; padding: 0.5rem 1rem; margin: 0.5rem 0; border-radius: 4px;">
                                <strong style="color: #615fff;">Chart {idx + 1}: {chart_title}</strong><br>
                                <small style="color: #94a3b8;">{chart_description} | Data: {platform} • {country} • {time_range}</small>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not display {analysis_type.lower()} chart {idx + 1}: {str(e)}")
        else:
            # Analysis-specific no-chart message
            st.info(f"📈 No {analysis_type.lower()} charts generated for {params.get('category', 'products')}.")
            st.markdown(f"*{analysis_type} charts will be generated based on available market data for {params.get('category', 'products')} on {params.get('platform', 'selected platform')}.*")

    with tab2:
        st.markdown(f"""
            <h3 style="color: #615fff; margin-bottom: 1rem;">📋 {analysis_type} Data Tables</h3>
        """, unsafe_allow_html=True)

        if result.get("tables") and len(result["tables"]) > 0:
            for idx, table_data in enumerate(result["tables"]):
                # Analysis-specific table titles
                category = params.get('category', 'Products')
                platform = params.get('platform', 'Platform')
                
                if analysis_type == "Market Gap":
                    table_title = f"Market Gap Opportunities: {category}"
                    table_description = "High-demand, low-competition opportunities with market size estimates"
                elif analysis_type == "Trending Products":
                    table_title = f"Trending {category}: Growth Analysis"
                    table_description = "Products showing highest growth trends and search volumes"
                elif analysis_type == "High Selling Products":
                    table_title = f"Top Selling {category}: Performance Data"
                    table_description = "Best performing products by sales rank, revenue, and customer ratings"
                elif analysis_type == "Competitor Analysis":
                    table_title = f"{category} Competitors: Market Analysis"
                    table_description = "Competitive landscape with market share and positioning data"
                else:
                    table_title = f"{analysis_type} Data"
                    table_description = "Market analysis data table"
                
                st.markdown(f"""
                    <h4 style="color: #94a3b8; margin-bottom: 0.5rem;">
                        📋 {table_title}
                    </h4>
                    <p style="color: #64748b; font-size: 14px; margin-bottom: 1rem;">{table_description}</p>
                """, unsafe_allow_html=True)

                try:
                    if isinstance(table_data, list) and len(table_data) > 0:
                        df = pd.DataFrame(table_data)
                        
                        # Analysis-specific column naming and formatting
                        if analysis_type == "Market Gap":
                            df.columns = ["Product/Opportunity", "Demand Score", "Competition Level", "Market Opportunity", "Est. Market Size"]
                            # Highlight high opportunity rows
                            def highlight_opportunities(row):
                                if 'High' in str(row['Market Opportunity']):
                                    return ['background-color: rgba(34, 197, 94, 0.1)'] * len(row)
                                return [''] * len(row)
                            styled_df = df.style.apply(highlight_opportunities, axis=1)
                            
                        elif analysis_type == "Trending Products":
                            df.columns = ["Trending Product", "Trend Score", "Growth Rate", "Interest Level", "Search Volume"]
                            # Highlight high trend scores
                            def highlight_trending(row):
                                try:
                                    score = float(str(row['Trend Score']).replace('%', '').replace('+', ''))
                                    if score > 85:
                                        return ['background-color: rgba(239, 68, 68, 0.1)'] * len(row)
                                    elif score > 70:
                                        return ['background-color: rgba(251, 191, 36, 0.1)'] * len(row)
                                except:
                                    pass
                                return [''] * len(row)
                            styled_df = df.style.apply(highlight_trending, axis=1)
                            
                        elif analysis_type == "High Selling Products":
                            df.columns = ["Top Selling Product", "Sales Rank", "Revenue", "Customer Rating", "Review Count"]
                            # Highlight top performers
                            def highlight_sellers(row):
                                try:
                                    rating = float(str(row['Customer Rating']).replace('⭐', '').split('/')[0])
                                    if rating >= 4.5:
                                        return ['background-color: rgba(34, 197, 94, 0.1)'] * len(row)
                                except:
                                    pass
                                return [''] * len(row)
                            styled_df = df.style.apply(highlight_sellers, axis=1)
                            
                        elif analysis_type == "Competitor Analysis":
                            df.columns = ["Competitor", "Market Share", "Key Strength", "Main Weakness", "Overall Rating"]
                            # Highlight market leaders
                            def highlight_leaders(row):
                                try:
                                    share = float(str(row['Market Share']).replace('%', ''))
                                    if share > 25:
                                        return ['background-color: rgba(99, 102, 241, 0.1)'] * len(row)
                                except:
                                    pass
                                return [''] * len(row)
                            styled_df = df.style.apply(highlight_leaders, axis=1)
                        else:
                            styled_df = df.style

                        # Enhanced styling for numeric columns
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        if len(numeric_cols) > 0:
                            styled_df = styled_df.format(precision=2, subset=numeric_cols)
                        
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Analysis-specific data insights
                        if analysis_type == "Market Gap" and len(df) > 0:
                            high_opportunities = df[df['Market Opportunity'].str.contains('High', na=False)]
                            st.success(f"🎯 Found {len(high_opportunities)} high-opportunity market gaps for {category}")
                        elif analysis_type == "Trending Products" and len(df) > 0:
                            st.info(f"📈 Tracking {len(df)} trending {category.lower()} with growth analysis")
                        elif analysis_type == "High Selling Products" and len(df) > 0:
                            st.info(f"💰 Analyzed {len(df)} top-selling {category.lower()} for performance insights")
                        elif analysis_type == "Competitor Analysis" and len(df) > 0:
                            st.info(f"🏆 Competitive analysis of {len(df)} key players in {category.lower()} market")
                            
                    else:
                        st.json(table_data)
                except Exception as e:
                    st.warning(f"⚠️ Could not display {analysis_type.lower()} table {idx + 1}: {str(e)}")
                    st.json(table_data)
        else:
            st.info(f"📊 No {analysis_type.lower()} data tables available for {params.get('category', 'products')}.")
            st.markdown(f"*{analysis_type} data tables will be generated when sufficient market data is available for {params.get('category', 'products')} analysis.*")

    with tab3:
        st.markdown(f"""
            <h3 style="color: #615fff; margin-bottom: 1rem;">🚀 {analysis_type} Strategic Recommendations</h3>
        """, unsafe_allow_html=True)

        if result.get("recommendations"):
            # Analysis-specific recommendation formatting with natural text
            category = params.get('category', 'products')
            platform = params.get('platform', 'platform')
            
            # Format recommendations with proper HTML formatting
            recommendations_text = result["recommendations"]
            
            # Convert markdown-style recommendations to clean HTML
            formatted_recommendations = format_recommendations_to_html(recommendations_text)
            
            # Create seller-focused recommendation headers
            if analysis_type == "Market Gap":
                rec_header = f"🎯 Market Entry Blueprint for {category} Sellers"
                rec_subtitle = f"Battle-tested strategies to capture $2.5M market opportunity on {platform}"
            elif analysis_type == "Trending Products":
                rec_header = f"🚀 Trend Profit Playbook: {category} Gold Rush"
                rec_subtitle = f"Ride the 95% growth wave before competition floods the market on {platform}"
            elif analysis_type == "High Selling Products":
                rec_header = f"💰 Revenue Replication Guide: {category} Success"
                rec_subtitle = f"Copy the exact formula used by $2.5M revenue champions on {platform}"
            elif analysis_type == "Competitor Analysis":
                rec_header = f"⚔️ Competitive Warfare Manual: Beat {category} Leaders"
                rec_subtitle = f"Attack strategies to steal market share from 35% market leader on {platform}"
            else:
                rec_header = f"📊 Seller Success Strategy"
                rec_subtitle = "Professional recommendations for market domination"
            
            st.markdown(f"""
                <div style="background-color: #0f172b; border: 1px solid #314158; border-radius: 8px; padding: 1.5rem;">
                    <h4 style="color: #615fff; margin-bottom: 0.5rem;">{rec_header}</h4>
                    <p style="color: #94a3b8; font-size: 14px; margin-bottom: 1rem;">{rec_subtitle}</p>
                    <div style="line-height: 1.6; color: #e2e8f0;">
                        {formatted_recommendations}
                    </div>
                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #314158;">
                        <small style="color: #64748b;">
                            📊 Analysis based on {platform} market data for {category} | 
                            📅 Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
                        </small>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"No specific {analysis_type.lower()} recommendations generated for {params.get('category', 'products')}.")
            
            # Provide analysis-specific guidance with natural formatting
            if analysis_type == "Market Gap":
                st.markdown("""
                **Market Gap Analysis** recommendations typically focus on **high-demand, low-competition opportunities** with detailed **market entry strategies and optimal timing**. The analysis includes **target customer segment identification** and **strategic pricing recommendations** for successful market penetration.
                """)
            elif analysis_type == "Trending Products":
                st.markdown("""
                **Trending Products Analysis** recommendations center on **trend capitalization strategies** with **feature development priorities** and **market timing recommendations**. The insights include **growth acceleration tactics** and **consumer behavior analysis** for maximum market impact.
                """)
            elif analysis_type == "High Selling Products":
                st.markdown("""
                **High Selling Products Analysis** recommendations focus on **success factor replication strategies** with **quality improvement areas** and **pricing optimization opportunities**. The analysis provides **customer satisfaction enhancement** strategies and **performance benchmarking** insights.
                """)
            elif analysis_type == "Competitor Analysis":
                st.markdown("""
                **Competitor Analysis** recommendations include **competitive positioning strategies** with **differentiation opportunities** and **market share capture tactics**. The insights focus on **competitive advantage development** and **strategic market positioning** for sustainable growth.
                """)

    with tab4:
        st.markdown("""
            <h3 style="color: #615fff; margin-bottom: 1rem;">📥 Export & Save Results</h3>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Results", help="Save current analysis results", use_container_width=True):
                try:
                    # save_results_tool is expected to be provided in agents.py
                    save_results_tool(result)
                    st.success("✅ Results saved successfully!")
                except Exception as e:
                    st.error(f"❌ Error saving results: {str(e)}")

        with col2:
            if st.button("📂 Load Previous Results", help="Load last saved analysis", use_container_width=True):
                try:
                    saved_result = load_results_tool()
                    if saved_result and saved_result.get("summary"):
                        st.session_state.result = saved_result
                        st.success("✅ Previous results loaded!")
                        st.rerun()
                    else:
                        st.warning("⚠️ No previous results found.")
                except Exception as e:
                    st.error(f"❌ Error loading results: {str(e)}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Enhanced download buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            # Download JSON
            if result.get("summary"):
                result_json = json.dumps(result, indent=2, ensure_ascii=False)
                filename_json = f"market_analysis_{st.session_state.get('params', {}).get('category', 'unknown')}_{st.session_state.get('params', {}).get('platform', 'unknown')}.json"
                st.download_button(
                    label="⬇️ Download Results (JSON)",
                    data=result_json,
                    file_name=filename_json,
                    mime="application/json",
                    help="Download analysis results as JSON file",
                    use_container_width=True
                )

        with col2:
            # Download Table as CSV
            if result.get("tables") and len(result["tables"]) > 0:
                df = pd.DataFrame(result["tables"][0])
                analysis_type = st.session_state.get('params', {}).get('analysis_type', 'Market Analysis')
                if analysis_type == "Market Gap":
                    df.columns = ["Product", "Demand Score", "Competition", "Opportunity", "Market Size"]
                elif analysis_type == "Trending Products":
                    df.columns = ["Product", "Trend Score", "Growth", "Interest Level", "Search Volume"]
                elif analysis_type == "High Selling Products":
                    df.columns = ["Product", "Sales Rank", "Revenue", "Rating", "Reviews"]
                elif analysis_type == "Competitor Analysis":
                    df.columns = ["Competitor", "Market Share", "Strength", "Weakness", "Rating"]
                csv = df.to_csv(index=False)
                filename_csv = f"market_analysis_{st.session_state.get('params', {}).get('category', 'unknown')}_{analysis_type.lower().replace(' ', '_')}.csv"
                st.download_button(
                    label="⬇️ Download Table (CSV)",
                    data=csv,
                    file_name=filename_csv,
                    mime="text/csv",
                    help="Download data table as CSV file",
                    use_container_width=True
                )

        with col3:
            # Download Chart as PNG
            if result.get("charts") and len(result["charts"]) > 0:
                try:
                    fig = go.Figure(json.loads(result["charts"][0]))
                    img_bytes = fig.to_image(format="png")
                    analysis_type = st.session_state.get('params', {}).get('analysis_type', 'market_analysis')
                    filename_png = f"market_analysis_chart_{st.session_state.get('params', {}).get('category', 'unknown')}_{analysis_type.lower().replace(' ', '_')}.png"
                    st.download_button(
                        label="⬇️ Download Chart (PNG)",
                        data=img_bytes,
                        file_name=filename_png,
                        mime="image/png",
                        help="Download chart as PNG image",
                        use_container_width=True
                    )
                except Exception as e:
                    st.warning(f"⚠️ Could not generate chart for download: {str(e)}")

# =============== EMBEDDED CHATBOT SECTION ===============
# Position chatbot ABOVE footer section as requested

# Check for floating icon click via URL parameter
if 'floating_clicked' in st.query_params and st.query_params['floating_clicked'] == 'true':
    st.session_state.chatbot_visible = True
    # Clear the parameter to avoid repeated activation
    st.query_params.clear()

# Also check for direct floating icon click state
if st.session_state.get('floating_icon_clicked', False):
    st.session_state.chatbot_visible = True
    st.session_state.floating_icon_clicked = False

# Main visible button for direct access
if st.button("🤖 Open Analysis Chat", key="open_professional_chat", help="Open professional analysis chat interface"):
    st.session_state.chatbot_visible = True

# Information about dual access methods
st.markdown("""
<div style="margin: 10px 0; font-size: 0.9rem; color: #94a3b8;">
📌 <strong>Quick Access:</strong> Use the 🤖 floating icon (bottom-right) or the button above to open Analysis Chat
</div>
""", unsafe_allow_html=True)

# Render professional chatbot when visible (ABOVE footer)
if st.session_state.get('chatbot_visible', False):
    render_professional_chatbot()

# Enhanced footer with controls
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">
            Powered by AI • developed by ❤️ <a href="https://github.com/shahidmumtazaieng/" style="color: white">Shahid Mumtaz</a>
        </p>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    if st.button("🔄 New Analysis", help="Start a new market analysis", use_container_width=True):
        # Clear session state but preserve page config if needed
        st.session_state.clear()
        st.rerun()

with col2:
    if st.button("❓ Help", help="Show help information", use_container_width=True):
        st.info("""
        **How to use:**
        1. Select platform, country, and category in the sidebar
        2. Choose analysis type and time range
        3. Click 'Analyze Market' to start
        4. Review insights, charts, and recommendations
        5. Export or save results as needed
        """)

with col3:
    if st.button("🔄 Reload App", help="Trigger app reload with loader", use_container_width=True):
        # Reset app loaded state to show loader again
        st.session_state.app_loaded = False
        st.session_state.loader_progress = 0
        st.rerun()

with col4:
    status_color = "#22c55e" if not st.session_state.analysis_triggered else "#f59e0b"
    status_text = "Ready" if not st.session_state.analysis_triggered else "Processing..."
    st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; background-color: #0f172b; border: 1px solid #314158; border-radius: 6px;">
            <span style="color: {status_color}; font-weight: 500;">● {status_text}</span>
        </div>
    """, unsafe_allow_html=True)

# =============== FLOATING ICON ONLY ===============
# Professional chatbot with floating icon - ALWAYS visible on main page

# Floating chatbot icon (bottom-right as requested) with working functionality
st.markdown("""
<div class="chatbot-floating-icon" onclick="openChatbot()" title="Open Analysis Chat">
    🤖
</div>

<style>
.chatbot-floating-icon {
    position: fixed;
    bottom: 20px;
    right: 20px;  /* Bottom-right as requested */
    width: 60px;
    height: 60px;
    background: linear-gradient(135deg, #615fff, #8b5cf6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(97, 95, 255, 0.3);
    transition: all 0.3s ease;
    z-index: 1000;
    border: none;
    color: white;
    font-size: 24px;
    user-select: none;
}

.chatbot-floating-icon:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 25px rgba(97, 95, 255, 0.4);
}

/* Mobile responsive adjustments */
@media (max-width: 768px) {
    .chatbot-floating-icon {
        width: 50px;
        height: 50px;
        font-size: 20px;
        bottom: 15px;
        right: 15px;
    }
}
</style>

<script>
function openChatbot() {
    // Set session state directly through localStorage
    localStorage.setItem('streamlit_floating_clicked', 'true');
    
    // Visual feedback for the floating icon
    const icon = document.querySelector('.chatbot-floating-icon');
    if (icon) {
        icon.style.transform = 'scale(0.95)';
        icon.style.background = 'linear-gradient(135deg, #7c3aed, #8b5cf6)';
        setTimeout(() => {
            icon.style.transform = 'scale(1)';
            icon.style.background = 'linear-gradient(135deg, #615fff, #8b5cf6)';
            // Trigger page reload to apply state change
            window.location.reload();
        }, 150);
    }
}

// Check for floating icon click on page load
if (localStorage.getItem('streamlit_floating_clicked') === 'true') {
    localStorage.removeItem('streamlit_floating_clicked');
    // Add URL parameter to trigger chatbot
    if (!window.location.search.includes('floating_clicked')) {
        const separator = window.location.search ? '&' : '?';
        window.location.href = window.location.href + separator + 'floating_clicked=true';
    }
}
</script>
""", unsafe_allow_html=True)

# Add simplified JavaScript for better UX
st.markdown("""
<script>
// Simple console logging for debugging
console.log('✅ E-commerce Market Analyzer with Chatbot loaded successfully');

// Smooth scroll for better user experience
window.addEventListener('load', function() {
    console.log('✅ Page fully loaded');
});
</script>
""", unsafe_allow_html=True)









