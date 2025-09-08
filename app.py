import streamlit as st
import pandas as pd
import json
from typing import Dict, Any
import plotly.graph_objects as go
from agents import agent_orchestrator, load_results_tool

# Custom CSS for the dark theme
st.markdown("""
<style>
    @import url('https://fonts.gstatic.com/s/spacegrotesk/v21/V8mDoQDjQSkFtoMM3T6r8E7mPbF4C_k3HqU.woff2');
    
    :root {
        --text-color: #e2e8f0;
        --primary-color: #615fff;
        --background-color: #1d293d;
        --secondary-background-color: #0f172b;
        --border-color: #314158;
        --font-family: 'Space Grotesk', sans-serif;
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
    
    h2 {
        font-size: 1.5rem;
        font-weight: 400;
    }
    
    h3 {
        font-size: 1rem;
        font-weight: 400;
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
    
    /* Widget styling */
    .stSelectbox, .stTextInput, .stRadio {
        font-family: var(--font-family);
    }
    
    .stSelectbox > div > div {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        color: var(--text-color);
    }
    
    .stTextInput > div > div > input {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        color: var(--text-color);
        font-family: var(--font-family);
    }
    
    /* Metric styling */
    .css-1xarl3l {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--secondary-background-color);
        border-bottom: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-color);
        font-family: var(--font-family);
        background-color: transparent;
        border: 1px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
        border-radius: 6px 6px 0 0;
    }
    
    /* Success/Error/Warning styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px;
        border: 1px solid var(--border-color);
        font-family: var(--font-family);
    }
    
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.1);
        border-color: #22c55e;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1);
        border-color: #ef4444;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1);
        border-color: #f59e0b;
    }
    
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1);
        border-color: #3b82f6;
    }
    
    /* Progress bar styling */
    .stProgress .css-1cpxqw2 {
        background-color: var(--secondary-background-color);
        border-radius: 8px;
    }
    
    .stProgress .css-1cpxqw2 .css-1eynrej {
        background-color: var(--primary-color);
        border-radius: 8px;
    }
    
    /* Container styling */
    .css-1kyxreq {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
    }

    /* Custom emoji styling for better visibility */
    .emoji {
        filter: brightness(1.2);
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="E-commerce Market Analyzer", page_icon="🛍️")

# Custom title
st.markdown("""
    <h1 style="text-align: center; margin-bottom: 0.5rem;">
        🛍️ E-commerce Market Analyzer
    </h1>
    <p style="text-align: center; color: #94a3b8; font-family: 'Space Grotesk', sans-serif; margin-bottom: 2rem; font-size: 1.1rem;">
        Discover <strong style="color: #615fff;">real-time market gaps, trends, and high-selling products</strong> using Tavily and Gemini.
    </p>
""", unsafe_allow_html=True)

# Initialize session state
if "analysis_triggered" not in st.session_state:
    st.session_state.analysis_triggered = False
if "result" not in st.session_state:
    st.session_state.result = None

# Sidebar
with st.sidebar:
    st.markdown("""
        <h2 style="color: #615fff; border-bottom: 2px solid #314158; padding-bottom: 0.5rem; margin-bottom: 1rem;">
            ⚙️ Configuration
        </h2>
    """, unsafe_allow_html=True)
    
    platform = st.selectbox("🏪 Platform", ["Amazon", "eBay", "Walmart"])
    country = st.selectbox("🌍 Country", ["US", "UK", "DE", "JP"])
    category = st.text_input("🏷️ Product Category/Keywords", "smart home devices")
    analysis_type = st.radio("📊 Analysis Type", ["Market Gap", "Trending Products", "High Selling Products", "Competitor Analysis"])
    time_range = st.select_slider("⏰ Time Range", options=["Last Week", "Last Month", "Last 3 Months", "Last 6 Months"], value="Last Month")

    if st.button("🔍 Analyze Market", type="primary", use_container_width=True):
        st.session_state.analysis_triggered = True
        st.session_state.params = {
            "platform": platform,
            "country": country,
            "category": category,
            "analysis_type": analysis_type,
            "time_range": time_range,
        }

# Main analysis
if st.session_state.analysis_triggered:
    params = st.session_state.params
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        progress_bar.progress(20)
        status_text.markdown("🔍 **Searching for market data...**")
        user_query = f"Perform a '{params['analysis_type']}' analysis for '{params['category']}' on '{params['platform']}' in '{params['country']}' for '{params['time_range']}'."
        progress_bar.progress(40)
        status_text.markdown("🤖 **Processing with AI agents...**")
        result = agent_orchestrator({"question": user_query})
        progress_bar.progress(80)
        status_text.markdown("📊 **Generating visualizations...**")
        st.session_state.result = result
        progress_bar.progress(100)
        status_text.markdown("✅ **Analysis complete!**")
        progress_bar.empty()
        status_text.empty()
        st.success("🎉 Market analysis completed successfully!")
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error during analysis: {str(e)}")
    finally:
        st.session_state.analysis_triggered = False

# Results display
if st.session_state.result:
    result = st.session_state.result
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "📋 Data Tables", "🚀 Recommendations", "📥 Export"])

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Results", use_container_width=True):
                try:
                    save_results_tool(result)
                    st.success("✅ Results saved successfully!")
                except Exception as e:
                    st.error(f"❌ Error saving results: {str(e)}")
        with col2:
            if st.button("📂 Load Previous Results", use_container_width=True):
                try:
                    saved_result = load_results_tool()
                    if saved_result:
                        st.session_state.result = saved_result
                        st.success("✅ Previous results loaded!")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading results: {str(e)}")

        col1, col2, col3 = st.columns(3)
        with col1:
            result_json = json.dumps(result, indent=2, ensure_ascii=False)
            st.download_button("⬇️ Download Results (JSON)", data=result_json, file_name="analysis.json", mime="application/json", use_container_width=True)
        with col2:
            if result.get("tables"):
                df = pd.DataFrame(result["tables"][0])
                csv = df.to_csv(index=False)
                st.download_button("⬇️ Download Table (CSV)", data=csv, file_name="analysis.csv", mime="text/csv", use_container_width=True)
        with col3:
            if result.get("charts"):
                try:
                    fig = go.Figure(json.loads(result["charts"][0]))
                    img_bytes = fig.to_image(format="png")
                    st.download_button("⬇️ Download Chart (PNG)", data=img_bytes, file_name="chart.png", mime="image/png", use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Could not export chart: {str(e)}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1,1,1])

with col1:
    if st.button("🔄 New Analysis", use_container_width=True):
        st.session_state.clear()
        st.rerun()

with col2:
    if st.button("❓ Help", use_container_width=True):
        st.info("""
        **How to use:**
        1. Select platform, country, and category in the sidebar
        2. Choose analysis type and time range
        3. Click 'Analyze Market'
        4. Review insights, charts, and recommendations
        5. Export or save results as needed
        """)

with col3:
    status_color = "#22c55e" if not st.session_state.analysis_triggered else "#f59e0b"
    status_text = "Ready" if not st.session_state.analysis_triggered else "Processing..."
    st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; background-color: #0f172b; border: 1px solid #314158; border-radius: 6px;">
            <span style="color: {status_color}; font-weight: 500;">● {status_text}</span>
        </div>
    """, unsafe_allow_html=True)
