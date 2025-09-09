import streamlit as st
import pandas as pd
import json
from typing import Dict, Any
import plotly.graph_objects as go
from agents import agent_orchestrator, load_results_tool, save_results_tool

# Page config (set early)
st.set_page_config(layout="wide", page_title="E-commerce Market Analyzer", page_icon="🛍️")

# Custom CSS for the dark theme and unified button styling
st.markdown("""
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
            font-size: 1.8rem;
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
</style>
""", unsafe_allow_html=True)

# Custom title with enhanced styling
st.markdown("""
    <h1 style="text-align: center; margin-bottom: 0.5rem;">
        🛍️ E-commerce Market Analyzer
    </h1>
    <p style="text-align: center; color: #94a3b8; font-family: 'Space Grotesk', sans-serif; margin-bottom: 2rem; font-size: 1.1rem;">
        Discover <strong style="color: #615fff;">real-time market gaps, trends, and high-selling products</strong> using TTS Sir Buland GPT.
    </p>
""", unsafe_allow_html=True)

# Initialize session state
if "analysis_triggered" not in st.session_state:
    st.session_state.analysis_triggered = False
if "result" not in st.session_state:
    st.session_state.result = None

# Enhanced sidebar with custom styling
with st.sidebar:
    st.markdown("""
        <h2 style="color: #615fff; border-bottom: 2px solid #314158; padding-bottom: 0.5rem; margin-bottom: 1rem;">
            ⚙️ Configuration
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
        "🏷️ Product Category/Keywords",
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

# Enhanced results display
if st.session_state.result:
    result = st.session_state.result

    st.markdown("---")

    # Enhanced Key Insights Section
    st.markdown("""
        <h2 style="color: #615fff; border-left: 4px solid #615fff; padding-left: 1rem; margin-bottom: 1rem;">
            💡 Key Insights
        </h2>
    """, unsafe_allow_html=True)

    if result.get("summary"):
        st.markdown(f"""
            <div style="background-color: #0f172b; border: 1px solid #314158; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem;">
                <strong style="color: #615fff;">Analysis Summary:</strong><br>
                <span style="line-height: 1.6;">{result["summary"]}</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No summary insights available.")

    # Enhanced tabs with custom styling
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "📋 Data Tables", "🚀 Recommendations", "📥 Export"])

    with tab1:
        st.markdown("""
            <h3 style="color: #615fff; margin-bottom: 1rem;">📊 Market Visualizations</h3>
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
                        title_font_color='#615fff'
                    )

                    # Customize caption based on analysis type
                    analysis_type = st.session_state.get('params', {}).get('analysis_type', 'Market Analysis')
                    if analysis_type == "Market Gap":
                        caption = f"Chart {idx + 1}: Demand vs. Opportunity for {st.session_state.get('params', {}).get('category', 'Products')}"
                    elif analysis_type == "Trending Products":
                        caption = f"Chart {idx + 1}: Trend Score for {st.session_state.get('params', {}).get('category', 'Products')}"
                    elif analysis_type == "High Selling Products":
                        caption = f"Chart {idx + 1}: Rating for Top Selling {st.session_state.get('params', {}).get('category', 'Products')}"
                    elif analysis_type == "Competitor Analysis":
                        caption = f"Chart {idx + 1}: Competitor Ratings for {st.session_state.get('params', {}).get('category', 'Products')}"
                    else:
                        caption = f"Chart {idx + 1}: Market Analysis Visualization"

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={'displayModeBar': True, 'staticPlot': False},
                    )
                    st.caption(caption)
                except Exception as e:
                    st.warning(f"⚠️ Could not display chart {idx + 1}: {str(e)}")
        else:
            st.info("📈 No charts were generated for this analysis.")
            st.markdown("*Charts will be generated based on available market data. Try refining your query.*")

    with tab2:
        st.markdown("""
            <h3 style="color: #615fff; margin-bottom: 1rem;">📋 Detailed Data Tables</h3>
        """, unsafe_allow_html=True)

        if result.get("tables") and len(result["tables"]) > 0:
            for idx, table_data in enumerate(result["tables"]):
                st.markdown(f"""
                    <h4 style="color: #94a3b8; margin-bottom: 0.5rem;">
                        Table {idx + 1}: {st.session_state.get('params', {}).get('analysis_type', 'Market Analysis')}
                    </h4>
                """, unsafe_allow_html=True)

                try:
                    if isinstance(table_data, list) and len(table_data) > 0:
                        df = pd.DataFrame(table_data)
                        # Rename columns for display
                        analysis_type = st.session_state.get('params', {}).get('analysis_type', 'Market Analysis')
                        if analysis_type == "Market Gap":
                            df.columns = ["Product", "Demand Score", "Competition", "Opportunity", "Market Size"]
                        elif analysis_type == "Trending Products":
                            df.columns = ["Product", "Trend Score", "Growth", "Interest Level", "Search Volume"]
                        elif analysis_type == "High Selling Products":
                            df.columns = ["Product", "Sales Rank", "Revenue", "Rating", "Reviews"]
                        elif analysis_type == "Competitor Analysis":
                            df.columns = ["Competitor", "Market Share", "Strength", "Weakness", "Rating"]

                        # Enhanced styling for numeric columns
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        styled_df = df.style.highlight_max(axis=0, subset=numeric_cols).format(precision=2, subset=numeric_cols)
                        st.dataframe(styled_df, use_container_width=True)
                    else:
                        st.json(table_data)
                except Exception as e:
                    st.warning(f"⚠️ Could not display table {idx + 1}: {str(e)}")
                    st.json(table_data)
        else:
            st.info("📊 No data tables available for this analysis.")
            st.markdown("*Ensure sufficient data is available from the search query.*")

    with tab3:
        st.markdown("""
            <h3 style="color: #615fff; margin-bottom: 1rem;">🚀 Strategic Recommendations</h3>
        """, unsafe_allow_html=True)

        if result.get("recommendations"):
            st.markdown(f"""
                <div style="background-color: #0f172b; border: 1px solid #314158; border-radius: 8px; padding: 1.5rem;">
                    <strong style="color: #615fff;">Actionable Insights:</strong><br>
                    <span style="line-height: 1.6;">{result["recommendations"]}</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No specific recommendations generated.")

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

# Enhanced footer with controls
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">
            Powered by AI • developed by ❤️ <a href="https://github.com/shahidmumtazaieng/">Shahid Mumtaz</a>
        </p>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

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
    status_color = "#22c55e" if not st.session_state.analysis_triggered else "#f59e0b"
    status_text = "Ready" if not st.session_state.analysis_triggered else "Processing..."
    st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; background-color: #0f172b; border: 1px solid #314158; border-radius: 6px;">
            <span style="color: {status_color}; font-weight: 500;">● {status_text}</span>
        </div>
    """, unsafe_allow_html=True)



