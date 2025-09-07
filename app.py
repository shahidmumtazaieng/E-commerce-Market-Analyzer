import streamlit as st
import pandas as pd
import json
from typing import Dict, Any
import plotly.graph_objects as go
from agents import agent_orchestrator, load_results_tool

st.set_page_config(layout="wide", page_title="E-commerce Market Analyzer")

st.title("🛍️ E-commerce Market Analyzer")
st.markdown("Discover **real-time market gaps, trends, and high-selling products** using Tavily and Gemini.")

# Initialize session state
if "analysis_triggered" not in st.session_state:
    st.session_state.analysis_triggered = False
if "result" not in st.session_state:
    st.session_state.result = None

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    platform = st.selectbox(
        "Platform",
        ["Amazon", "eBay", "Walmart"],
        help="Select the e-commerce platform to analyze."
    )
    country = st.selectbox(
        "Country",
        ["US", "UK", "DE", "JP"],
        help="Choose the country for market analysis."
    )
    category = st.text_input(
        "Product Category/Keywords",
        "smart home devices",
        help="Enter product category or keywords (e.g., 'smart home devices')."
    )
    analysis_type = st.radio(
        "Analysis Type",
        ["Market Gap", "Trending Products", "High Selling Products", "Competitor Analysis"],
        help="Select the type of analysis to perform."
    )
    time_range = st.select_slider(
        "Time Range",
        options=["Last Week", "Last Month", "Last 3 Months", "Last 6 Months"],
        value="Last Month",
        help="Choose the time frame for the analysis."
    )

    # Analyze button
    if st.button("🔍 Analyze Market", type="primary"):
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
    
    # Display analysis parameters
    st.subheader(f"🔎 Analyzing: {params['category']}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Platform", params['platform'])
    with col2:
        st.metric("Country", params['country'])
    with col3:
        st.metric("Analysis Type", params['analysis_type'])
    with col4:
        st.metric("Time Range", params['time_range'])
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Update progress
        progress_bar.progress(20)
        status_text.text("🔍 Searching for market data...")
        
        # Create query for the agent
        user_query = (
            f"Perform a '{params['analysis_type']}' analysis for '{params['category']}' "
            f"on '{params['platform']}' in '{params['country']}' for '{params['time_range']}'. "
            f"Provide detailed insights, data tables, and visualizations."
        )
        
        progress_bar.progress(40)
        status_text.text("🤖 Processing with AI agents...")
        
        # Run the analysis
        result = agent_orchestrator({"question": user_query})
        
        progress_bar.progress(80)
        status_text.text("📊 Generating visualizations...")
        
        # Store results
        st.session_state.result = result
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
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

# Display results if available
if st.session_state.result:
    result = st.session_state.result
    
    st.markdown("---")
    
    # Key Insights Section
    st.header("💡 Key Insights")
    if result.get("summary"):
        st.markdown(f"**Analysis Summary:**")
        st.markdown(result["summary"])
    else:
        st.info("No summary insights available.")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "📋 Data Tables", "🚀 Recommendations", "📥 Export"])
    
    with tab1:
        st.subheader("📊 Market Visualizations")
        if result.get("charts") and len(result["charts"]) > 0:
            for idx, chart_json in enumerate(result["charts"]):
                try:
                    # Load Plotly chart from JSON
                    fig = go.Figure(json.loads(chart_json))
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
                        caption=caption
                    )
                except Exception as e:
                    st.warning(f"⚠️ Could not display chart {idx + 1}: {str(e)}")
        else:
            st.info("📈 No charts were generated for this analysis.")
            st.markdown("*Charts will be generated based on available market data. Try refining your query.*")
    
    with tab2:
        st.subheader("📋 Detailed Data Tables")
        if result.get("tables") and len(result["tables"]) > 0:
            for idx, table_data in enumerate(result["tables"]):
                st.markdown(f"**Table {idx + 1}: {st.session_state.get('params', {}).get('analysis_type', 'Market Analysis')}**")
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
                        # Style numeric columns
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
        st.subheader("🚀 Strategic Recommendations")
        if result.get("recommendations"):
            st.markdown("**Actionable Insights:**")
            st.markdown(result["recommendations"])
        else:
            st.info("No specific recommendations generated.")
    
    with tab4:
        st.subheader("📥 Export & Save Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Results", help="Save current analysis results"):
                try:
                    save_results_tool(result)
                    st.success("✅ Results saved successfully!")
                except Exception as e:
                    st.error(f"❌ Error saving results: {str(e)}")
        
        with col2:
            if st.button("📂 Load Previous Results", help="Load last saved analysis"):
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
        
        # Download JSON
        if result.get("summary"):
            result_json = json.dumps(result, indent=2, ensure_ascii=False)
            st.download_button(
                label="⬇️ Download Results (JSON)",
                data=result_json,
                file_name=f"market_analysis_{st.session_state.get('params', {}).get('category', 'unknown')}_{st.session_state.get('params', {}).get('platform', 'unknown')}.json",
                mime="application/json",
                help="Download analysis results as JSON file"
            )
        
        # Download Table as CSV
        if result.get("tables") and len(result["tables"]) > 0:
            df = pd.DataFrame(result["tables"][0])
            # Rename columns for export
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
            st.download_button(
                label="⬇️ Download Table (CSV)",
                data=csv,
                file_name=f"market_analysis_{st.session_state.get('params', {}).get('category', 'unknown')}_{analysis_type.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                help="Download data table as CSV file"
            )
        
        # Download Chart as PNG (if available)
        if result.get("charts") and len(result["charts"]) > 0:
            try:
                fig = go.Figure(json.loads(result["charts"][0]))
                img_bytes = fig.to_image(format="png")
                st.download_button(
                    label="⬇️ Download Chart (PNG)",
                    data=img_bytes,
                    file_name=f"market_analysis_chart_{st.session_state.get('params', {}).get('category', 'unknown')}_{analysis_type.lower().replace(' ', '_')}.png",
                    mime="image/png",
                    help="Download chart as PNG image"
                )
            except Exception as e:
                st.warning(f"⚠️ Could not generate chart for download: {str(e)}")

# Footer with controls
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🔄 New Analysis", help="Start a new market analysis"):
        st.session_state.clear()
        st.rerun()

with col2:
    if st.button("❓ Help", help="Show help information"):
        st.info("""
        **How to use:**
        1. Select platform, country, and category in the sidebar
        2. Choose analysis type and time range
        3. Click 'Analyze Market' to start
        4. Review insights, charts, and recommendations
        5. Export or save results as needed
        """)

with col3:
    st.metric("Status", "Ready" if not st.session_state.analysis_triggered else "Processing...")