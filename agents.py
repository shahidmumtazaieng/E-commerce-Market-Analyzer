import datetime
import json
import os
import re
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any, List, TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnablePassthrough
import operator
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Using gemini-1.5-flash as gemini-2.5-flash is not available
        temperature=0.3,
        max_tokens=5048,
        max_retries=3,
    )
    print("✅ Google Gemini LLM initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize Gemini LLM: {e}")
    llm = None

# Initialize Tavily Tools
try:
    tavily_search = TavilySearch(max_results=10, topic="general")
    tavily_extract = TavilyExtract()
    print("✅ Tavily tools initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize Tavily: {e}")
    tavily_search = None
    tavily_extract = None

# Create results directory
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "last_result.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pydantic Models for Structured Output
class MarketGapProduct(BaseModel):
    product: str = Field(description="Product name")
    demand_score: float = Field(description="Demand score (0-10)")
    competition: str = Field(description="Competition level: Low, Medium, High")
    opportunity: str = Field(description="Opportunity level: Low, Medium, High")
    market_size: str = Field(description="Estimated market size, e.g., $1M")

class TrendingProduct(BaseModel):
    product: str = Field(description="Product name")
    trend_score: int = Field(description="Trend score (0-100)")
    growth: str = Field(description="Growth percentage, e.g., +200%")
    interest_level: str = Field(description="Interest level: Low, Medium, High, Very High")
    search_volume: str = Field(description="Monthly search volume, e.g., 100K/month")

class HighSellingProduct(BaseModel):
    product: str = Field(description="Product name")
    sales_rank: int = Field(description="Sales rank")
    revenue: str = Field(description="Estimated revenue, e.g., $2.5M")
    rating: float = Field(description="Average rating (0-5)")
    reviews: str = Field(description="Number of reviews, e.g., 12.5K")

class Competitor(BaseModel):
    competitor: str = Field(description="Competitor name")
    market_share: str = Field(description="Market share percentage, e.g., 32%")
    strength: str = Field(description="Key strength")
    weakness: str = Field(description="Key weakness")
    rating: str = Field(description="Average rating, e.g., 4.2/5")

class AnalysisResult(BaseModel):
    summary: str = Field(description="Summary of the analysis")
    tables: List[List[Dict]] = Field(description="List of data tables")
    charts: List[str] = Field(description="List of Plotly chart JSON strings")
    recommendations: str = Field(description="Actionable recommendations")

# LangChain Chains
def search_market_data(query: str) -> str:
    """Search for market data using Tavily"""
    if not tavily_search:
        return json.dumps({"results": [], "query": query, "response_time": 0.0})
    try:
        results = tavily_search.run(query)
        return json.dumps(results, indent=2)
    except Exception as e:
        print(f"⚠️ Tavily search error: {e}. Returning empty results.")
        return json.dumps({"results": [], "query": query, "response_time": 0.0})

search_tool = Tool(
    name="TavilySearch",
    func=search_market_data,
    description="Search for current market data, trends, and product information using Tavily. Input is a search query string."
)

extract_tool = Tool(
    name="TavilyExtract",
    func=tavily_extract.run,
    description="Extract insights from a given URL or long text using TavilyExtract. Input is a URL or text."
)

def process_market_data(raw_data: str, analysis_type: str, category: str, platform: str, region: str, time_period: str) -> List[Dict]:
    """Extract structured data using Pydantic models"""
    if not llm:
        raise ValueError("LLM not initialized.")

    # Select Pydantic model based on analysis_type
    if analysis_type.lower() == "market gap":
        Model = MarketGapProduct
        analysis_desc = "market gap analysis, identifying products with high demand but low competition"
        table_fields = "product, demand_score, competition, opportunity, market_size"
    elif analysis_type.lower() == "trending products":
        Model = TrendingProduct
        analysis_desc = "trending products analysis, identifying products with high growth and interest"
        table_fields = "product, trend_score, growth, interest_level, search_volume"
    elif analysis_type.lower() == "high selling products":
        Model = HighSellingProduct
        analysis_desc = "high selling products analysis, identifying top sellers with sales rank, revenue, ratings"
        table_fields = "product, sales_rank, revenue, rating, reviews"
    elif analysis_type.lower() == "competitor analysis":
        Model = Competitor
        analysis_desc = "competitor analysis, identifying key competitors with market share, strengths, weaknesses"
        table_fields = "competitor, market_share, strength, weakness, rating"
    else:
        raise ValueError("Unsupported analysis type")

    parser = JsonOutputParser(pydantic_object=Model)
    extract_prompt = ChatPromptTemplate.from_template(
        f"Extract a list of 4-6 entries for {analysis_desc} from the following search results for {category} on {platform} in {region} for {time_period}.\n"
        f"Search results: {{raw_data}}\n\n"
        f"Output as a JSON list of objects with fields: {table_fields}.\n"
        f"Estimate numerical values if not explicitly stated, based on content.\n"
        f"Ensure output is a valid JSON array. If the data is incomplete, generate reasonable placeholders.\n"
        f"Format instructions: {{format_instructions}}"
    )

    extract_chain = (
        {"raw_data": RunnablePassthrough(), "format_instructions": lambda _: parser.get_format_instructions()}
        | extract_prompt
        | llm
        | parser
    )

    try:
        extracted_data = extract_chain.invoke(raw_data)
        if not isinstance(extracted_data, list) or len(extracted_data) == 0:
            print(f"⚠️ Extracted data is empty or invalid. Returning fallback data.")
            extracted_data = [
                {
                    table_fields.split(", ")[0]: f"{category}",
                    table_fields.split(", ")[1]: 5.0 if "score" in table_fields else 1,
                    table_fields.split(", ")[2]: "Medium",
                    table_fields.split(", ")[3]: "Medium" if "opportunity" in table_fields else "",
                    table_fields.split(", ")[4]: "$0.5M" if "market_size" in table_fields else "1K/month"
                }
            ]
    except Exception as e:
        print(f"Extraction error: {e}. Returning fallback data.")
        extracted_data = [
            {
                table_fields.split(", ")[0]: f"{category} Placeholder",
                table_fields.split(", ")[1]: 5.0 if "score" in table_fields else 1,
                table_fields.split(", ")[2]: "Medium",
                table_fields.split(", ")[3]: "Medium" if "opportunity" in table_fields else "Placeholder",
                table_fields.split(", ")[4]: "$0.5M" if "market_size" in table_fields else "1K/month"
            }
        ]

    return extracted_data

def generate_analysis(data: List[Dict], analysis_type: str) -> Dict[str, Any]:
    """Generate summary and recommendations"""
    if not llm:
        raise ValueError("LLM not initialized.")

    summary_prompt = ChatPromptTemplate.from_template(
        f"Generate a concise summary for {analysis_type} analysis based on: {{data}}.\n"
        f"Highlight key findings."
    )
    summary_chain = summary_prompt | llm | StrOutputParser()

    rec_prompt = ChatPromptTemplate.from_template(
        f"Generate 4-5 actionable recommendations for {analysis_type} based on: {{data}}.\n"
        f"Format as a numbered list."
    )
    rec_chain = rec_prompt | llm | StrOutputParser()

    try:
        summary = summary_chain.invoke({"data": json.dumps(data, indent=2)})
        recommendations = rec_chain.invoke({"data": json.dumps(data, indent=2)})
    except Exception as e:
        print(f"Analysis generation error: {e}. Using fallback.")
        summary = f"Analysis for {analysis_type} completed with limited data."
        recommendations = "1. Refine search query for more specific results.\n2. Check API connectivity.\n3. Try a different time period.\n4. Consider alternative platforms."

    return {"summary": summary, "recommendations": recommendations}

def create_analysis_chart(data: List[Dict], analysis_type: str) -> str:
    """Create Plotly chart"""
    try:
        if not data:
            return ""

        product_names = [item["product"] if "product" in item else item["competitor"] for item in data]
        
        if analysis_type.lower() == "market gap":
            y_values = [item["demand_score"] for item in data]
            colors = ['#ff4444' if item["opportunity"] == "High" else '#ffaa44' if item["opportunity"] == "Medium" else '#44ff44' for item in data]
            title = "Market Gap Analysis - Demand vs Opportunity"
            y_title = "Demand Score"
        elif analysis_type.lower() == "trending products":
            y_values = [item["trend_score"] for item in data]
            colors = ['#ff4444' if item["interest_level"] in ["High", "Very High"] else '#ffaa44' if item["interest_level"] == "Medium" else '#44ff44' for item in data]
            title = "Trending Products Analysis - Trend Score"
            y_title = "Trend Score"
        elif analysis_type.lower() == "high selling products":
            y_values = [item["rating"] for item in data]
            colors = ['#ff4444' if item["rating"] >= 4.5 else '#ffaa44' if item["rating"] >= 4.0 else '#44ff44' for item in data]
            title = "High Selling Products Analysis - Rating"
            y_title = "Rating"
        elif analysis_type.lower() == "competitor analysis":
            y_values = [float(item["rating"].split('/')[0]) if '/' in item["rating"] else float(item["rating"]) for item in data]
            colors = ['#ff4444' if y >= 4.5 else '#ffaa44' if y >= 4.0 else '#44ff44' for y in y_values]
            title = "Competitor Analysis - Rating"
            y_title = "Rating"
        else:
            return ""

        fig = go.Figure(data=[
            go.Bar(
                x=product_names,
                y=y_values,
                marker_color=colors,
                text=y_values,
                textposition='auto',
                name=y_title
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="Items",
            yaxis_title=y_title,
            yaxis=dict(range=[0, max(y_values) * 1.1]),
            showlegend=False,
            template="plotly",
            font=dict(size=12),
            margin=dict(l=50, r=50, t=80, b=100)
        )

        return fig.to_json()
    except Exception as e:
        print(f"Chart creation error: {e}")
        return ""

# Multi-Agent with LangGraph
class AgentState(TypedDict):
    messages: Annotated[List[Dict], operator.add]
    next: str
    raw_data: str
    extracted_data: List[Dict]
    analysis: Dict[str, Any]
    chart: str
    analysis_type: str
    category: str
    platform: str
    region: str
    time_period: str
    iteration_count: int  # Added to track iterations

def search_node(state: AgentState) -> AgentState:
    """Search agent node"""
    query = f"{state['category']} {state['analysis_type']} {state['platform']} market analysis trends 2024 {state['region']} {state['time_period']}"
    raw_data = search_market_data(query)
    return {"raw_data": raw_data, "next": "extract"}

def extract_node(state: AgentState) -> AgentState:
    """Extract agent node"""
    extracted_data = process_market_data(
        state["raw_data"],
        state["analysis_type"],
        state["category"],
        state["platform"],
        state["region"],
        state["time_period"]
    )
    return {"extracted_data": extracted_data, "next": "analyze"}

def analyze_node(state: AgentState) -> AgentState:
    """Analysis agent node"""
    analysis = generate_analysis(state["extracted_data"], state["analysis_type"])
    return {"analysis": analysis, "next": "visualize"}

def visualize_node(state: AgentState) -> AgentState:
    """Visualization agent node"""
    chart = create_analysis_chart(state["extracted_data"], state["analysis_type"])
    return {"chart": chart, "next": "supervisor"}

def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor node to determine next step"""
    # Initialize or increment iteration counter
    iteration_count = state.get("iteration_count", 0) + 1
    
    # Force exit after 100 iterations to prevent infinite loops
    if iteration_count >= 100:
        print(f"⚠️ Max iterations (100) reached. Forcing workflow to FINISH.")
        return {"next": "FINISH", "iteration_count": iteration_count}

    if not state.get("raw_data"):
        return {"next": "search", "iteration_count": iteration_count}
    elif not state.get("extracted_data"):
        return {"next": "extract", "iteration_count": iteration_count}
    elif not state.get("analysis"):
        return {"next": "analyze", "iteration_count": iteration_count}
    elif not state.get("chart"):
        return {"next": "visualize", "iteration_count": iteration_count}
    return {"next": "FINISH", "iteration_count": iteration_count}

# LangGraph Workflow
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("search", search_node)
graph.add_node("extract", extract_node)
graph.add_node("analyze", analyze_node)
graph.add_node("visualize", visualize_node)

graph.add_edge("search", "supervisor")
graph.add_edge("extract", "supervisor")
graph.add_edge("analyze", "supervisor")
graph.add_edge("visualize", "supervisor")
graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"search": "search", "extract": "extract", "analyze": "analyze", "visualize": "visualize", "FINISH": END}
)
graph.set_entry_point("supervisor")

workflow = graph.compile()

def save_results_tool(data: Dict[str, Any]) -> str:
    """Save analysis results to file"""
    try:
        data["timestamp"] = datetime.datetime.now().isoformat()
        data["version"] = "1.0"
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return "Results saved successfully"
    except Exception as e:
        print(f"Save error: {e}")
        return f"Error saving results: {str(e)}"

def load_results_tool() -> Dict[str, Any]:
    """Load saved analysis results"""
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                required_keys = ["summary", "tables", "charts", "recommendations"]
                for key in required_keys:
                    if key not in data:
                        data[key] = [] if key in ["tables", "charts"] else ""
                return data
        else:
            return {
                "summary": "No saved results found. Run an analysis first.",
                "tables": [],
                "charts": [],
                "recommendations": "Start by running a market analysis query."
            }
    except Exception as e:
        return {
            "summary": f"Error loading results: {str(e)}",
            "tables": [],
            "charts": [],
            "recommendations": "Check file permissions and try again."
        }

def agent_orchestrator(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Main orchestrator function using LangGraph"""
    try:
        question = inputs.get("question", "")
        
        if "load" in question.lower() and "result" in question.lower():
            return load_results_tool()
        
        # Parse query parameters
        analysis_type = "general"
        category = "products"
        platform = "Amazon"
        region = "US"
        time_period = "Last Month"
        
        if "market gap" in question.lower():
            analysis_type = "market gap"
        elif "trending" in question.lower():
            analysis_type = "trending products"
        elif "high selling" in question.lower():
            analysis_type = "high selling products"
        elif "competitor" in question.lower():
            analysis_type = "competitor analysis"
        
        category_match = re.search(r"for ['\"](.*?)['\"]", question)
        if category_match:
            category = category_match.group(1)
        
        platform_match = re.search(r"on ['\"](.*?)['\"]", question)
        if platform_match:
            platform = platform_match.group(1)
        
        region_match = re.search(r"in ['\"](.*?)['\"]", question)
        if region_match:
            region = region_match.group(1)
        
        time_period_match = re.search(r"for ['\"]([^']*?)['\"]\.", question)
        if time_period_match:
            time_period = time_period_match.group(1)

        # Initialize state
        state = {
            "messages": [],
            "next": "search",
            "raw_data": "",
            "extracted_data": [],
            "analysis": {},
            "chart": "",
            "analysis_type": analysis_type,
            "category": category,
            "platform": platform,
            "region": region,
            "time_period": time_period,
            "iteration_count": 0  # Initialize iteration counter
        }

        # Run workflow
        final_state = workflow.invoke(state, recursion_limit=5000)  # Increased to 5000
        
        result = {
            "summary": final_state["analysis"].get("summary", "Analysis completed with limited data."),
            "tables": [final_state["extracted_data"]],
            "charts": [final_state["chart"]] if final_state["chart"] else [],
            "recommendations": final_state["analysis"].get("recommendations", ""),
            "status": "✅ Analysis complete and saved to results folder"
        }

        save_results_tool(result)
        return result

    except Exception as e:
        error_result = {
            "summary": f"Analysis encountered an error: {str(e)}. Please try again with different parameters.",
            "tables": [],
            "charts": [],
            "recommendations": "Check your API keys and network connection. Try simplifying your query.",
            "status": "❌ Analysis failed"
        }
        save_results_tool(error_result)

        return error_result
