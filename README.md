"# E-commerce-Market-Analyzer" 
# Tavily Analytic Agents

This project provides a multi-agent workflow for market analysis using LangChain, LangGraph, Google Gemini, and Tavily APIs. It extracts, analyzes, and visualizes market data for products, trends, competitors, and more, saving results in structured JSON format.

## Features

- **Multi-agent workflow**: Search, extract, analyze, and visualize market data.
- **LLM-powered extraction**: Uses Google Gemini for structured data extraction.
- **Tavily integration**: Fetches real-time market data and insights.
- **Structured output**: Results saved as JSON, including tables, charts, summaries, and recommendations.
- **Customizable analysis**: Supports market gap, trending products, high selling products, and competitor analysis.
- **Plotly charts**: Generates interactive charts for visual insights.

## Requirements

- Python 3.8+
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Google Gemini API](https://github.com/langchain-ai/langchain-google-genai)
- [Tavily API](https://github.com/tavily/tavily-python)
- [Plotly](https://plotly.com/python/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- Other dependencies: pandas, pydantic

## Setup

1. **Clone the repository**  
   ```
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

3. **Configure environment variables**  
   - Create a `.env` file with your Google Gemini and Tavily API keys:
     ```
     GEMINI_API_KEY=your_gemini_api_key
     TAVILY_API_KEY=your_tavily_api_key
     ```

## Usage

Run the main orchestrator function with a query:

```python
from agents import agent_orchestrator

inputs = {"question": "Show market gap analysis for 'wireless earbuds' on 'Amazon' in 'US' for 'Last Month'."}
result = agent_orchestrator(inputs)
print(result)
```

- Results are saved in the `results/last_result.json` file.
- To load the last saved results:
  ```python
  from agents import load_results_tool
  data = load_results_tool()
  print(data)
  ```

## Customization

- **Increase workflow steps**: Adjust `max_steps` in `workflow.invoke(state, max_steps=...)` for deeper recursion.
- **Add new analysis types**: Extend Pydantic models and update the workflow logic.

## File Structure

- `agents.py` — Main workflow and agent logic.
- `results/last_result.json` — Saved analysis results.

## License

MIT License

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Google Gemini](https://ai.google.dev/)
- [Tavily](https://www.tavily.com/)
