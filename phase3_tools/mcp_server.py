import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path)

from phase3_tools.tools import search_wikipedia, scrape_webpage

app = FastAPI(
    title="AI Research Agent - Simulated MCP Tool Server",
    description="Exposes local tools via HTTP JSON-RPC for external or local LLM execution",
    version="1.0.0"
)

# Request schema for tool calls
class ToolCallRequest(BaseModel):
    arguments: dict

# Tool registry metadata (simulating MCP schemas)
TOOL_METADATA = [
    {
        "name": "search_wikipedia",
        "description": "Searches Wikipedia for a given query and returns a summary of the most relevant page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search term to lookup"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "scrape_webpage",
        "description": "Fetches a webpage from a URL and extracts its text content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The http/https link to scrape"}
            },
            "required": ["url"]
        }
    }
]

@app.get("/tools")
def list_tools():
    """Lists all tools available on this server (MCP list tools endpoint)."""
    return {"tools": TOOL_METADATA}

@app.post("/tools/{tool_name}/call")
def call_tool(tool_name: str, request: ToolCallRequest):
    """Executes a tool call with the provided arguments (MCP call tool endpoint)."""
    arguments = request.arguments
    
    if tool_name == "search_wikipedia":
        if "query" not in arguments:
            raise HTTPException(status_code=400, detail="Missing required argument: 'query'")
        result = search_wikipedia.invoke({"query": arguments["query"]})
        return {"content": [{"type": "text", "text": result}]}
        
    elif tool_name == "scrape_webpage":
        if "url" not in arguments:
            raise HTTPException(status_code=400, detail="Missing required argument: 'url'")
        result = scrape_webpage.invoke({"url": arguments["url"]})
        return {"content": [{"type": "text", "text": result}]}
        
    else:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

if __name__ == "__main__":
    import uvicorn
    print("Starting simulated MCP Tool Server...")
    print("API documentation available at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
