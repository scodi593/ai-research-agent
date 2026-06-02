from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from langgraph.graph import StateGraph
# pyrefly: ignore [missing-import]
from duckduckgo_search import DDGS

load_dotenv()

print("Testing Groq connection...")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

response = llm.invoke([HumanMessage(content="Say hello in one sentence.")])
print("LLM response:", response.content)
print()

print("Testing LangGraph import...")

print("LangGraph: OK")

print()
print("Testing DuckDuckGo search...")

with DDGS() as ddgs:
    results = list(ddgs.text("LangChain", max_results=1))

    print("Results:", results)

    if results:
        print("Search: OK -", results[0]["title"])
    else:
        print("No search results returned")

print()
print("All checks passed. Ready for Phase 2!")