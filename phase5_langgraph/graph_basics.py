import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path)

# 1. Define State
class GraphState(TypedDict):
    query: str
    category: str  # "research" or "chat"
    response: str

# 2. Define Nodes
def category_classifier_node(state: GraphState) -> dict:
    """Classifies the user query into either 'research' or 'chat'."""
    print("[Node] category_classifier_node")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the user's query into one of two categories: 'research' (requires factual searching) or 'chat' (general conversation/greeting/opinion). Reply with ONLY the category word: 'research' or 'chat'."),
        ("user", "{query}")
    ])
    
    classifier_chain = prompt | llm
    category = classifier_chain.invoke({"query": state["query"]}).content.strip().lower()
    
    # Simple validation fallback
    if "research" in category:
        category = "research"
    else:
        category = "chat"
        
    print(f"   Query Classified As: {category.upper()}")
    return {"category": category}

def chat_node(state: GraphState) -> dict:
    """Handles general greetings and chit-chat."""
    print("[Node] chat_node")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly chatbot. Provide a warm, conversational reply."),
        ("user", "{query}")
    ])
    response = (prompt | llm).invoke({"query": state["query"]}).content
    return {"response": response}

def research_fallback_node(state: GraphState) -> dict:
    """Simulates starting a search query response."""
    print("[Node] research_fallback_node")
    return {"response": f"This is a factual query about '{state['query']}' that requires searching. (In Phase 6, we will route this to a real web research workflow!)"}

# 3. Define Conditional Edge Router
def route_by_category(state: GraphState) -> Literal["chat_node", "research_node"]:
    """Routes execution based on the category classified."""
    if state["category"] == "research":
        return "research_node"
    return "chat_node"

def build_graph():
    # Initialize the graph with State
    workflow = StateGraph(GraphState)
    
    # Add Nodes
    workflow.add_node("classifier", category_classifier_node)
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("research_node", research_fallback_node)
    
    # Set Entry Point
    workflow.add_edge(START, "classifier")
    
    # Add Conditional Edges
    workflow.add_conditional_edges(
        "classifier",
        route_by_category,
        {
            "chat_node": "chat_node",
            "research_node": "research_node"
        }
    )
    
    # Add End Edges
    workflow.add_edge("chat_node", END)
    workflow.add_edge("research_node", END)
    
    # Compile the graph
    app = workflow.compile()
    return app

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in environment.")
    else:
        graph = build_graph()
        
        # Test Case 1: Chat query
        print("\n--- Running Test 1: Chat Query ---")
        input1 = {"query": "Hello there! How was your day?"}
        result1 = graph.invoke(input1)
        print(f"Result: {result1['response']}")
        
        # Test Case 2: Research query
        print("\n--- Running Test 2: Research Query ---")
        input2 = {"query": "What is the distance between Earth and Mars?"}
        result2 = graph.invoke(input2)
        print(f"Result: {result2['response']}")
