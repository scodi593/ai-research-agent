import os
from typing import TypedDict, Literal, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Import search tool
from phase3_tools.tools import search_wikipedia

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path)

# 1. State Definition
class ResearchState(TypedDict):
    topic: str
    research_notes: str
    draft: str
    feedback: str
    revision_count: int
    final_report: str
    current_agent: str

# 2. Researcher Node
def researcher_node(state: ResearchState) -> dict:
    print("\n[Agent] Researcher Agent")
    topic = state["topic"]
    
    # Run wikipedia tool to gather background info
    print(f"   Searching Wikipedia for: '{topic}'...")
    search_results = search_wikipedia.invoke({"query": topic})
    
    # Use LLM to summarize key findings from research
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research expert. Analyze the search results and extract key factual notes relevant to the topic: {topic}. Include names, dates, key concepts, and statistics if available."),
        ("user", "Search Results:\n{results}")
    ])
    
    notes = (prompt | llm).invoke({"topic": topic, "results": search_results}).content
    print("   Research complete. Notes compiled.")
    return {
        "research_notes": notes,
        "current_agent": "Researcher"
    }

# 3. Writer Node
def writer_node(state: ResearchState) -> dict:
    print("\n[Agent] Writer Agent")
    topic = state["topic"]
    notes = state["research_notes"]
    feedback = state.get("feedback", "")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
    
    if feedback:
        # Re-drafting based on feedback
        print("   Incorporating feedback and revising draft...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional technical writer. Revise the original draft based on the feedback provided. Use the research notes to fill in any gaps."),
            ("user", "Topic: {topic}\n\nResearch Notes:\n{notes}\n\nOriginal Draft:\n{draft}\n\nFeedback for Revision:\n{feedback}")
        ])
        draft = (prompt | llm).invoke({
            "topic": topic,
            "notes": notes,
            "draft": state["draft"],
            "feedback": feedback
        }).content
    else:
        # Initial drafting
        print("   Creating initial report draft...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional technical writer. Synthesize the provided research notes into a comprehensive, beautifully structured Markdown report about: {topic}. The report should include an Introduction, Key Findings/Sections, and a Conclusion."),
            ("user", "Research Notes:\n{notes}")
        ])
        draft = (prompt | llm).invoke({"topic": topic, "notes": notes}).content
        
    print("   Draft complete.")
    return {
        "draft": draft,
        "feedback": "", # Clear feedback after handling
        "current_agent": "Writer"
    }

# 4. Reviewer Node
def reviewer_node(state: ResearchState) -> dict:
    print("\n[Agent] Reviewer Agent")
    draft = state["draft"]
    revision_count = state.get("revision_count", 0)
    
    # If we have reached the limit of revisions, just approve it
    if revision_count >= 1:
        print("   Maximum revisions reached. Approving report.")
        return {
            "final_report": draft,
            "feedback": "APPROVED",
            "current_agent": "Reviewer"
        }
        
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an editor checking a research draft. Evaluate if it contains details and is ready. If it requires improvements, write a concise critique. If it is already excellent and complete, write 'APPROVED'."),
        ("user", "Draft to review:\n{draft}")
    ])
    
    review_output = (prompt | llm).invoke({"draft": draft}).content.strip()
    
    if "APPROVED" in review_output.upper():
        print("   Draft approved by Reviewer!")
        return {
            "final_report": draft,
            "feedback": "APPROVED",
            "current_agent": "Reviewer"
        }
    else:
        print(f"   Draft rejected. Feedback compiled: {review_output[:100]}...")
        return {
            "feedback": review_output,
            "revision_count": revision_count + 1,
            "current_agent": "Reviewer"
        }

# 5. Routing Logic
def route_review(state: ResearchState) -> Literal["writer_node", "end"]:
    """Determines whether to revise the draft or finalize the report."""
    if state["feedback"] == "APPROVED":
        return "end"
    return "writer_node"

def build_research_graph():
    # Initialize state graph
    workflow = StateGraph(ResearchState)
    
    # Add Nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)
    
    # Set Entry and Connections
    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")
    
    # Add Router
    workflow.add_conditional_edges(
        "reviewer",
        route_review,
        {
            "writer_node": "writer",
            "end": END
        }
    )
    
    app = workflow.compile()
    return app

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in environment.")
    else:
        graph = build_research_graph()
        initial_state = {
            "topic": "Generative Adversarial Networks (GANs)",
            "research_notes": "",
            "draft": "",
            "feedback": "",
            "revision_count": 0,
            "final_report": "",
            "current_agent": ""
        }
        
        print("Running Multi-Agent Research Graph...")
        # Stream the nodes execution
        for event in graph.stream(initial_state):
            for node, state_update in event.items():
                print(f"\n[Completed Node: {node}]")
                if "final_report" in state_update and state_update["final_report"]:
                    print("\n--- FINAL REPORT ---")
                    print(state_update["final_report"])
