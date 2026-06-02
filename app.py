import os
import streamlit as st
from dotenv import load_dotenv

# Set page configuration first before any other streamlit commands
st.set_page_config(
    page_title="AI Research Agent Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path)

from phase6_multiagent.research_graph import build_research_graph

# Custom CSS for custom premium styles (vibrant color accents, sleek UI cards)
st.markdown("""
<style>
    /* Premium modern gradients and styling */
    .report-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #FF4B4B 0%, #7F00FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .agent-card {
        border-radius: 12px;
        padding: 1.5rem;
        background-color: #0E1117;
        border: 1px solid #30363D;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        color: white;
        background-color: #7F00FF;
        margin-bottom: 0.5rem;
    }
    .success-badge {
        background-color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Main Application Layout
st.markdown("<h1 class='report-title'>🔬 AI Research Agent Studio</h1>", unsafe_allow_html=True)
st.markdown("##### Stateful Multi-Agent Cooperative Research Workflow powered by LangGraph & Groq")

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/nolan/128/bot.png", width=80)
st.sidebar.markdown("### Agent Configuration")

# Verify API connection
api_key = os.getenv("GROQ_API_KEY")
if api_key:
    st.sidebar.success("🔗 Connected to Groq API")
else:
    st.sidebar.error("❌ GROQ_API_KEY missing in .env")

model_option = st.sidebar.selectbox(
    "Choose Reasoning LLM",
    ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
    help="Select the model for agent decision making and report generation."
)

max_revisions = st.sidebar.slider(
    "Max Editor Revisions",
    min_value=0,
    max_value=2,
    value=1,
    help="Controls how many times the Reviewer Agent can request rewrites from the Writer Agent."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### How it works")
st.sidebar.info("""
1. **Researcher Agent** searches Wikipedia, gathers facts and compiles structured notes.
2. **Writer Agent** takes the notes and drafts a structured technical report in Markdown.
3. **Reviewer Agent** reads the draft and either approves it or sends feedback for revision.
4. **Final Report** is rendered with a download button!
""")

# Sample Topics helper
sample_topics = [
    "Quantum Cryptography and Security",
    "Evolution of Large Language Models",
    "Fusion Energy Progress in 2026",
    "Self-Driving Cars and Computer Vision"
]

selected_sample = st.selectbox(
    "Pick a sample topic or type below:",
    ["-- Custom --"] + sample_topics
)

# Topic input
if selected_sample != "-- Custom --":
    topic_input = st.text_input("Research Topic", value=selected_sample)
else:
    topic_input = st.text_input("Research Topic", placeholder="Enter the topic you want to research...")

# Run Research Button
run_button = st.button("🚀 Start Multi-Agent Research", type="primary", use_container_width=True)

# Main Workspace
if run_button:
    if not api_key:
        st.error("Please set GROQ_API_KEY in your .env file before running.")
    elif not topic_input.strip():
        st.warning("Please enter a valid research topic.")
    else:
        st.markdown("### 🏃 Agent Workspace")
        
        # Placeholders for live steps update
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Grid layout for agent cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            research_container = st.empty()
        with col2:
            writer_container = st.empty()
        with col3:
            reviewer_container = st.empty()
            
        final_report_container = st.empty()
        
        # Build state and run graph
        graph = build_research_graph()
        initial_state = {
            "topic": topic_input,
            "research_notes": "",
            "draft": "",
            "feedback": "",
            "revision_count": 0,
            "final_report": "",
            "current_agent": ""
        }
        
        try:
            # We stream graph execution
            # Nodes: 'researcher', 'writer', 'reviewer'
            step_count = 0
            for event in graph.stream(initial_state):
                for node, state_update in event.items():
                    step_count += 1
                    
                    if node == "researcher":
                        progress_bar.progress(33)
                        status_text.write("🕵️ **Researcher Agent** is analyzing sources and compiling notes...")
                        
                        notes_preview = state_update.get("research_notes", "")
                        research_container.markdown(f"""
                        <div class='agent-card'>
                            <span class='badge'>🕵️ Researcher</span>
                            <h4>Research Notes</h4>
                            <p style='font-size: 0.85rem; color: #8B949E;'>Compiled facts & search summaries.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        with research_container.container():
                            st.markdown("### 🕵️ Researcher Notes")
                            st.write(notes_preview)
                            
                    elif node == "writer":
                        progress_bar.progress(66)
                        status_text.write("✍️ **Writer Agent** is drafting the structured report...")
                        
                        draft_preview = state_update.get("draft", "")
                        writer_container.markdown(f"""
                        <div class='agent-card'>
                            <span class='badge' style='background-color:#007BFF;'>✍️ Writer</span>
                            <h4>Draft Report</h4>
                            <p style='font-size: 0.85rem; color: #8B949E;'>Synthesizing notes into Markdown.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        with writer_container.container():
                            st.markdown("### ✍️ Report Draft")
                            st.info("Draft generated successfully. Sending to review...")
                            with st.expander("View Draft Raw Content"):
                                st.code(draft_preview, language="markdown")
                                
                    elif node == "reviewer":
                        progress_bar.progress(90)
                        status_text.write("🔍 **Reviewer Agent** is checking draft quality...")
                        
                        feedback_text = state_update.get("feedback", "")
                        final_report = state_update.get("final_report", "")
                        
                        is_approved = "APPROVED" in feedback_text.upper() or final_report != ""
                        badge_style = "background-color:#28a745;" if is_approved else "background-color:#DC3545;"
                        badge_label = "🔍 Reviewer (Approved)" if is_approved else "🔍 Reviewer (Needs Revision)"
                        
                        reviewer_container.markdown(f"""
                        <div class='agent-card'>
                            <span class='badge' style='{badge_style}'>{badge_label}</span>
                            <h4>Editor Review</h4>
                            <p style='font-size: 0.85rem; color: #8B949E;'>Feedback and quality gating.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with reviewer_container.container():
                            st.markdown("### 🔍 Editor Feedback")
                            if is_approved:
                                st.success("✅ The draft meets all criteria and has been approved!")
                            else:
                                st.warning(f"⚠️ Revisions Requested:\n{feedback_text}")
                                
                        if final_report:
                            progress_bar.progress(100)
                            status_text.write("🎉 **Workflow Complete!** Final report generated.")
                            
                            with final_report_container.container():
                                st.markdown("---")
                                st.markdown("### 📄 Final Approved Report")
                                st.markdown(final_report)
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download Research Report (Markdown)",
                                    data=final_report,
                                    file_name=f"{topic_input.lower().replace(' ', '_')}_report.md",
                                    mime="text/markdown"
                                )
                                
        except Exception as e:
            st.error(f"An error occurred during graph execution: {e}")
            progress_bar.progress(0)
            status_text.empty()
