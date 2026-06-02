from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
import os

load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# ── 1. invoke: single call, waits for full response ──────────────────────────
print("--invoke--")
response = llm.invoke("whats is an AI agent in 2 sentance")
print(response.content)
print("TokenUsage : ", response.response_metadata.get("token_usage"))

# ── 2. stream: tokens arrive one by one, real-time feel ───────────────────────
print("\n-- stream --")
stream = llm.stream("Explain how Large Language Models work in 3–4 steps.")
for chunk in stream:
    print(chunk.content, end="", flush=True)
print()

# ── 3. batch: multiple prompts in one call (efficient) ───────────────────────
print("\n=== batch ===")
questions = [
    "What is LangChain in one sentence?",
    "What is LangGraph in one sentence?",
    "What is an MCP server in one sentence?"
]
responses = llm.batch(questions)
for q, r in zip(questions, responses):
    print(f"Q: {q}\nA: {r.content}\n")


# ── 4. multi-turn conversation using message types ────────────────────────────
print("\n=== multi-turn ===")
messages = [
    SystemMessage(content="You are a concise AI tutor. Keep answers to 1-2 sentences."),
    HumanMessage(content="What is a prompt template?"),
    AIMessage(content="A prompt template is a reusable string with variables that gets filled in at runtime before sending to the LLM."),
    HumanMessage(content="Give me an example of one.")
]
response = llm.invoke(messages)
print(response.content)