import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_groq import ChatGroq

# Load env variables from .env in root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path)

# Dictionary to store chat histories by session ID
sessions_db = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Helper function to retrieve or create a chat history for a session."""
    if session_id not in sessions_db:
        sessions_db[session_id] = InMemoryChatMessageHistory()
    return sessions_db[session_id]

def run_memory_demo():
    print("\n--- Running LangChain Memory Demonstration ---")
    
    # Initialize the LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    # Prompt template containing a MessagesPlaceholder for the chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly personal tutor. Keep your answers brief and helpful."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}")
    ])
    
    # Combine prompt and LLM
    runnable = prompt | llm
    
    # Wrap the runnable with history tracking capabilities
    # RunnableWithMessageHistory handles reading messages from history and saving new messages automatically
    chain_with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # We will simulate a conversation session
    session_id = "session_user_42"
    config = {"configurable": {"session_id": session_id}}
    
    # Turn 1
    print("User: Hi, my name is Satya.")
    response1 = chain_with_history.invoke({"input": "Hi, my name is Satya."}, config=config)
    print(f"Agent: {response1.content}\n")
    
    # Turn 2
    print("User: What is 12 + 15?")
    response2 = chain_with_history.invoke({"input": "What is 12 + 15?"}, config=config)
    print(f"Agent: {response2.content}\n")
    
    # Turn 3 (Checks if the agent remembers name from Turn 1)
    print("User: Do you remember my name?")
    response3 = chain_with_history.invoke({"input": "Do you remember my name?"}, config=config)
    print(f"Agent: {response3.content}\n")
    
    # Print the full session history to show how it is stored
    history = get_session_history(session_id)
    print("--- Under the Hood (Stored Message History) ---")
    for message in history.messages:
        role = "User" if message.type == "human" else "Agent"
        print(f"[{role}]: {message.content}")

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in environment.")
    else:
        run_memory_demo()
