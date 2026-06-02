import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# Load env variables from .env in root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path)

# 1. Basic Text Chain (LCEL)
def run_basic_chain():
    print("\n--- Running Basic LCEL Chain ---")
    
    # Initialize the LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful science communicator. Explain topics in simple terms."),
        ("user", "Explain what {topic} is in two sentences.")
    ])
    
    # Combine prompt, model, and output parser into a chain
    chain = prompt | llm | StrOutputParser()
    
    # Invoke the chain
    result = chain.invoke({"topic": "Quantum Computing"})
    print(f"Topic: Quantum Computing\nResponse:\n{result}\n")

# 2. Structured Output Chain
class ResearchTopicInfo(BaseModel):
    summary: str = Field(description="A concise summary of the topic (1-2 sentences).")
    key_terms: list[str] = Field(description="List of 3 key terms or concepts related to the topic.")
    difficulty_level: str = Field(description="Target audience difficulty (Beginner, Intermediate, Advanced).")

def run_structured_chain():
    print("--- Running Structured Output Chain ---")
    
    # Initialize LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    
    # Bind structure to model using Pydantic
    structured_llm = llm.with_structured_output(ResearchTopicInfo)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the given topic and return structured information."),
        ("user", "Analyze this topic: {topic}")
    ])
    
    # Combine prompt and structured LLM
    chain = prompt | structured_llm
    
    # Invoke the chain
    result = chain.invoke({"topic": "Superconductivity"})
    print(f"Topic: Superconductivity")
    print(f"Summary: {result.summary}")
    print(f"Key Terms: {result.key_terms}")
    print(f"Difficulty Level: {result.difficulty_level}\n")

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in environment.")
    else:
        run_basic_chain()
        run_structured_chain()
