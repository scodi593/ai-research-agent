import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from phase3_tools.tools import search_wikipedia, scrape_webpage

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path)

# System prompt for ReAct Agent
REACT_PROMPT = """You are a helpful research assistant. Solve the user's task using the tools provided.
You MUST follow a strict loop: Thought, Action, Observation, Thought, Action, Observation... until you have the final answer.

Available Tools:
- search_wikipedia(query): Searches Wikipedia for information.
- scrape_webpage(url): Extracts content from a specific web link.

Format to use:
Thought: Reflect on the user request and think about what tool to call next (or if you have the final answer).
Action: tool_name(arguments) - only use the exact name of one of the available tools. E.g., Action: search_wikipedia(query="machine learning") or Action: scrape_webpage(url="https://example.com")
Observation: [The result of running the tool. This is provided by the system, do not write this yourself.]
... (this Thought/Action/Observation can repeat N times)
Thought: I now have the final answer.
Final Answer: [Write your complete, detailed final answer here.]

Begin the task!

User Task: {task}
"""

class ReActAgent:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(model=model_name, temperature=0)
        self.tools = {
            "search_wikipedia": search_wikipedia,
            "scrape_webpage": scrape_webpage
        }

    def run(self, task: str, max_turns: int = 5) -> str:
        print(f"\n[Running ReAct Agent] Task: '{task}'")
        scratchpad = ""
        prompt_formatted = REACT_PROMPT.format(task=task)
        
        for turn in range(1, max_turns + 1):
            print(f"\n--- Turn {turn} ---")
            current_prompt = prompt_formatted + scratchpad
            response = self.llm.invoke(current_prompt).content
            print(response)
            
            # Append agent response to scratchpad
            scratchpad += "\n" + response
            
            # Check if agent has provided the final answer
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                return final_answer
            
            # Extract Action
            action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", response)
            if not action_match:
                # If the agent didn't output action, but didn't output Final Answer either, prompt it to finalize
                scratchpad += "\nThought: I should formulate the Final Answer now."
                continue
                
            tool_name = action_match.group(1)
            tool_args_str = action_match.group(2)
            
            # Parse arguments
            args = {}
            arg_match = re.search(r'(\w+)\s*=\s*["\'](.*?)["\']', tool_args_str)
            if arg_match:
                args[arg_match.group(1)] = arg_match.group(2)
            else:
                # Fallback if arguments are raw
                clean_arg = tool_args_str.strip('"\'')
                if tool_name == "search_wikipedia":
                    args["query"] = clean_arg
                elif tool_name == "scrape_webpage":
                    args["url"] = clean_arg
            
            # Execute tool
            if tool_name in self.tools:
                print(f"[Tool Execution] {tool_name} with args {args}")
                tool = self.tools[tool_name]
                try:
                    observation = tool.invoke(args)
                except Exception as e:
                    observation = f"Error running tool: {str(e)}"
            else:
                observation = f"Error: Tool '{tool_name}' is not available."
                
            print(f"[Tool Observation]\n{observation[:300]}...")
            
            # Append observation to scratchpad
            scratchpad += f"\nObservation: {observation}"
            
        return "Failed to complete task within max turns."

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in environment.")
    else:
        agent = ReActAgent()
        task = "Find who won the Nobel Prize in Physics in 2024 and why."
        result = agent.run(task)
        print("\n================ FINAL ANSWER ================")
        print(result)
