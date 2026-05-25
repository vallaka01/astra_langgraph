import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph import create_graph

load_dotenv()


def main():
    """Main application that runs the multi-agent chatbot."""
    app = create_graph()
    
    print("=== Multi-Agent AI Application with AstraDB and Wikipedia ===")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "route": "",
            "context": ""
        }
        
        try:
            result = app.invoke(initial_state)
            
            if result["messages"]:
                last_message = result["messages"][-1]
                print(f"Bot: {last_message.content}\n")
                
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
