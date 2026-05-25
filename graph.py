import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from router import QueryRouter
from retriever import AstraDBRetriever
from tools import tools

load_dotenv()


class State(TypedDict):
    """State management class for the multi-agent application."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str  # "vector_db" or "wikipedia"
    context: str  # Retrieved context from vector DB or Wikipedia


def router_node(state: State) -> State:
    """Routes the query to the appropriate data source."""
    query_router = QueryRouter()
    
    # Get the last user message
    last_message = state["messages"][-1]
    query = last_message.content if isinstance(last_message, HumanMessage) else str(last_message)
    
    # Determine route
    route = query_router.route(query)
    
    return {"route": route}


def vector_db_node(state: State) -> State:
    """Retrieves relevant documents from AstraDB vector store."""
    retriever = AstraDBRetriever()
    
    # Get the last user message
    last_message = state["messages"][-1]
    query = last_message.content if isinstance(last_message, HumanMessage) else str(last_message)
    
    # Query the vector store
    context = retriever.get_context(query, top_k=3)
    
    return {"context": context}


def wikipedia_node(state: State) -> State:
    """Searches Wikipedia for information."""
    from tools import search_wikipedia
    
    # Get the last user message
    last_message = state["messages"][-1]
    query = last_message.content if isinstance(last_message, HumanMessage) else str(last_message)
    
    # Search Wikipedia
    context = search_wikipedia.invoke({"query": query})
    
    return {"context": context}


def response_generator(state: State) -> State:
    """Generates the final response based on retrieved context."""
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Get the last user message and context
    last_message = state["messages"][-1]
    query = last_message.content if isinstance(last_message, HumanMessage) else str(last_message)
    context = state.get("context", "")
    route = state.get("route", "unknown")
    
    # Generate response
    prompt = f"""You are a helpful assistant. Answer the user's question based on the provided context.

User Question: {query}

Source: {route}

Context:
{context}

Provide a clear and helpful answer based on the context above."""
    
    response = llm.invoke(prompt)
    
    return {"messages": [AIMessage(content=response.content)]}


def should_route(state: State) -> str:
    """Determines which node to route to based on the route decision."""
    route = state.get("route", "wikipedia")
    return route


def create_graph():
    """Creates and compiles the LangGraph workflow."""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("vector_db", vector_db_node)
    workflow.add_node("wikipedia", wikipedia_node)
    workflow.add_node("response_generator", response_generator)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edge from router
    workflow.add_conditional_edges(
        "router",
        should_route,
        {
            "vector_db": "vector_db",
            "wikipedia": "wikipedia"
        }
    )
    
    # Add edges from data sources to response generator
    workflow.add_edge("vector_db", "response_generator")
    workflow.add_edge("wikipedia", "response_generator")
    
    # Add edge from response generator to end
    workflow.add_edge("response_generator", END)
    
    return workflow.compile()


if __name__ == "__main__":
    # Test the graph
    from pprint import pprint
    
    app = create_graph()
    
    # Test with a query
    test_query = "What is LangChain?"
    initial_state = {
        "messages": [HumanMessage(content=test_query)],
        "route": "",
        "context": ""
    }
    
    print(f"Query: {test_query}\n")
    print("=== Streaming Events ===\n")
    
    for event in app.stream(initial_state):
        pprint(event)
        print("-" * 50)
