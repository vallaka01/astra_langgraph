"""
Test Script for Multi-Agent AI Application with LangGraph and AstraDB

This script tests various scenarios to verify the application works correctly.
Each test scenario is documented with its purpose and expected behavior.
"""

from langchain_core.messages import HumanMessage
from graph import create_graph
from router import QueryRouter
from retriever import AstraDBRetriever
from tools import search_wikipedia


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def test_router():
    """
    Test Scenario 1: Router Classification
    
    Purpose: Verify the LLM router correctly classifies queries.
    
    Test Cases:
    - Technical queries should route to vector_db
    - General knowledge queries should route to wikipedia
    
    Expected: Router returns appropriate route for each query type.
    """
    print_section("Test 1: Router Classification")
    
    router = QueryRouter()
    
    test_queries = [
        ("What is LangChain?", "vector_db"),
        ("Who is the president of the United States?", "wikipedia"),
        ("How do I use embeddings?", "vector_db"),
        ("What is the capital of France?", "wikipedia"),
        ("Explain vector databases", "vector_db"),
    ]
    
    for query, expected_route in test_queries:
        route = router.route(query)
        status = "✓" if route == expected_route else "✗"
        print(f"{status} Query: {query}")
        print(f"   Expected: {expected_route}, Got: {route}\n")


def test_wikipedia_tool():
    """
    Test Scenario 2: Wikipedia Tool
    
    Purpose: Verify the Wikipedia search tool works correctly.
    
    Test Cases:
    - Search for a well-known entity
    - Search for a general knowledge topic
    
    Expected: Tool returns relevant Wikipedia information.
    """
    print_section("Test 2: Wikipedia Tool")
    
    test_queries = [
        "Python programming language",
        "Machine learning"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        result = search_wikipedia.invoke({"query": query})
        print(f"Result: {result[:200]}...\n")


def test_vector_db_retriever():
    """
    Test Scenario 3: Vector DB Retriever
    
    Purpose: Verify the AstraDB retriever can query the vector store.
    
    Test Cases:
    - Query for technical topics that should be in ingested documents
    
    Expected: Retriever returns relevant documents from AstraDB.
    
    Note: This test requires data to be ingested first via data_ingestion.py
    """
    print_section("Test 3: Vector DB Retriever")
    
    retriever = AstraDBRetriever()
    
    test_queries = [
        "What is an AI agent?",
        "Explain prompt engineering"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        try:
            context = retriever.get_context(query, top_k=2)
            print(f"Context preview: {context[:300]}...\n")
        except Exception as e:
            print(f"Error: {e}\n")
            print("Note: Ensure data has been ingested via data_ingestion.py\n")


def test_graph_workflow():
    """
    Test Scenario 4: LangGraph Workflow
    
    Purpose: Verify the complete LangGraph workflow executes correctly.
    
    Test Cases:
    - Query that should route to vector_db
    - Query that should route to wikipedia
    
    Expected: Graph executes all nodes and returns a response.
    """
    print_section("Test 4: LangGraph Workflow")
    
    app = create_graph()
    
    test_cases = [
        "What is LangChain?",
        "Who is Albert Einstein?"
    ]
    
    for query in test_cases:
        print(f"Query: {query}")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "route": "",
            "context": ""
        }
        
        try:
            result = app.invoke(initial_state)
            
            if result["messages"]:
                last_message = result["messages"][-1]
                print(f"Response: {last_message.content[:200]}...\n")
            
            if result.get("route"):
                print(f"Route taken: {result['route']}\n")
                
        except Exception as e:
            print(f"Error: {e}\n")


def test_end_to_end():
    """
    Test Scenario 5: End-to-End Application
    
    Purpose: Verify the complete application flow from user input to response.
    
    Test Cases:
    - Technical query (should use vector DB)
    - General knowledge query (should use Wikipedia)
    - Ambiguous query (should still work)
    
    Expected: Application handles all query types and returns coherent responses.
    """
    print_section("Test 5: End-to-End Application")
    
    app = create_graph()
    
    test_queries = [
        "What is a prompt in AI?",
        "What is the population of Tokyo?",
        "How do LLMs work?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "route": "",
            "context": ""
        }
        
        try:
            result = app.invoke(initial_state)
            
            if result["messages"]:
                last_message = result["messages"][-1]
                print(f"Response: {last_message.content[:300]}...\n")
                
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Run all test scenarios."""
    print("\n" + "=" * 60)
    print("  Multi-Agent AI Application - Test Suite")
    print("=" * 60)
    
    # Run all tests
    test_router()
    test_wikipedia_tool()
    test_vector_db_retriever()
    test_graph_workflow()
    test_end_to_end()
    
    print_section("Test Suite Complete")
    print("Review the results above to verify all scenarios passed.\n")


if __name__ == "__main__":
    main()
