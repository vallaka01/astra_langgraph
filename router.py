import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class QueryRouter:
    """Routes user queries between vector database and external tools."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query router. Your task is to determine whether a user query should be:
1. "vector_db" - For queries about specific topics, technical documentation, or information that might be in the ingested documents
2. "wikipedia" - For general knowledge, factual information about people, places, events, or current information

Respond with only one of these two options: vector_db or wikipedia"""),
            ("human", "User query: {query}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def route(self, query: str) -> str:
        """
        Route the query to the appropriate data source.
        
        Args:
            query: The user's query
            
        Returns:
            "vector_db" or "wikipedia"
        """
        result = self.chain.invoke({"query": query})
        result = result.strip().lower()
        
        # Ensure valid routing
        if result not in ["vector_db", "wikipedia"]:
            # Default to wikipedia if unclear
            return "wikipedia"
        
        return result


if __name__ == "__main__":
    router = QueryRouter()
    
    test_queries = [
        "What is LangChain?",
        "Who is the president of the United States?",
        "How do I use embeddings?",
        "What is the capital of France?",
        "Explain vector databases"
    ]
    
    for query in test_queries:
        route = router.route(query)
        print(f"{query} -> {route}")
