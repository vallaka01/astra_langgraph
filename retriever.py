import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Cassandra
from config import AstraDBConfig

load_dotenv()


class AstraDBRetriever:
    """Retriever for querying AstraDB vector store using LangChain Cassandra."""
    
    def __init__(self):
        self.config = AstraDBConfig()
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize Cassandra vector store
        self.vector_store = Cassandra(
            embedding=self.embeddings,
            table_name=self.config.collection_name,
            session=None
        )
    
    def query(self, query_text: str, top_k: int = 3):
        """
        Query the vector store for relevant documents.
        
        Args:
            query_text: The query text to search for
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents with content and metadata
        """
        # Perform similarity search
        documents = self.vector_store.similarity_search(query_text, k=top_k)
        
        # Format results
        results = []
        for doc in documents:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return results
    
    def get_context(self, query_text: str, top_k: int = 3):
        """
        Get formatted context from retrieved documents.
        
        Args:
            query_text: The query text to search for
            top_k: Number of top results to return
            
        Returns:
            Formatted string with retrieved context
        """
        documents = self.query(query_text, top_k)
        
        if not documents:
            return "No relevant documents found in the vector database."
        
        context = "Retrieved from vector database:\n\n"
        for i, doc in enumerate(documents, 1):
            context += f"Document {i}:\n{doc['content']}\n"
            if doc.get("metadata"):
                context += f"Metadata: {doc['metadata']}\n"
            context += "\n"
        
        return context


if __name__ == "__main__":
    retriever = AstraDBRetriever()
    
    query = "What is LangChain?"
    
    try:
        context = retriever.get_context(query)
        print(context)
    except Exception as e:
        print(f"Error: {e}")
