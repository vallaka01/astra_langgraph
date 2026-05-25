import os
from dotenv import load_dotenv
import cassio

# Load .env file explicitly
load_dotenv(dotenv_path="/Users/vallaka/Documents/MyCodeProjects/astra_langgraph/.env")


class AstraDBConfig:
    """Configuration class for AstraDB connection using Cassio."""
    
    def __init__(self):
        self.api_token = os.getenv("ASTRA_DB_API_TOKEN")
        self.database_id = os.getenv("ASTRA_DB_ID")
        self.keyspace = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
        self.collection_name = os.getenv("ASTRA_DB_COLLECTION", "documents")
        
        if not self.api_token or not self.database_id:
            raise ValueError(
                "ASTRA_DB_API_TOKEN and ASTRA_DB_ID must be set in .env file"
            )
        
        # Initialize Cassio
        cassio.init(
            token=self.api_token,
            database_id=self.database_id,
            keyspace=self.keyspace
        )
    
    def get_collection(self):
        """Get the vector collection using Cassio - returns the initialized cassio instance."""
        return cassio


if __name__ == "__main__":
    # Test the connection
    try:
        config = AstraDBConfig()
        collection = config.get_collection()
        print("AstraDB connection successful")
    except Exception as e:
        print(f"Error: {e}")
