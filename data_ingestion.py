import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Cassandra
from config import AstraDBConfig

load_dotenv()


def main():
    """Main function to ingest data from URLs into AstraDB."""
    config = AstraDBConfig()
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    vector_store = Cassandra(
        embedding=embeddings,
        table_name=config.collection_name,
        session=None
    )
    
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        "https://python.langchain.com/docs/get_started/introduction",
    ]
    
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            vector_store.add_documents(chunks)
            print(f"Ingested {len(chunks)} chunks from {url}")
        except Exception as e:
            print(f"Error ingesting {url}: {e}")


if __name__ == "__main__":
    main()
