# Multi-Agent AI Application with LangGraph and AstraDB

A multi-agent AI application that routes user queries between a vector database (AstraDB) and external tools (Wikipedia) using LangGraph.

## Architecture

```
User Input → Router → [Vector DB | Wikipedia] → Response Generator → Output
```

## Components

- **Router**: LLM-based classifier that routes queries to the appropriate data source
- **Vector DB**: AstraDB vector store for ingested documents
- **Wikipedia**: External tool for general knowledge queries
- **Response Generator**: Synthesizes answers from retrieved context

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure environment variables** in `.env`:
```
ASTRA_DB_API_TOKEN=your_astra_db_api_token
ASTRA_DB_API_ENDPOINT=your_astra_db_api_endpoint
ASTRA_DB_KEYSPACE=default_keyspace
ASTRA_DB_COLLECTION=documents
OPENAI_API_KEY=your_openai_api_key
```

3. **Test AstraDB connection**:
```bash
python config.py
```

4. **Ingest data (optional)**:
```bash
python data_ingestion.py
```

## Usage

### Run the main application:
```bash
python app.py
```

### Test individual components:
```bash
# Test router
python router.py

# Test retriever
python retriever.py

# Test graph
python graph.py
```

## File Structure

- `config.py` - AstraDB configuration and connection
- `data_ingestion.py` - Data ingestion from URLs to AstraDB
- `retriever.py` - Vector database query functionality
- `tools.py` - Wikipedia search tool
- `router.py` - LLM-based query routing
- `graph.py` - LangGraph workflow definition
- `app.py` - Main application interface

## Example Queries

- **Vector DB**: "What is LangChain?", "How do I use embeddings?"
- **Wikipedia**: "Who is the president of the United States?", "What is the capital of France?"
