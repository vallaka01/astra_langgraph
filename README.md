# Multi-Agent AI Application with LangGraph and AstraDB

A multi-agent AI application that routes user queries between a vector database (AstraDB) and external tools (Wikipedia) using LangGraph.

## Flow Chart

```
┌─────────────┐
│ User Input  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Router    │ (LLM Classifier)
└──────┬──────┘
       │
       ├───► Vector DB ──► Response Generator ──► Output
       │       (AstraDB)
       │
       └───► Wikipedia ──► Response Generator ──► Output
```

**Workflow:**
1. User submits a query
2. Router (LLM) classifies the query as either technical (vector DB) or general knowledge (Wikipedia)
3. If technical: Query AstraDB vector store for relevant documents
4. If general: Search Wikipedia for information
5. Response Generator synthesizes the answer using retrieved context
6. Return answer to user

## Environment Setup

### Prerequisites
- Python 3.11 or higher
- AstraDB account (free tier available at https://astra.datastax.com)
- OpenAI API key

### Step 1: Clone the Repository
```bash
git clone https://github.com/vallaka01/astra_langgraph.git
cd astra_langgraph
```

### Step 2: Create and Activate Virtual Environment
```bash
# Using conda
conda create -n astra_langgraph python=3.11
conda activate astra_langgraph

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `langgraph` - Multi-agent workflow orchestration
- `langchain` - LLM framework
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community tools (Cassandra vector store, WebBaseLoader)
- `cassio` - AstraDB Python client
- `wikipedia` - Wikipedia API
- `python-dotenv` - Environment variable management

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# AstraDB Configuration
ASTRA_DB_API_TOKEN=AstraCS:your_token_here
ASTRA_DB_API_ENDPOINT=https://your-database-id.apps.astra.datastax.com
ASTRA_DB_ID=your-database-id
ASTRA_DB_KEYSPACE=default_keyspace
ASTRA_DB_COLLECTION=documents

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
```

**Getting AstraDB Credentials:**
1. Sign up at https://astra.datastax.com
2. Create a new database
3. Copy the API token and database ID from the dashboard
4. The database ID is the UUID in your endpoint URL

**Getting OpenAI API Key:**
1. Sign up at https://platform.openai.com
2. Go to API Keys section
3. Create a new API key

### Step 5: Test AstraDB Connection
```bash
python config.py
```

Expected output: `AstraDB connection successful`

### Step 6: Ingest Data (Optional but Recommended)

Ingest sample documents into AstraDB for vector search:

```bash
python data_ingestion.py
```

This will ingest content from:
- AI Agent blog posts
- Prompt Engineering articles
- LangChain documentation

## How to Run the Code

### Option 1: Run the Main Application (Interactive Chatbot)

```bash
python app.py
```

This starts an interactive CLI where you can:
- Type queries and get responses
- The system automatically routes to vector DB or Wikipedia
- Type `quit` or `exit` to end the session

**Example Session:**
```
=== Multi-Agent AI Application with AstraDB and Wikipedia ===
Type 'quit' or 'exit' to end the conversation.

You: What is LangChain?
Bot: LangChain is a framework designed to help you build and manage AI agents...

You: Who is Albert Einstein?
Bot: Albert Einstein was a theoretical physicist who is best known for developing the theory of relativity...

You: quit
Goodbye!
```

### Option 2: Run the Test Suite

```bash
python test_app.py
```

This runs 5 test scenarios:
1. Router Classification - Tests query routing
2. Wikipedia Tool - Tests Wikipedia search
3. Vector DB Retriever - Tests AstraDB queries
4. LangGraph Workflow - Tests complete graph execution
5. End-to-End Application - Tests full application flow

### Option 3: Test Individual Components

```bash
# Test router
python router.py

# Test retriever
python retriever.py

# Test Wikipedia tool
python tools.py
```

## Components

- **Router** (`router.py`): LLM-based classifier that routes queries to the appropriate data source
- **Vector DB** (`retriever.py`): AstraDB vector store for ingested documents with semantic search
- **Wikipedia** (`tools.py`): External tool for general knowledge queries
- **Response Generator** (`graph.py`): Synthesizes answers from retrieved context
- **Data Ingestion** (`data_ingestion.py`): Loads and processes URLs into AstraDB

## File Structure

```
astra_langgraph/
├── .env                    # Environment variables (not in git)
├── .gitignore             # Git ignore file
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── config.py             # AstraDB configuration and connection
├── data_ingestion.py     # Data ingestion from URLs to AstraDB
├── retriever.py          # Vector database query functionality
├── tools.py              # Wikipedia search tool
├── router.py             # LLM-based query routing
├── graph.py              # LangGraph workflow definition
├── app.py                # Main application interface
└── test_app.py           # Test suite with documented scenarios
```

## Example Queries

**Vector DB Queries** (routes to AstraDB):
- "What is LangChain?"
- "How do I use embeddings?"
- "Explain prompt engineering"
- "What is an AI agent?"

**Wikipedia Queries** (routes to Wikipedia):
- "Who is the president of the United States?"
- "What is the capital of France?"
- "What is the population of Tokyo?"
- "Who is Albert Einstein?"

## Troubleshooting

**Issue: AstraDB connection fails**
- Verify your API token and database ID in `.env`
- Ensure your AstraDB database is active
- Check internet connection

**Issue: Data ingestion fails**
- Ensure URLs are accessible
- Check if you have enough quota in AstraDB
- Verify OpenAI API key is valid

**Issue: Router always routes to Wikipedia**
- Ensure data has been ingested into AstraDB
- Check if vector store has documents
- The router may default to Wikipedia if vector DB is empty

**Issue: Import errors**
- Ensure you're using the correct Python environment
- Reinstall dependencies: `pip install -r requirements.txt`
- Use the langchain_v1 conda environment if available
