# Business RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot system designed for SaaS knowledge bases. This system enables users to query company documents, FAQs, and support articles with contextual, AI-powered responses.

## 🎯 Features

- **Document Ingestion**: Load and process PDF, TXT, and Markdown files
- **Vector Search**: Semantic search using FAISS or Chroma vector stores
- **Conversational AI**: Multi-turn conversations with context awareness
- **Query Refinement**: Automatic query optimization for better retrieval
- **Source Citation**: Transparent source attribution for all answers
- **Web UI**: Streamlit-based interactive chat interface
- **REST API**: FastAPI backend for integration
- **Docker Support**: Containerized deployment ready
- **Flexible Embeddings**: Support for OpenAI or Hugging Face embeddings

## 📋 Requirements

- Python 3.10+
- OpenAI API key (for LLM and optional embeddings)
- 4GB+ RAM (for local embeddings)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/business-rag-chatbot.git
cd business-rag-chatbot

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:

```env
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=openai
VECTOR_STORE_TYPE=faiss
```

### 3. Ingest Documents

Place your documents in the `data/` directory, then run:

```bash
python ingest.py --data-dir ./data
```

This will:
- Load all PDF, TXT, and MD files from the directory
- Split them into chunks
- Generate embeddings
- Create and save the vector store

### 4. Run the Chatbot

**Option A: Streamlit UI**

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

**Option B: FastAPI Server**

```bash
python api.py
```

API will be available at `http://localhost:8000`

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `LLM_MODEL` | LLM model name | `gpt-3.5-turbo` |
| `LLM_TEMPERATURE` | LLM temperature | `0.7` |
| `LLM_MAX_TOKENS` | Max tokens per response | `1000` |
| `EMBEDDING_MODEL` | Embedding type (`openai` or `sentence-transformers`) | `openai` |
| `EMBEDDING_MODEL_NAME` | Hugging Face model name | `all-MiniLM-L6-v2` |
| `VECTOR_STORE_TYPE` | Vector store (`faiss` or `chroma`) | `faiss` |
| `VECTOR_STORE_PATH` | Path to vector store | `./vectorstore` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |
| `TOP_K_RETRIEVAL` | Number of docs to retrieve | `5` |
| `RERANK_RESULTS` | Enable reranking | `false` |

### Configuration File

You can also use a Python config file (`src/config.py`) for programmatic configuration.

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t business-rag-chatbot .

# Run with docker-compose
docker-compose up -d
```

### Docker Compose Services

- **rag-api**: FastAPI server on port 8000
- **rag-ui**: Streamlit UI on port 8501

Both services share the same data and vectorstore volumes.

### Environment Variables in Docker

Create a `.env` file or set environment variables:

```bash
docker run -e OPENAI_API_KEY=your_key business-rag-chatbot
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Test coverage includes:
- Document ingestion and chunking
- Vector store operations
- Embedding generation
- RAG chain functionality (requires API key)

## 📁 Project Structure

```
business-rag-chatbot/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── ingestion.py            # Document loading and chunking
│   ├── embeddings.py           # Embedding model management
│   ├── vectorstore.py          # Vector store operations
│   ├── rag_chain.py            # RAG chain implementation
│   └── query_refinement.py     # Query optimization
├── tests/
│   ├── test_ingestion.py
│   ├── test_vectorstore.py
│   └── test_rag_chain.py
├── data/                       # Document storage
│   ├── sample.txt
│   └── README.md
├── ingest.py                   # Ingestion CLI script
├── app.py                      # Streamlit UI
├── api.py                      # FastAPI server
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## 📧 Support

- Telegram: https://t.me/az_tekDev
- Twitter: https://x.com/az_tekDev

