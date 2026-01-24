# Homework 5: RAG Application with Langfuse Observability

A Retrieval-Augmented Generation (RAG) system using local Ollama LLM with comprehensive Langfuse tracing and evaluation.

## Features

- **RAG System**: Document loading, chunking, embedding, retrieval, and answer generation
- **Local LLM**: Uses Ollama with qwen2.5:1.5b (smallest model)
- **Full Observability**: Langfuse integration with traces, spans, generations, events, and scores
- **Evaluation**: Custom evaluators and LLM-as-a-judge
- **Datasets**: Test datasets for systematic evaluation

## Project Structure

```
hometask_5/
├── rag_app.py                  # Main RAG application with Langfuse
├── rag_langfuse_demo.ipynb     # Jupyter notebook with full demo
├── test_rag_basic.py           # Basic RAG test (no Langfuse required)
├── test_full_system.py         # Full system test with Langfuse
├── docker-compose.langfuse.yml # Docker Compose for local Langfuse
├── setup_langfuse.sh           # Setup script
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
└── README.md                   # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd hometask_5
pip install -r requirements.txt
```

### 2. Ensure Ollama is Running

```bash
# Check if Ollama is installed
ollama --version

# Pull the smallest model
ollama pull qwen2.5:1.5b

# Start Ollama server (if not already running)
ollama serve
```

### 3. Set Up Langfuse

#### Option A: Langfuse Cloud (Recommended for quick start)

1. Go to [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a free account
3. Create a new project
4. Go to Settings > API Keys
5. Copy Public Key and Secret Key

Create `.env` file:
```bash
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b
```

#### Option B: Local Langfuse with Docker

```bash
# Start Langfuse locally
docker-compose -f docker-compose.langfuse.yml up -d

# Wait for services to start
sleep 10

# Access at http://localhost:3000
# Create account and project
# Get API keys from Settings > API Keys
```

Create `.env` file:
```bash
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-your-local-key
LANGFUSE_SECRET_KEY=sk-lf-your-local-key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b
```

### 4. Run Tests

```bash
# Basic RAG test (no Langfuse required)
python test_rag_basic.py

# Full system test with Langfuse
python test_full_system.py

# Run main application
python rag_app.py
```

### 5. Use Jupyter Notebook

```bash
jupyter notebook rag_langfuse_demo.ipynb
```

## Langfuse Entities Demonstrated

### 1. Traces
- Full execution path for document loading
- Full execution path for RAG queries
- Linked to user_id and session_id

### 2. Spans
- `text_splitting`: Document chunking operations
- `embedding_generation`: Vector embedding creation
- `document_retrieval`: Similarity search
- `llm_answer_generation`: LLM response generation

### 3. Generations
- LLM calls with input/output tracking
- Token usage estimation
- Model parameters and timing

### 4. Events
- `indexing_started` / `indexing_completed`
- `retrieval_complete`
- `query_error` (on failures)

### 5. Scores
- `document_processing`: Loading quality
- `response_quality`: Auto-scored response quality
- `llm_judge_overall`: LLM-as-judge evaluation
- `keyword_coverage`: Custom metric
- `response_time`: Performance metric
- `context_utilization`: RAG effectiveness

### 6. Datasets
- `rag_evaluation_dataset`: Test cases with expected outputs
- Linked experiments for tracking runs

### 7. Custom Evaluators
- **LLM-as-Judge**: Uses LLM to evaluate responses on relevance, accuracy, completeness, clarity
- **Programmatic**: Answer length, keyword coverage, response time, context utilization

## Screenshots for Submission

After running the notebook/scripts, take screenshots of:

1. **Traces (General View)**: Main traces list
2. **Trace Detail (Expanded)**: Single trace with all spans/generations
3. **Observations**: Spans and generations tab
4. **Dashboards**: Metrics overview
5. **LLM-as-Judge**: Traces with judge scores
6. **Annotation Queue**: Human review setup (if configured)
7. **Custom Evaluator**: Score distributions

## Metrics Tracked

| Metric | Description |
|--------|-------------|
| Execution Time | Total, retrieval, generation times |
| Token Usage | Input/output token counts |
| Documents Retrieved | Number of relevant chunks |
| Context Length | Size of context passed to LLM |
| Quality Scores | Auto and manual evaluations |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Application                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │  Documents  │───▶│  Chunking   │───▶│  Embeddings │    │
│  └─────────────┘    └─────────────┘    └──────┬──────┘    │
│                                               │            │
│                                               ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Answer    │◀───│  LLM (Ollama)│◀───│  ChromaDB   │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                     Langfuse Tracing                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Trace ─▶ Span (Retrieval) ─▶ Generation (LLM)      │  │
│  │           ─▶ Events ─▶ Scores                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Ollama not responding
```bash
# Check if running
curl http://localhost:11434/api/tags

# Restart
pkill ollama
ollama serve
```

### Langfuse connection issues
```bash
# Check environment variables
echo $LANGFUSE_HOST
echo $LANGFUSE_PUBLIC_KEY

# Verify .env file exists and is loaded
cat .env
```

### Langfuse SDK version compatibility
This project uses **Langfuse SDK v2.x** (`langfuse>=2.0.0,<3.0.0`) for compatibility with local Langfuse v2 servers. If you're using Langfuse Cloud or Langfuse v3 server, you can update to SDK v3.x, but you'll need to update the API calls (v3 uses `start_as_current_span()` instead of `trace()`).

### ChromaDB errors
```bash
# Clear local database
rm -rf ./chroma_data
```

## References

- [Langfuse Documentation](https://langfuse.com/docs)
- [Ollama Documentation](https://ollama.ai/docs)
- [LangChain Documentation](https://python.langchain.com/docs)
