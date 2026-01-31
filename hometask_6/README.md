# RAG System with Vector Database Comparison

This project implements a complete RAG (Retrieval Augmented Generation) system with comparative analysis of vector databases and ANN algorithms.

## Project Overview

### Features Implemented

1. **Vector Database Setup and Indexing**
   - ChromaDB (embedded, persistent storage)
   - Milvus Lite (embedded version for ANN comparison)
   - Custom dataset generation (1200+ documents)

2. **ANN Algorithm Comparison**
   - FLAT (exact search, baseline)
   - IVF_FLAT (inverted file index)
   - HNSW (Hierarchical Navigable Small World graphs)

3. **Search Capabilities**
   - Semantic vector search
   - Multiple similarity metrics (Cosine, L2, Inner Product)
   - Metadata filtering (category, difficulty, year)
   - Hybrid search (vector + keyword boost)
   - Batch processing

4. **End-to-End RAG Pipeline**
   - Document retrieval
   - Context preparation
   - LLM-based answer generation (Ollama - local models)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Prerequisites

### Ollama Setup (Required for LLM)

This project uses Ollama for local LLM inference. Install and set up Ollama:

1. **Install Ollama**: https://ollama.ai/download

2. **Pull a model** (choose one):
```bash
# Recommended: Llama 3.2 (default)
ollama pull llama3.2

# Alternatives:
ollama pull mistral
ollama pull gemma2
ollama pull phi3
```

3. **Start Ollama server**:
```bash
ollama serve
```

4. **Verify installation**:
```bash
ollama list
```

## Usage

### Running the Jupyter Notebook

```bash
jupyter notebook rag_system.ipynb
```

### Running the Python Script

```bash
python rag_system.py
```

### Changing the LLM Model

To use a different Ollama model, modify the `model_name` parameter:

```python
# In Python script or notebook:
rag_system = RAGSystem(
    chroma_manager,
    embedding_gen,
    llm_provider="ollama",
    model_name="mistral"  # Change to your preferred model
)
```

## Dataset

The project generates a synthetic dataset about programming and technology topics:

- **Categories**: Python, JavaScript, Machine Learning, Web Development, Databases, DevOps, Security, Cloud Computing, Data Science, Algorithms
- **Metadata**: category, difficulty (beginner/intermediate/advanced), year (2020-2024)
- **Size**: 1200+ documents

## ANN Algorithms Explained

### FLAT (Brute Force)
- **Pros**: 100% recall, exact results
- **Cons**: Slowest, O(n) complexity
- **Use case**: Small datasets, ground truth baseline

### IVF_FLAT (Inverted File Index)
- **Pros**: Good balance of speed and accuracy
- **Cons**: Requires tuning nlist and nprobe parameters
- **Parameters**:
  - `nlist`: Number of clusters (higher = more partitions)
  - `nprobe`: Number of clusters to search (higher = more accurate)
- **Use case**: Large datasets where some recall loss is acceptable

### HNSW (Hierarchical Navigable Small World)
- **Pros**: Excellent speed/accuracy trade-off, no training required
- **Cons**: Higher memory usage
- **Parameters**:
  - `M`: Maximum number of connections per layer
  - `efConstruction`: Search quality during construction
  - `ef`: Search quality during query (higher = more accurate)
- **Use case**: Production systems requiring high recall

## Similarity Metrics

| Metric | Description | Range | Best For |
|--------|-------------|-------|----------|
| Cosine | Angle between vectors | -1 to 1 | Semantic similarity |
| L2 (Euclidean) | Absolute distance | 0 to inf | Physical distances |
| Inner Product | Dot product | -inf to inf | Normalized vectors |

## Performance Results

Typical performance on the test dataset (1200 documents):

| Index Type | Indexing Time | Avg Search Time | Recall@10 |
|------------|---------------|-----------------|-----------|
| FLAT | ~2s | ~5ms | 100% |
| IVF_FLAT_128 | ~2s | ~2ms | ~95% |
| HNSW_M16 | ~3s | ~1ms | ~98% |

## Project Structure

```
hometask_6/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
├── rag_system.ipynb         # Main Jupyter notebook
├── rag_system.py            # Python script version
├── chroma_db/               # ChromaDB storage (created at runtime)
└── *.db                     # Milvus Lite databases (created at runtime)
```

## Key Learnings

1. **HNSW provides the best speed/accuracy trade-off** for most production use cases
2. **Batch processing significantly improves throughput** - use it for bulk operations
3. **Metadata filtering adds negligible overhead** - use it liberally for precision
4. **3-5 context documents are typically sufficient** for RAG answer generation
5. **Cosine similarity works best** for semantic search with normalized embeddings
6. **Local LLMs (Ollama) work great** for RAG without API costs or internet dependency

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
ollama list

# Start Ollama server
ollama serve

# Pull model if not available
ollama pull llama3.2
```

### Memory Issues with Large Models
Use smaller models like `phi3` or `gemma2:2b` if you encounter memory issues.

## References

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Milvus Documentation](https://milvus.io/docs)
- [Ollama Documentation](https://ollama.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
