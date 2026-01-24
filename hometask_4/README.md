# Homework 4: Text Generation Quality Evaluation

## Track A: Question Answering (QA)

### Track Justification
Track A (Question Answering) was chosen for this assignment because:
1. QA is a fundamental NLP task that tests model comprehension and factual accuracy
2. SberQuAD provides high-quality Russian language QA pairs with context
3. Clear metrics (F1, Exact Match) allow objective comparison between models

### Dataset
- **Source:** SberQuAD (sberbank-ai/sberquad) from Hugging Face
- **Size:** 50 samples for evaluation
- **Format:** Context paragraph + Question + Reference answer(s)

### Models Compared (via Ollama - FREE, local)
1. **qwen2.5:1.5b** - Alibaba's efficient small model (1.5B parameters)
2. **gemma2:2b** - Google's compact model (2B parameters)

---

## Project Structure

```
hometask_4/
├── qa_evaluation.ipynb      # Main Jupyter notebook with experiments
├── qa_evaluation_results.csv  # Detailed results (generated)
├── qa_evaluation_summary.json # Summary statistics (generated)
├── evaluation_plots.png      # Visualization plots (generated)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Reproduction Instructions

### 1. Install Ollama (FREE, local LLM server)

Download and install Ollama from: https://ollama.ai

Then pull the required models:
```bash
ollama pull qwen2.5:1.5b
ollama pull gemma2:2b
```

Make sure Ollama is running (it starts automatically after installation).

### 2. Environment Setup

```bash
cd hometask_4

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook qa_evaluation.ipynb
```

Or use JupyterLab:
```bash
jupyter lab
```

Run all cells sequentially.

---

## Implemented Metrics

### Required Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Exact Match** | Binary match after normalization | 0-1 |
| **F1 Score** | Token-level precision/recall | 0-1 |
| **BLEU Score** | N-gram overlap (via sacrebleu) | 0-1 |
| **Semantic Similarity** | Cosine similarity of embeddings | 0-1 |
| **Generation Time** | Response latency in seconds | >0 |
| **Response Length** | Character count | >0 |

### Embedding Model
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for Russian language support

---

## Experiments Conducted

### Experiment 1: Model Comparison
- Compare qwen2.5:1.5b vs gemma2:2b on zero-shot QA
- 50 samples from SberQuAD

### Experiment 2: Zero-shot vs Few-shot
- Compare prompting strategies
- 3 few-shot examples

### Experiment 3: Temperature Analysis
- Test temperatures: 0.0, 0.3, 0.7, 1.0
- Measure impact on answer consistency

### Experiment 4: Prompt Format A/B Testing
- Simple prompt vs detailed instructional prompt
- Measure impact on answer quality and length

---

## Analysis Components

### Quantitative Analysis
- Comparative tables with mean/std for all metrics
- Statistical comparison between models and settings

### Qualitative Analysis
- Top 5 best predictions (highest F1)
- Top 5 worst predictions (lowest F1)
- Error pattern categorization

### Visualizations
- Model comparison bar chart
- Generation time boxplots
- Temperature impact line chart
- Response length distributions

---

## Key Findings

### Metrics Informativeness
- **Most informative:** F1-score (captures partial matches effectively)
- **Complementary:** Semantic similarity (captures meaning with different words)
- **Strict:** Exact match (useful for short factual answers)
- **Less useful for QA:** BLEU (designed for longer machine translation)

### Model Recommendations
1. Small local models (1.5B-2B params) can handle basic QA tasks
2. Temperature 0.0-0.3 gives most consistent results
3. Detailed prompts help constrain answer format

---

## Output Files

After running the notebook:
- `qa_evaluation_results.csv` - Complete results with all metrics
- `qa_evaluation_summary.json` - Aggregated statistics
- `evaluation_plots.png` - All visualization plots

---

## Dependencies

See `requirements.txt` for full list. Key packages:
- `requests` - Ollama API client
- `datasets` - HuggingFace datasets
- `sentence-transformers` - Semantic similarity
- `sacrebleu` - BLEU score calculation
- `pandas`, `numpy` - Data processing
- `matplotlib`, `seaborn` - Visualization

---
