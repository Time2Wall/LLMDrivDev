# Homework 3: LLM Inference Tracking with vLLM

This homework demonstrates the full cycle of using LLM as a judge (LLM-as-a-Judge): from deploying a model via vLLM to setting up custom evaluation metrics with MLflow.

## Model Used

**Qwen/Qwen2-0.5B-Instruct** - One of the smallest instruction-tuned models with chat template support (~0.5B parameters).

This model was chosen because:
- Small size (works on CPU and low-end GPUs)
- Supports chat templates (bonus criteria)
- OpenAI-compatible API format
- Good instruction following for its size

## Project Structure

```
hometask_3/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Docker setup for vLLM and MLflow
├── vllm_interaction.py       # Script for HTTP/OpenAI client interaction
└── mlflow_experiment.py      # MLflow experiment with LLM-as-Judge metric
```

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional, for containerized setup)
- GPU with CUDA support (optional, for faster inference)
- ~2GB disk space for model download

## Installation

### 1. Create Virtual Environment

```bash
cd hometask_3
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running vLLM Server

### Option A: Direct Python (Recommended for Development)

```bash
# GPU version (faster)
vllm serve Qwen/Qwen2-0.5B-Instruct --host 0.0.0.0 --port 8000

# CPU version (slower, works without GPU)
vllm serve Qwen/Qwen2-0.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float32 \
    --device cpu
```

### Option B: Docker Compose

```bash
# GPU version
docker-compose --profile gpu up -d vllm-gpu

# CPU version
docker-compose --profile cpu up -d vllm-cpu
```

### Verify Server is Running

```bash
# Check health endpoint
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/v1/models
```

Expected output:
```json
{
  "data": [
    {
      "id": "Qwen/Qwen2-0.5B-Instruct",
      "object": "model",
      ...
    }
  ]
}
```

## Running the Scripts

### 1. vLLM Interaction Demo

This script demonstrates three ways to interact with the vLLM server:

```bash
python vllm_interaction.py
```

**Features demonstrated:**
- HTTP requests using `httpx`
- HTTP requests using `requests`
- OpenAI Python client compatibility

**Sample output:**
```
============================================================
vLLM Interaction Demo
Model: Qwen/Qwen2-0.5B-Instruct
Server: http://localhost:8000
============================================================

[1] Checking server health...
Server is healthy!

[2] Available models:
  - Qwen/Qwen2-0.5B-Instruct

[3] Test Question: What is the capital of Germany?

--- Method 1: Using httpx ---
Response: The capital of Germany is Berlin.

--- Method 2: Using requests ---
Response: The capital of Germany is Berlin.

--- Method 3: Using OpenAI client ---
Response: The capital of Germany is Berlin.
```

### 2. MLflow Experiment with LLM-as-Judge

This script runs an evaluation experiment using the local model as a judge:

```bash
python mlflow_experiment.py
```

**Features demonstrated:**
- Custom GenAI metric creation with `make_genai_metric`
- LLM-as-Judge evaluation pattern
- MLflow experiment tracking
- Grading prompt design with examples

**To view MLflow results:**

```bash
# Start MLflow UI
mlflow ui --port 5000

# Or use docker-compose
docker-compose --profile mlflow up -d mlflow
```

Then open http://localhost:5000 in your browser.

## How the LLM-as-Judge Metric Works

### Grading Prompt

The custom metric evaluates responses on a 1-5 scale based on:
1. **Relevance**: Does the answer address the question?
2. **Accuracy**: Is the information correct?
3. **Clarity**: Is it well-structured?

### Scoring Scale

| Score | Meaning |
|-------|---------|
| 5 | Excellent - Directly addresses with accurate, well-structured info |
| 4 | Good - Addresses well with minor issues |
| 3 | Acceptable - Partially addresses, missing key info |
| 2 | Poor - Tangentially related, doesn't properly answer |
| 1 | Unacceptable - Irrelevant or incorrect |

### Few-Shot Examples

The metric includes grading examples for better consistency:

```python
EvaluationExample(
    input="What is the capital of France?",
    output="The capital of France is Paris...",
    score=5,
    justification="Directly addresses with correct answer and context"
)
```

## Model Behavior Analysis

### Strengths (Qwen2-0.5B-Instruct)
- Correctly answers simple factual questions
- Follows instruction format consistently
- Fast inference even on CPU
- Chat template support enables proper conversation handling

### Limitations
- May struggle with complex reasoning
- Limited knowledge for specialized domains
- Occasionally verbose or repetitive
- As a judge, may be inconsistent with subjective evaluations

### Recommendations
- For production LLM-as-Judge, use larger models (7B+)
- Temperature=0 for more consistent judging
- Provide clear grading examples for better calibration

## API Reference

### vLLM OpenAI-Compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion (recommended) |
| `/v1/completions` | POST | Text completion |

### Chat Completion Request Format

```json
{
  "model": "Qwen/Qwen2-0.5B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of Germany?"}
  ],
  "temperature": 0.7,
  "max_tokens": 256,
  "top_p": 0.95
}
```

## Troubleshooting

### "Connection refused" Error
- Ensure vLLM server is running: `curl http://localhost:8000/health`
- Check if port 8000 is available: `netstat -an | grep 8000`

### Out of Memory (GPU)
- Use CPU mode instead
- Reduce `max_model_len` parameter
- Try an even smaller model

### Slow Inference (CPU)
- CPU inference is ~10-50x slower than GPU
- Consider using a cloud GPU instance
- Reduce `max_tokens` for faster responses

### Model Download Issues
- Ensure stable internet connection
- Check HuggingFace token if required: `export HUGGING_FACE_HUB_TOKEN=your_token`
- Try manual download: `huggingface-cli download Qwen/Qwen2-0.5B-Instruct`

## Screenshots

*(Add screenshots of:)*
1. vLLM server startup logs
2. vllm_interaction.py output
3. MLflow UI - Experiments tab
4. MLflow UI - Run details with metrics

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [MLflow GenAI Metrics](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat)
- [Qwen2 Model Card](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
