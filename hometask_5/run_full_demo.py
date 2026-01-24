"""
Full RAG Demo with All Langfuse Features
=========================================
Runs RAG queries and applies all evaluators:
- LLM-as-Judge evaluation
- Custom programmatic evaluators (keyword_coverage, response_time, context_utilization)
- Dataset creation and experiment
"""

import os
import time
import uuid
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langfuse import Langfuse
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Configuration
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
USER_ID = f"user_{uuid.uuid4().hex[:8]}"

# Initialize clients
langfuse = Langfuse(
    host=LANGFUSE_HOST,
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
)

llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.7)
embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

# Sample documents
SAMPLE_DOCUMENTS = [
    {
        "content": """
        Machine Learning Fundamentals

        Machine learning is a subset of artificial intelligence that enables systems to learn
        and improve from experience without being explicitly programmed. There are three main
        types of machine learning:

        1. Supervised Learning: The algorithm learns from labeled training data.
        2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
        3. Reinforcement Learning: The algorithm learns through trial and error.
        """,
        "metadata": {"source": "ml_fundamentals.txt", "topic": "machine_learning"}
    },
    {
        "content": """
        Deep Learning and Neural Networks

        Deep learning uses artificial neural networks with multiple layers.
        Key concepts include:
        - Neurons and Layers: Input, hidden, and output layers
        - Activation Functions: ReLU, Sigmoid, Tanh
        - Backpropagation: Training by calculating gradients and adjusting weights
        - Architectures: CNNs for images, RNNs for sequences, Transformers for NLP
        """,
        "metadata": {"source": "deep_learning.txt", "topic": "deep_learning"}
    },
    {
        "content": """
        Large Language Models (LLMs)

        LLMs are AI systems trained on vast text data using Transformer architecture.
        - Training: Unsupervised learning followed by RLHF fine-tuning
        - Capabilities: Text generation, summarization, translation, QA, coding
        - Examples: GPT-4, Claude, LLaMA, Qwen, Mistral
        - RAG: Retrieval-Augmented Generation combines LLMs with knowledge retrieval
        """,
        "metadata": {"source": "llm_overview.txt", "topic": "llm"}
    }
]

# Test items with expected outputs
TEST_ITEMS = [
    {
        "question": "What are the three types of machine learning?",
        "expected": "supervised learning, unsupervised learning, and reinforcement learning",
        "keywords": ["supervised", "unsupervised", "reinforcement"]
    },
    {
        "question": "What is backpropagation?",
        "expected": "algorithm for training neural networks by calculating gradients",
        "keywords": ["backpropagation", "gradient", "neural", "weights", "training"]
    },
    {
        "question": "What is RAG and why is it useful?",
        "expected": "Retrieval-Augmented Generation combines LLMs with knowledge retrieval",
        "keywords": ["retrieval", "augmented", "generation", "llm", "knowledge"]
    }
]

# RAG Prompt
RAG_PROMPT = PromptTemplate(
    template="""Use the context to answer the question.
If you cannot find the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# LLM Judge Prompt
JUDGE_PROMPT = PromptTemplate(
    template="""You are an expert evaluator. Assess the AI response quality.

Question: {question}
Expected (key points): {expected}
Actual Response: {response}

Score 0-10 for each:
1. Relevance: Does it answer the question?
2. Accuracy: Are facts correct?
3. Completeness: Covers key points?
4. Clarity: Well-written?

Return JSON only:
{{"relevance": <score>, "accuracy": <score>, "completeness": <score>, "clarity": <score>, "overall": <average>, "explanation": "<brief>"}}

JSON:""",
    input_variables=["question", "expected", "response"]
)


def load_documents():
    """Load documents with tracing."""
    trace = langfuse.trace(
        name="document_indexing",
        user_id=USER_ID,
        metadata={"num_documents": len(SAMPLE_DOCUMENTS)}
    )

    trace.event(name="indexing_started")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in SAMPLE_DOCUMENTS]
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"demo_{uuid.uuid4().hex[:8]}"
    )

    trace.event(name="indexing_completed", metadata={"chunks": len(chunks)})
    trace.score(name="indexing_success", value=1.0)
    langfuse.flush()

    print(f"Indexed {len(chunks)} chunks")
    return vectorstore


def rag_query(question: str, vectorstore: Chroma, session_id: str) -> Dict[str, Any]:
    """Execute RAG query with full tracing."""
    trace = langfuse.trace(
        name="rag_query",
        user_id=USER_ID,
        session_id=session_id,
        input=question,
        metadata={"model": OLLAMA_MODEL}
    )

    total_start = time.time()

    # Retrieval
    retrieval_span = trace.span(name="document_retrieval", input={"question": question})
    retrieval_start = time.time()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    retrieval_time = time.time() - retrieval_start
    retrieval_span.end(output={"docs": len(docs), "context_len": len(context)})

    trace.event(name="retrieval_complete", metadata={"docs": len(docs)})

    # Generation
    generation = trace.generation(
        name="llm_generation",
        model=OLLAMA_MODEL,
        input={"question": question, "context_len": len(context)}
    )

    gen_start = time.time()
    prompt = RAG_PROMPT.format(context=context, question=question)
    answer = llm.invoke(prompt)
    gen_time = time.time() - gen_start

    input_tokens = int(len(prompt.split()) * 1.3)
    output_tokens = int(len(answer.split()) * 1.3)

    generation.end(
        output=answer,
        usage={"input": input_tokens, "output": output_tokens}
    )

    total_time = time.time() - total_start

    trace.update(output=answer)
    trace.score(
        name="response_quality",
        value=1.0 if len(answer) > 50 else 0.5,
        comment="Auto-scored by length"
    )

    langfuse.flush()

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "trace_id": trace.id,
        "metrics": {
            "total_time": total_time,
            "retrieval_time": retrieval_time,
            "generation_time": gen_time
        }
    }


def llm_judge_evaluate(question: str, expected: str, response: str, trace_id: str):
    """LLM-as-Judge evaluation - creates llm_judge_* scores."""
    eval_trace = langfuse.trace(
        name="llm_judge_evaluation",
        user_id=USER_ID,
        input={
            "question": question,
            "expected_answer": expected,
            "response_to_evaluate": response[:500]
        },
        metadata={"evaluated_trace_id": trace_id}
    )

    judge_gen = eval_trace.generation(
        name="judge_evaluation",
        model=OLLAMA_MODEL,
        input={"question": question, "expected": expected, "response": response[:300]}
    )

    prompt = JUDGE_PROMPT.format(question=question, expected=expected, response=response[:500])
    judge_response = llm.invoke(prompt)

    judge_gen.end(output=judge_response)

    # Parse scores
    try:
        json_match = re.search(r'\{[^{}]+\}', judge_response, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            scores = {"overall": 5.0, "relevance": 5, "accuracy": 5, "completeness": 5, "clarity": 5}
    except:
        scores = {"overall": 5.0, "relevance": 5, "accuracy": 5, "completeness": 5, "clarity": 5}

    # Update trace with output
    eval_trace.update(output={
        "scores": scores,
        "evaluation_complete": True
    })

    # Add scores to original trace
    overall = float(scores.get("overall", 5)) / 10.0
    langfuse.score(trace_id=trace_id, name="llm_judge_overall", value=overall,
                   comment=scores.get("explanation", "LLM judge evaluation"))

    for metric in ["relevance", "accuracy", "completeness", "clarity"]:
        if metric in scores:
            langfuse.score(trace_id=trace_id, name=f"llm_judge_{metric}",
                          value=float(scores[metric]) / 10.0)

    langfuse.flush()
    return scores


def custom_evaluators(result: Dict, expected_keywords: List[str]):
    """Custom evaluators - creates keyword_coverage, response_time, context_utilization scores."""
    trace_id = result['trace_id']
    answer = result['answer'].lower()

    # 1. Keyword Coverage
    keywords_found = sum(1 for kw in expected_keywords if kw.lower() in answer)
    keyword_score = keywords_found / len(expected_keywords) if expected_keywords else 0
    langfuse.score(
        trace_id=trace_id,
        name="keyword_coverage",
        value=keyword_score,
        comment=f"Found {keywords_found}/{len(expected_keywords)} keywords"
    )

    # 2. Response Time
    total_time = result['metrics']['total_time']
    if total_time < 2:
        time_score = 1.0
    elif total_time < 5:
        time_score = 0.8
    elif total_time < 10:
        time_score = 0.6
    else:
        time_score = max(0.2, 1.0 - total_time / 30)
    langfuse.score(
        trace_id=trace_id,
        name="response_time",
        value=time_score,
        comment=f"Total time: {total_time:.2f}s"
    )

    # 3. Context Utilization
    context_words = set(result['context'].lower().split())
    answer_words = set(answer.split())
    overlap = len(context_words.intersection(answer_words))
    context_score = min(1.0, overlap / 20)
    langfuse.score(
        trace_id=trace_id,
        name="context_utilization",
        value=context_score,
        comment=f"Word overlap: {overlap}"
    )

    langfuse.flush()

    return {
        "keyword_coverage": keyword_score,
        "response_time": time_score,
        "context_utilization": context_score
    }


def create_dataset():
    """Create evaluation dataset."""
    dataset_name = "rag_evaluation_dataset"
    try:
        langfuse.create_dataset(
            name=dataset_name,
            description="RAG system evaluation dataset"
        )
        print(f"Created dataset: {dataset_name}")
    except:
        print(f"Dataset {dataset_name} already exists")

    for item in TEST_ITEMS:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input={"question": item["question"]},
            expected_output=item["expected"],
            metadata={"keywords": item["keywords"]}
        )

    langfuse.flush()
    print(f"Added {len(TEST_ITEMS)} items to dataset")


def main():
    print("=" * 60)
    print("Full RAG Demo with All Langfuse Evaluators")
    print("=" * 60)

    # Load documents
    print("\n[1/4] Loading documents...")
    vectorstore = load_documents()

    # Create dataset
    print("\n[2/4] Creating evaluation dataset...")
    create_dataset()

    # Run queries
    print("\n[3/4] Running RAG queries...")
    session_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = []

    for item in TEST_ITEMS:
        print(f"\n  Question: {item['question']}")
        result = rag_query(item['question'], vectorstore, session_id)
        result['expected'] = item['expected']
        result['keywords'] = item['keywords']
        results.append(result)
        print(f"  Answer: {result['answer'][:100]}...")
        print(f"  Trace ID: {result['trace_id']}")

    # Run evaluations
    print("\n[4/4] Running evaluations...")

    for i, result in enumerate(results):
        print(f"\n  Evaluating query {i+1}...")

        # LLM-as-Judge
        print("    - LLM-as-Judge...")
        judge_scores = llm_judge_evaluate(
            result['question'],
            result['expected'],
            result['answer'],
            result['trace_id']
        )
        print(f"      Overall: {judge_scores.get('overall', 'N/A')}/10")

        # Custom evaluators
        print("    - Custom evaluators...")
        custom_scores = custom_evaluators(result, result['keywords'])
        print(f"      Keyword coverage: {custom_scores['keyword_coverage']:.2f}")
        print(f"      Response time: {custom_scores['response_time']:.2f}")
        print(f"      Context utilization: {custom_scores['context_utilization']:.2f}")

    langfuse.flush()

    print("\n" + "=" * 60)
    print("COMPLETE! Check Langfuse for all metrics:")
    print("=" * 60)
    print(f"\nDashboard: {LANGFUSE_HOST}")
    print("\nScores to look for:")
    print("  - llm_judge_overall (LLM-as-Judge)")
    print("  - llm_judge_relevance")
    print("  - llm_judge_accuracy")
    print("  - llm_judge_completeness")
    print("  - llm_judge_clarity")
    print("  - keyword_coverage (Custom)")
    print("  - response_time (Custom)")
    print("  - context_utilization (Custom)")
    print("  - response_quality (Auto)")
    print("  - indexing_success")
    print(f"\nTrace IDs:")
    for r in results:
        print(f"  - {r['trace_id']}")


if __name__ == "__main__":
    main()
