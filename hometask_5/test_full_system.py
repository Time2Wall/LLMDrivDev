"""
Full System Test with Langfuse Integration
==========================================
Tests the complete RAG + Langfuse observability system.

Requirements:
1. Ollama running with qwen2.5:1.5b model
2. Langfuse configured (local or cloud)

Usage:
  python test_full_system.py

Environment variables (or .env file):
  LANGFUSE_HOST=http://localhost:3000 (or https://cloud.langfuse.com)
  LANGFUSE_PUBLIC_KEY=pk-lf-...
  LANGFUSE_SECRET_KEY=sk-lf-...
"""

import os
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv

# Load environment
load_dotenv()


def check_ollama() -> bool:
    """Check if Ollama is accessible."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            print(f"  Available models: {model_names}")
            return any("qwen2.5" in m or "gemma" in m for m in model_names)
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def check_langfuse() -> bool:
    """Check if Langfuse is configured."""
    host = os.getenv("LANGFUSE_HOST", "")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")

    print(f"  Host: {host}")
    print(f"  Public Key: {'configured' if public_key and not public_key.startswith('pk-lf-your') else 'NOT SET'}")
    print(f"  Secret Key: {'configured' if secret_key and not secret_key.startswith('sk-lf-your') else 'NOT SET'}")

    if not host or not public_key or not secret_key:
        return False

    if public_key.startswith("pk-lf-your") or secret_key.startswith("sk-lf-your"):
        return False

    # Try to connect using v2 API
    try:
        from langfuse import Langfuse
        client = Langfuse(
            host=host,
            public_key=public_key,
            secret_key=secret_key
        )
        # Test connection by creating a simple trace (v2 API)
        trace = client.trace(name="connection_test", user_id="test")
        client.flush()
        print("  Connection test: SUCCESS")
        return True
    except Exception as e:
        print(f"  Connection test: FAILED - {e}")
        return False


def run_rag_test() -> Dict[str, Any]:
    """Run a complete RAG test with Langfuse tracing (v2 API)."""
    from langfuse import Langfuse
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    # Configuration
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    USER_ID = f"test_user_{uuid.uuid4().hex[:6]}"

    # Initialize Langfuse
    langfuse = Langfuse(
        host=os.getenv("LANGFUSE_HOST"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    )

    # Initialize LLM and embeddings
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.7)
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    # Sample document
    test_doc = """
    Artificial Intelligence (AI) is the simulation of human intelligence by machines.
    Machine Learning is a subset of AI that enables systems to learn from data.
    Deep Learning uses neural networks with multiple layers.
    Natural Language Processing (NLP) focuses on human-computer language interaction.
    Computer Vision enables machines to interpret visual information.
    """

    # Create trace for document loading (v2 API)
    load_trace = langfuse.trace(
        name="test_document_loading",
        user_id=USER_ID,
        metadata={"test": True}
    )

    load_span = load_trace.span(name="splitting_and_indexing")

    # Split and index
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = [Document(page_content=test_doc, metadata={"source": "test.txt"})]
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"test_{uuid.uuid4().hex[:8]}"
    )

    load_span.end(output={"chunks": len(chunks)})
    load_trace.score(name="loading_success", value=1.0)

    # Query trace (v2 API)
    question = "What is the relationship between AI and Machine Learning?"

    query_trace = langfuse.trace(
        name="test_rag_query",
        user_id=USER_ID,
        input=question,
        metadata={"test": True}
    )

    # Retrieval span
    retrieval_span = query_trace.span(name="retrieval", input={"question": question})
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in retrieved_docs])
    retrieval_span.end(output={"docs": len(retrieved_docs), "context_len": len(context)})

    # Generation (v2 API)
    generation = query_trace.generation(
        name="answer_generation",
        model=OLLAMA_MODEL,
        input={"question": question, "context_len": len(context)}
    )

    prompt = f"""Answer based on context.

Context: {context}

Question: {question}

Answer:"""

    start = time.time()
    answer = llm.invoke(prompt)
    gen_time = time.time() - start

    generation.end(
        output=answer,
        usage={"input": len(prompt.split()), "output": len(answer.split())},
        metadata={"duration": gen_time}
    )

    query_trace.update(output=answer)
    query_trace.score(name="response_quality", value=0.8, comment="Test query")

    # Add event
    query_trace.event(name="test_completed", metadata={"success": True})

    # Create dataset item for testing
    try:
        langfuse.create_dataset(
            name="test_evaluation_dataset",
            description="Test dataset for system validation"
        )
    except:
        pass  # Dataset may already exist

    langfuse.create_dataset_item(
        dataset_name="test_evaluation_dataset",
        input={"question": question},
        expected_output="Machine Learning is a subset of AI",
        metadata={"test_run": True}
    )

    langfuse.flush()

    return {
        "success": True,
        "question": question,
        "answer": answer[:200],
        "trace_id": query_trace.id,
        "generation_time": gen_time,
        "chunks_created": len(chunks),
        "docs_retrieved": len(retrieved_docs)
    }


def main():
    print("=" * 70)
    print("Full System Test: RAG + Langfuse")
    print("=" * 70)
    print()

    # Step 1: Check Ollama
    print("[1/3] Checking Ollama...")
    ollama_ok = check_ollama()
    print(f"      Result: {'OK' if ollama_ok else 'FAILED'}")
    print()

    if not ollama_ok:
        print("ERROR: Ollama is not running or no suitable model found.")
        print("       Please run: ollama pull qwen2.5:1.5b")
        print("       Then: ollama serve")
        sys.exit(1)

    # Step 2: Check Langfuse
    print("[2/3] Checking Langfuse configuration...")
    langfuse_ok = check_langfuse()
    print(f"      Result: {'OK' if langfuse_ok else 'NOT CONFIGURED'}")
    print()

    if not langfuse_ok:
        print("WARNING: Langfuse is not properly configured.")
        print("         Please set up Langfuse:")
        print("         1. Use Langfuse Cloud: https://cloud.langfuse.com")
        print("         2. Or run locally with Docker: docker-compose -f docker-compose.langfuse.yml up -d")
        print("         3. Update .env file with your API keys")
        print()
        print("         Running basic RAG test without Langfuse...")
        print()

        # Run basic test
        os.system("python test_rag_basic.py")
        sys.exit(0)

    # Step 3: Run full test
    print("[3/3] Running full RAG + Langfuse test...")
    try:
        result = run_rag_test()

        print()
        print("=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(f"  Status: SUCCESS")
        print(f"  Question: {result['question']}")
        print(f"  Answer: {result['answer']}...")
        print(f"  Trace ID: {result['trace_id']}")
        print(f"  Generation Time: {result['generation_time']:.2f}s")
        print(f"  Chunks Created: {result['chunks_created']}")
        print(f"  Docs Retrieved: {result['docs_retrieved']}")
        print()
        print(f"View trace at: {os.getenv('LANGFUSE_HOST')}")
        print()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("All tests passed!")


if __name__ == "__main__":
    main()
