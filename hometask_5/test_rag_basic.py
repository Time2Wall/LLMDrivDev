"""
Basic RAG Test Script (without Langfuse)
========================================
Tests the RAG functionality with local Ollama.
Run this first to verify Ollama and RAG are working.
"""

import time
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:1.5b"


def test_ollama_connection():
    """Test basic Ollama connection."""
    print("Testing Ollama connection...")
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        response = llm.invoke("Say 'Hello, I am working!' in one sentence.")
        print(f"  LLM Response: {response}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_embeddings():
    """Test Ollama embeddings."""
    print("\nTesting embeddings...")
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        result = embeddings.embed_query("Test embedding")
        print(f"  Embedding dimension: {len(result)}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_rag_pipeline():
    """Test full RAG pipeline."""
    print("\nTesting RAG pipeline...")

    # Sample document
    documents = [
        Document(
            page_content="""
            Python is a high-level programming language known for its simplicity and readability.
            It was created by Guido van Rossum and first released in 1991.
            Python supports multiple programming paradigms including procedural, object-oriented,
            and functional programming. It is widely used in web development, data science,
            machine learning, and automation.
            """,
            metadata={"source": "python_intro.txt"}
        )
    ]

    # Initialize components
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    # Create vector store
    print("  Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="test_collection"
    )

    # Test retrieval
    question = "Who created Python and when was it released?"
    print(f"  Question: {question}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(question)
    print(f"  Retrieved {len(docs)} documents")

    # Build context and answer
    context = "\n".join([doc.page_content for doc in docs])

    prompt = PromptTemplate(
        template="""Answer the question based on the context.

Context: {context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    full_prompt = prompt.format(context=context, question=question)

    print("  Generating answer...")
    start = time.time()
    answer = llm.invoke(full_prompt)
    elapsed = time.time() - start

    print(f"  Answer: {answer}")
    print(f"  Generation time: {elapsed:.2f}s")

    return True


def main():
    print("=" * 60)
    print("RAG Basic Test Suite")
    print("=" * 60)

    results = {}

    # Test 1: Ollama Connection
    results["ollama"] = test_ollama_connection()

    # Test 2: Embeddings
    if results["ollama"]:
        results["embeddings"] = test_embeddings()
    else:
        print("\nSkipping embeddings test (Ollama not connected)")
        results["embeddings"] = False

    # Test 3: RAG Pipeline
    if results["embeddings"]:
        results["rag"] = test_rag_pipeline()
    else:
        print("\nSkipping RAG test (embeddings not working)")
        results["rag"] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for test, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test}: {status}")

    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))

    return all_passed


if __name__ == "__main__":
    main()
