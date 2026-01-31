"""Quick test script to verify RAG system components."""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import chromadb
        print("  - chromadb: OK")
    except ImportError as e:
        print(f"  - chromadb: FAILED ({e})")
        return False

    try:
        from pymilvus import MilvusClient
        print("  - pymilvus: OK")
    except ImportError as e:
        print(f"  - pymilvus: FAILED ({e})")
        return False

    try:
        from sentence_transformers import SentenceTransformer
        print("  - sentence-transformers: OK")
    except ImportError as e:
        print(f"  - sentence-transformers: FAILED ({e})")
        return False

    try:
        import ollama
        print("  - ollama: OK")
    except ImportError as e:
        print(f"  - ollama: FAILED ({e})")
        return False

    try:
        import pandas
        import numpy
        import tqdm
        print("  - pandas, numpy, tqdm: OK")
    except ImportError as e:
        print(f"  - utilities: FAILED ({e})")
        return False

    return True

def test_embedding_generation():
    """Test embedding generation."""
    print("\nTesting embedding generation...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode("Test sentence for embedding")
        print(f"  - Embedding dimension: {len(embedding)}")
        print("  - Embedding generation: OK")
        return True
    except Exception as e:
        print(f"  - Embedding generation: FAILED ({e})")
        return False

def test_chromadb():
    """Test ChromaDB operations."""
    print("\nTesting ChromaDB...")
    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.Client(Settings(anonymized_telemetry=False))

        # Create collection
        collection = client.create_collection(name="test_collection")

        # Add documents
        collection.add(
            ids=["doc1", "doc2"],
            embeddings=[[0.1] * 384, [0.2] * 384],
            documents=["Test document 1", "Test document 2"],
            metadatas=[{"category": "test"}, {"category": "test"}]
        )

        # Search
        results = collection.query(
            query_embeddings=[[0.15] * 384],
            n_results=2
        )

        print(f"  - Collection created and queried successfully")
        print(f"  - Found {len(results['ids'][0])} documents")
        print("  - ChromaDB: OK")
        return True
    except Exception as e:
        print(f"  - ChromaDB: FAILED ({e})")
        return False

def test_ollama_connection():
    """Test Ollama connection."""
    print("\nTesting Ollama connection...")
    try:
        import ollama
        models = ollama.list()
        print(f"  - Ollama connected successfully")
        # Handle different response formats
        model_list = models.get('models', []) if isinstance(models, dict) else getattr(models, 'models', [])
        if model_list:
            model_names = []
            for m in model_list[:5]:
                if isinstance(m, dict):
                    model_names.append(m.get('name', m.get('model', 'unknown')))
                else:
                    model_names.append(getattr(m, 'model', getattr(m, 'name', str(m))))
            print(f"  - Available models: {model_names}")
        else:
            print("  - No models installed (run: ollama pull llama3.2)")
        print("  - Ollama: OK")
        return True
    except Exception as e:
        print(f"  - Ollama: FAILED ({e})")
        print("  - Make sure Ollama is running: ollama serve")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("RAG SYSTEM COMPONENT TESTS")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Embeddings", test_embedding_generation()))
    results.append(("ChromaDB", test_chromadb()))
    results.append(("Ollama", test_ollama_connection()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! RAG system is ready to use.")
        print("\nRun the full system with:")
        print("  python rag_system.py")
        print("  OR")
        print("  jupyter notebook rag_system.ipynb")
    else:
        print("Some tests failed. Please check the output above.")
        if not results[3][1]:  # Ollama test
            print("\nNote: Ollama is optional. The system will use fallback")
            print("      answer generation if Ollama is not available.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
