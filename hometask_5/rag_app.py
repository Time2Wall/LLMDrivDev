"""
RAG Application with Ollama and Langfuse Tracing (v2 API)
=========================================================
A Retrieval-Augmented Generation system that:
- Loads and indexes documents
- Answers questions using local Ollama LLM
- Tracks all operations with Langfuse observability (v2 API)
"""

import os
import time
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

# Langfuse for observability (v2 API)
from langfuse import Langfuse

# LangChain components
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()


class RAGApplication:
    """
    RAG Application with full Langfuse v3 instrumentation.

    Tracks:
    - Traces: Full execution path of each query (via spans)
    - Spans: Individual operations (embedding, retrieval, generation)
    - Generations: LLM calls with token usage
    - Events: Key events (document loading, errors)
    - Scores: Quality metrics
    """

    def __init__(
        self,
        ollama_model: str = None,
        ollama_base_url: str = None,
        langfuse_host: str = None,
        langfuse_public_key: str = None,
        langfuse_secret_key: str = None,
        user_id: str = None
    ):
        # Configuration
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"

        # Initialize Langfuse
        self.langfuse = Langfuse(
            host=langfuse_host or os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            public_key=langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
        )

        # Initialize LLM
        self.llm = OllamaLLM(
            model=self.ollama_model,
            base_url=self.ollama_base_url,
            temperature=0.7,
        )

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=self.ollama_model,
            base_url=self.ollama_base_url,
        )

        # Vector store (will be initialized when documents are loaded)
        self.vectorstore = None
        self.retriever = None

        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # RAG prompt template
        self.prompt_template = PromptTemplate(
            template="""Use the following context to answer the question.
If you cannot find the answer in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )

        print(f"RAG Application initialized with model: {self.ollama_model}")
        print(f"User ID: {self.user_id}")

    def load_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> int:
        """
        Load and index documents with Langfuse tracing.

        Args:
            texts: List of document texts to index
            metadatas: Optional metadata for each document

        Returns:
            Number of chunks created
        """
        # Create a trace for document loading (v2 API)
        trace = self.langfuse.trace(
            name="document_loading",
            user_id=self.user_id,
            input={"num_documents": len(texts)},
            metadata={
                "model": self.ollama_model
            }
        )

        try:
            # Event: Start loading
            trace.event(
                name="loading_started",
                metadata={"document_count": len(texts)}
            )

            # Span: Text splitting
            split_span = trace.span(
                name="text_splitting",
                input={"num_texts": len(texts), "total_chars": sum(len(t) for t in texts)}
            )

            start_time = time.time()

            # Create documents
            documents = []
            for i, text in enumerate(texts):
                meta = metadatas[i] if metadatas else {"source": f"document_{i}"}
                documents.append(Document(page_content=text, metadata=meta))

            # Split documents
            chunks = self.text_splitter.split_documents(documents)

            split_time = time.time() - start_time
            split_span.end(
                output={
                    "num_chunks": len(chunks),
                    "avg_chunk_size": sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
                },
                metadata={"duration_seconds": split_time}
            )

            # Span: Embedding and indexing
            embed_span = trace.span(
                name="embedding_and_indexing",
                input={"num_chunks": len(chunks)}
            )

            start_time = time.time()

            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name="rag_documents"
            )

            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )

            embed_time = time.time() - start_time
            embed_span.end(
                output={"status": "success"},
                metadata={"duration_seconds": embed_time}
            )

            # Event: Loading completed
            trace.event(
                name="loading_completed",
                metadata={
                    "chunks_created": len(chunks),
                    "total_time_seconds": split_time + embed_time
                }
            )

            # Score: Document processing quality
            trace.score(
                name="document_processing",
                value=1.0,
                comment=f"Successfully processed {len(texts)} documents into {len(chunks)} chunks"
            )

            trace.update(output={"chunks_created": len(chunks), "status": "success"})

            print(f"Loaded {len(texts)} documents, created {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            trace.event(
                name="loading_error",
                level="ERROR",
                metadata={"error": str(e)}
            )
            trace.score(
                name="document_processing",
                value=0.0,
                comment=f"Error: {str(e)}"
            )
            raise

        finally:
            self.langfuse.flush()

    def query(
        self,
        question: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system with full Langfuse tracing.

        Args:
            question: The question to answer
            session_id: Optional session ID for grouping queries

        Returns:
            Dictionary with answer and metadata
        """
        if not self.vectorstore:
            raise ValueError("No documents loaded. Call load_documents() first.")

        session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"

        # Create main trace for the query (v2 API)
        trace = self.langfuse.trace(
            name="rag_query",
            user_id=self.user_id,
            session_id=session_id,
            input={"question": question},
            metadata={
                "model": self.ollama_model,
                "question_length": len(question)
            }
        )

        total_start_time = time.time()

        try:
            # Span: Document retrieval
            retrieval_span = trace.span(
                name="document_retrieval",
                input={"question": question}
            )

            retrieval_start = time.time()

            # Retrieve relevant documents
            docs = self.retriever.invoke(question)

            retrieval_time = time.time() - retrieval_start

            # Calculate relevance scores (approximation)
            context = "\n\n".join([doc.page_content for doc in docs])

            retrieval_span.end(
                output={
                    "num_documents": len(docs),
                    "context_length": len(context),
                    "sources": [doc.metadata.get("source", "unknown") for doc in docs]
                },
                metadata={"duration_seconds": retrieval_time}
            )

            # Event: Retrieval completed
            trace.event(
                name="retrieval_completed",
                metadata={
                    "documents_found": len(docs),
                    "retrieval_time": retrieval_time
                }
            )

            # Generation with LLM call (v2 API)
            generation = trace.generation(
                name="llm_generation",
                model=self.ollama_model,
                input={
                    "question": question,
                    "context_length": len(context)
                },
                metadata={"prompt_template": "rag_qa"}
            )

            generation_start = time.time()

            # Format prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )

            # Call LLM
            answer = self.llm.invoke(prompt)

            generation_time = time.time() - generation_start

            # Estimate token usage (rough approximation for Ollama)
            input_tokens = len(prompt.split()) * 1.3  # Rough estimation
            output_tokens = len(answer.split()) * 1.3

            generation.end(
                output=answer,
                usage={
                    "input": int(input_tokens),
                    "output": int(output_tokens),
                    "total": int(input_tokens + output_tokens)
                },
                metadata={"duration_seconds": generation_time}
            )

            total_time = time.time() - total_start_time

            # Update trace with output
            trace.update(
                output=answer,
                metadata={
                    "total_time_seconds": total_time,
                    "retrieval_time_seconds": retrieval_time,
                    "generation_time_seconds": generation_time
                }
            )

            # Score: Response quality (can be updated later with actual evaluation)
            trace.score(
                name="response_completeness",
                value=1.0 if len(answer) > 20 else 0.5,
                comment="Auto-scored based on response length"
            )

            # Prepare result
            result = {
                "answer": answer,
                "question": question,
                "sources": [doc.metadata for doc in docs],
                "context_used": context,
                "metrics": {
                    "total_time_seconds": total_time,
                    "retrieval_time_seconds": retrieval_time,
                    "generation_time_seconds": generation_time,
                    "documents_retrieved": len(docs),
                    "context_length": len(context),
                    "estimated_input_tokens": int(input_tokens),
                    "estimated_output_tokens": int(output_tokens)
                },
                "trace_id": trace.id
            }

            return result

        except Exception as e:
            trace.event(
                name="query_error",
                level="ERROR",
                metadata={"error": str(e)}
            )
            trace.score(
                name="response_completeness",
                value=0.0,
                comment=f"Error: {str(e)}"
            )
            raise

        finally:
            self.langfuse.flush()

    def score_response(
        self,
        trace_id: str,
        score_name: str,
        score_value: float,
        comment: Optional[str] = None
    ):
        """
        Add a score to an existing trace (for human evaluation or LLM-as-judge).

        Args:
            trace_id: The trace ID to score
            score_name: Name of the score metric
            score_value: Score value (0.0 to 1.0)
            comment: Optional comment explaining the score
        """
        self.langfuse.create_score(
            trace_id=trace_id,
            name=score_name,
            value=score_value,
            comment=comment
        )
        self.langfuse.flush()
        print(f"Score '{score_name}' = {score_value} added to trace {trace_id}")

    def flush(self):
        """Flush all pending Langfuse events."""
        self.langfuse.flush()


# Sample documents for testing
SAMPLE_DOCUMENTS = [
    """
    Machine Learning Fundamentals

    Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. There are three main
    types of machine learning:

    1. Supervised Learning: The algorithm learns from labeled training data. Examples include
       classification (predicting categories) and regression (predicting continuous values).
       Common algorithms: Linear Regression, Decision Trees, Random Forest, SVM.

    2. Unsupervised Learning: The algorithm finds patterns in unlabeled data. Examples include
       clustering (grouping similar items) and dimensionality reduction.
       Common algorithms: K-Means, PCA, DBSCAN.

    3. Reinforcement Learning: The algorithm learns through trial and error, receiving rewards
       or penalties for actions taken. Used in robotics, game playing, and autonomous vehicles.
    """,

    """
    Deep Learning and Neural Networks

    Deep learning is a subset of machine learning based on artificial neural networks.
    Key concepts include:

    - Neurons and Layers: Neural networks consist of interconnected nodes (neurons)
      organized in layers: input, hidden, and output layers.

    - Activation Functions: Functions like ReLU, Sigmoid, and Tanh introduce non-linearity,
      allowing networks to learn complex patterns.

    - Backpropagation: The algorithm used to train neural networks by calculating gradients
      and adjusting weights to minimize the loss function.

    - Common Architectures:
      * CNNs (Convolutional Neural Networks): Best for image processing
      * RNNs (Recurrent Neural Networks): Best for sequential data
      * Transformers: State-of-the-art for NLP tasks, basis for GPT and BERT
    """,

    """
    Large Language Models (LLMs)

    Large Language Models are AI systems trained on vast amounts of text data. Key aspects:

    - Architecture: Most modern LLMs use the Transformer architecture, which relies on
      self-attention mechanisms to process text efficiently.

    - Training: LLMs are trained on massive datasets using unsupervised learning, often
      followed by fine-tuning with human feedback (RLHF).

    - Capabilities: Text generation, summarization, translation, question answering,
      code generation, and reasoning tasks.

    - Examples: GPT-4, Claude, LLaMA, Qwen, Mistral

    - RAG (Retrieval-Augmented Generation): A technique that combines LLMs with external
      knowledge retrieval to provide more accurate and up-to-date responses.
    """
]


def main():
    """Main function demonstrating the RAG application."""
    print("=" * 60)
    print("RAG Application with Ollama and Langfuse")
    print("=" * 60)

    # Initialize the application
    app = RAGApplication()

    # Load sample documents
    print("\n--- Loading Documents ---")
    metadatas = [
        {"source": "ml_fundamentals.txt", "topic": "machine_learning"},
        {"source": "deep_learning.txt", "topic": "deep_learning"},
        {"source": "llm_overview.txt", "topic": "llm"}
    ]
    num_chunks = app.load_documents(SAMPLE_DOCUMENTS, metadatas)
    print(f"Created {num_chunks} chunks from {len(SAMPLE_DOCUMENTS)} documents")

    # Test queries
    test_questions = [
        "What are the three types of machine learning?",
        "What is backpropagation and how does it work?",
        "What is RAG and why is it useful for LLMs?",
    ]

    session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n--- Running Test Queries ---")
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuery {i}: {question}")
        print("-" * 40)

        result = app.query(question, session_id=session_id)

        print(f"Answer: {result['answer'][:500]}...")
        print(f"\nMetrics:")
        print(f"  - Total time: {result['metrics']['total_time_seconds']:.2f}s")
        print(f"  - Retrieval time: {result['metrics']['retrieval_time_seconds']:.2f}s")
        print(f"  - Generation time: {result['metrics']['generation_time_seconds']:.2f}s")
        print(f"  - Documents retrieved: {result['metrics']['documents_retrieved']}")
        print(f"  - Est. tokens: {result['metrics']['estimated_input_tokens']} in / {result['metrics']['estimated_output_tokens']} out")
        print(f"  - Trace ID: {result['trace_id']}")

    # Flush remaining events
    app.flush()

    print("\n" + "=" * 60)
    print("Test completed! Check Langfuse UI for detailed traces.")
    print("=" * 60)


if __name__ == "__main__":
    main()
