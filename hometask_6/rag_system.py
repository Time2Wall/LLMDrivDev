"""
RAG System with Vector Database Comparison

This script implements a complete RAG (Retrieval Augmented Generation) system with:
- Vector database setup and indexing (ChromaDB and Milvus)
- ANN algorithm comparison (IVF, HNSW, FLAT)
- Semantic search with various similarity metrics
- Metadata filtering and hybrid search
- Performance benchmarking
- End-to-end RAG pipeline
"""

import os
import time
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import Milvus (optional)
MILVUS_AVAILABLE = False
try:
    from pymilvus import MilvusClient
    # Test if milvus-lite is available
    _test_client = MilvusClient("./test_milvus_import.db")
    MILVUS_AVAILABLE = True
    print("Milvus Lite is available for ANN comparison")
except Exception as e:
    print(f"Milvus Lite not available: {e}")
    print("Using ChromaDB only (install 'pymilvus[milvus_lite]' for full ANN comparison)")


@dataclass
class Document:
    """Document class for storing text and metadata."""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class DatasetGenerator:
    """Generate a synthetic dataset about programming topics."""

    def __init__(self):
        self.categories = [
            "python", "javascript", "machine_learning", "web_development",
            "databases", "devops", "security", "cloud_computing",
            "data_science", "algorithms"
        ]
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, List[str]]:
        """Load document templates for each category."""
        templates = {
            "python": [
                "Python is a versatile programming language known for its {feature}. It excels in {use_case} and provides {benefit}. Key libraries include {libraries}.",
                "When working with Python, understanding {concept} is essential. This allows developers to {action} efficiently. Best practices include {practice}.",
                "Python's {module} module provides functionality for {purpose}. It supports {features} and is commonly used in {applications}.",
                "Advanced Python techniques like {technique} enable {capability}. This is particularly useful for {scenario} where {requirement} is needed.",
                "Python decorators are powerful tools for {decorator_use}. They wrap functions to add {functionality} without modifying the original code."
            ],
            "javascript": [
                "JavaScript enables {feature} in web applications. Modern frameworks like {framework} provide {benefit} for building {app_type}.",
                "Understanding {concept} in JavaScript is crucial for {purpose}. This enables developers to create {result} with {advantage}.",
                "The {api} API in JavaScript allows {functionality}. It's commonly used for {use_case} and supports {features}.",
                "JavaScript's async/await pattern simplifies {async_task}. Combined with {technology}, it enables {capability} in modern applications.",
                "ES6+ features like {feature} have transformed JavaScript development. They provide {benefit} and improve {aspect} of code."
            ],
            "machine_learning": [
                "Machine learning algorithm {algorithm} is used for {task}. It works by {mechanism} and achieves {performance} on {dataset_type} datasets.",
                "Neural networks with {architecture} excel at {task}. Training requires {requirement} and optimization using {optimizer}.",
                "Feature engineering for {domain} involves {technique}. This improves model {metric} by {improvement_method}.",
                "Transfer learning from {pretrained_model} enables {capability}. Fine-tuning on {task} requires {approach} for optimal results.",
                "Evaluating ML models requires understanding {metric}. Cross-validation with {strategy} ensures {benefit} and prevents {issue}."
            ],
            "web_development": [
                "Modern web development uses {technology} for {purpose}. This approach provides {benefit} and enables {feature}.",
                "RESTful APIs design principles include {principle}. Implementing {feature} ensures {benefit} for API consumers.",
                "Frontend frameworks like {framework} optimize {aspect}. They use {technique} for {purpose} and improve {metric}.",
                "Web security best practices include {practice}. Protecting against {vulnerability} requires {solution} implementation.",
                "Progressive Web Apps combine {feature1} and {feature2}. They provide {benefit} while maintaining {characteristic}."
            ],
            "databases": [
                "Database {db_type} excels at {use_case}. It provides {feature} and supports {capability} for {scenario}.",
                "Query optimization in {database} involves {technique}. Using {feature} improves {metric} by {factor}.",
                "Database indexing strategies for {scenario} include {strategy}. The {index_type} index provides {benefit} for {query_type} queries.",
                "ACID properties ensure {property} in databases. Transaction isolation levels like {level} prevent {issue}.",
                "NoSQL databases like {database} are designed for {use_case}. They offer {advantage} compared to {comparison}."
            ],
            "devops": [
                "CI/CD pipelines automate {process}. Using {tool} enables {benefit} and reduces {issue}.",
                "Container orchestration with {platform} manages {aspect}. It provides {feature} for {use_case}.",
                "Infrastructure as Code using {tool} ensures {benefit}. Configuration management becomes {characteristic}.",
                "Monitoring and observability with {tool} tracks {metric}. Alert systems detect {issue} and enable {response}.",
                "DevOps practices like {practice} improve {aspect}. Teams achieve {benefit} through {approach}."
            ],
            "security": [
                "Cybersecurity measures against {threat} include {solution}. Implementing {technique} protects {asset}.",
                "Authentication using {method} provides {level} security. Combined with {feature}, it prevents {attack}.",
                "Encryption algorithms like {algorithm} secure {data_type}. Key management requires {practice} for {compliance}.",
                "Security auditing involves {process}. Tools like {tool} identify {vulnerability} in {target}.",
                "Zero Trust architecture requires {principle}. Every {entity} must be {action} before {access}."
            ],
            "cloud_computing": [
                "Cloud service {service} provides {capability}. It enables {use_case} with {benefit} pricing model.",
                "Serverless computing with {platform} handles {workload}. Auto-scaling based on {trigger} optimizes {resource}.",
                "Cloud architecture pattern {pattern} addresses {challenge}. It provides {benefit} for {scenario}.",
                "Multi-cloud strategy using {providers} ensures {benefit}. Considerations include {factor} and {consideration}.",
                "Cloud cost optimization involves {technique}. Monitoring {metric} helps reduce {expense} by {amount}."
            ],
            "data_science": [
                "Data preprocessing involves {technique}. Handling {data_issue} requires {approach} for accurate {output}.",
                "Statistical analysis using {method} reveals {insight}. Hypothesis testing with {test} validates {claim}.",
                "Data visualization with {library} creates {chart_type}. Effective visualizations communicate {information}.",
                "Big data processing with {framework} handles {scale}. Distributed computing enables {capability}.",
                "A/B testing methodology requires {component}. Statistical significance at {level} ensures {validity}."
            ],
            "algorithms": [
                "Algorithm {algorithm} solves {problem} in O({complexity}) time. It uses {technique} for {optimization}.",
                "Data structure {structure} provides O({complexity}) for {operation}. It's ideal for {use_case}.",
                "Dynamic programming approach to {problem} reduces complexity from {initial} to {optimized}.",
                "Graph algorithm {algorithm} finds {result}. Applications include {application} and {use_case}.",
                "Sorting algorithm {algorithm} has {characteristic}. It performs best when {condition}."
            ]
        }
        return templates

    def _get_fill_values(self, category: str) -> Dict[str, List[str]]:
        """Get fill values for template placeholders by category."""
        # Comprehensive fill values for all categories
        fill_values = {
            "python": {
                "feature": ["clean syntax", "dynamic typing", "extensive standard library", "duck typing", "list comprehensions"],
                "use_case": ["data analysis", "web development", "automation", "machine learning", "scientific computing"],
                "benefit": ["rapid development", "code readability", "extensive ecosystem", "cross-platform compatibility"],
                "libraries": ["NumPy, Pandas, Matplotlib", "Django, Flask, FastAPI", "TensorFlow, PyTorch", "requests, beautifulsoup"],
                "concept": ["generators", "context managers", "decorators", "metaclasses", "descriptors"],
                "action": ["manage resources", "handle large datasets", "extend functionality", "create reusable code"],
                "practice": ["using virtual environments", "following PEP 8", "writing unit tests", "type hinting"],
                "module": ["asyncio", "collections", "itertools", "functools", "pathlib"],
                "purpose": ["asynchronous programming", "specialized containers", "efficient iteration", "functional programming"],
                "features": ["coroutines", "default dictionaries", "combinatorics", "higher-order functions"],
                "applications": ["network servers", "data processing", "algorithm implementation", "file handling"],
                "technique": ["metaclass programming", "monkey patching", "dependency injection", "aspect-oriented programming"],
                "capability": ["runtime class modification", "automatic logging", "loose coupling", "cross-cutting concerns"],
                "scenario": ["framework development", "debugging", "testing", "plugin systems"],
                "requirement": ["flexibility", "extensibility", "maintainability", "testability"],
                "decorator_use": ["logging", "caching", "authentication", "rate limiting", "validation"],
                "functionality": ["timing information", "memoization", "access control", "input validation"]
            },
            "javascript": {
                "feature": ["interactive UIs", "real-time updates", "client-side validation", "DOM manipulation"],
                "framework": ["React", "Vue.js", "Angular", "Svelte", "Next.js"],
                "benefit": ["component reusability", "virtual DOM", "state management", "server-side rendering"],
                "app_type": ["single-page applications", "progressive web apps", "mobile apps", "desktop applications"],
                "concept": ["closures", "prototypes", "event loop", "promises", "hoisting"],
                "purpose": ["managing scope", "object inheritance", "asynchronous operations", "error handling"],
                "result": ["responsive interfaces", "efficient code", "scalable applications", "maintainable systems"],
                "advantage": ["better performance", "cleaner code", "easier debugging", "improved UX"],
                "api": ["Fetch", "Web Storage", "WebSocket", "Service Worker", "Canvas"],
                "functionality": ["HTTP requests", "local data persistence", "real-time communication", "offline capabilities"],
                "use_case": ["API calls", "user preferences", "chat applications", "background sync"],
                "features": ["streaming", "IndexedDB", "binary data", "push notifications"],
                "async_task": ["API calls", "file operations", "database queries", "parallel processing"],
                "technology": ["Promise.all", "generators", "observables", "web workers"],
                "capability": ["non-blocking operations", "concurrent processing", "responsive UIs", "background tasks"],
                "aspect": ["readability", "maintainability", "performance", "developer experience"]
            },
            "machine_learning": {
                "algorithm": ["Random Forest", "XGBoost", "SVM", "k-NN", "Logistic Regression"],
                "task": ["classification", "regression", "clustering", "anomaly detection", "recommendation"],
                "mechanism": ["ensemble learning", "kernel tricks", "gradient boosting", "distance metrics"],
                "performance": ["high accuracy", "robust predictions", "fast inference", "interpretable results"],
                "dataset_type": ["tabular", "imbalanced", "high-dimensional", "time-series"],
                "architecture": ["CNN", "RNN", "Transformer", "GAN", "autoencoder"],
                "requirement": ["large datasets", "GPU resources", "careful hyperparameter tuning", "regularization"],
                "optimizer": ["Adam", "SGD", "RMSprop", "AdaGrad"],
                "domain": ["NLP", "computer vision", "time series", "tabular data"],
                "technique": ["normalization", "encoding", "dimensionality reduction", "feature selection"],
                "metric": ["accuracy", "F1-score", "AUC-ROC", "RMSE"],
                "improvement_method": ["capturing patterns", "reducing noise", "handling missing data", "creating interactions"],
                "pretrained_model": ["BERT", "ResNet", "GPT", "VGG", "EfficientNet"],
                "approach": ["gradual unfreezing", "learning rate scheduling", "data augmentation", "regularization"],
                "strategy": ["k-fold", "stratified", "time-series split", "leave-one-out"],
                "benefit": ["reliable estimates", "reduced variance", "better generalization"],
                "issue": ["overfitting", "data leakage", "selection bias"]
            },
            "web_development": {
                "technology": ["microservices", "GraphQL", "WebAssembly", "JAMstack", "Server Components"],
                "purpose": ["scalability", "flexible data fetching", "near-native performance", "static site generation"],
                "benefit": ["independent deployment", "reduced over-fetching", "fast execution", "improved SEO"],
                "feature": ["service discovery", "schema stitching", "memory safety", "edge computing"],
                "principle": ["statelessness", "resource naming", "HATEOAS", "versioning"],
                "framework": ["React", "Vue", "Angular", "Svelte"],
                "aspect": ["rendering", "state management", "routing", "code splitting"],
                "technique": ["virtual DOM", "reactivity", "lazy loading", "hydration"],
                "metric": ["Time to Interactive", "First Contentful Paint", "bundle size"],
                "practice": ["input validation", "HTTPS everywhere", "CSP headers", "secure cookies"],
                "vulnerability": ["XSS", "CSRF", "SQL injection", "clickjacking"],
                "solution": ["sanitization", "tokens", "parameterized queries", "frame options"],
                "feature1": ["offline support", "push notifications", "app-like experience"],
                "feature2": ["fast loading", "responsive design", "installability"],
                "characteristic": ["web accessibility", "cross-platform compatibility", "discoverability"]
            },
            "databases": {
                "db_type": ["PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra"],
                "use_case": ["complex queries", "document storage", "caching", "full-text search", "time-series data"],
                "feature": ["JSONB support", "flexible schema", "in-memory storage", "inverted indices", "wide-column storage"],
                "capability": ["full-text search", "horizontal scaling", "pub/sub", "aggregations", "tunable consistency"],
                "scenario": ["analytics workloads", "real-time applications", "session management", "log analysis"],
                "database": ["PostgreSQL", "MySQL", "MongoDB", "Cassandra"],
                "technique": ["query planning", "index optimization", "partitioning", "denormalization"],
                "factor": ["10x", "5x", "100x", "significant margin"],
                "strategy": ["B-tree indexes", "covering indexes", "partial indexes", "composite indexes"],
                "index_type": ["B-tree", "Hash", "GiST", "GIN"],
                "query_type": ["range", "equality", "spatial", "full-text"],
                "property": ["atomicity", "consistency", "isolation", "durability"],
                "level": ["Read Committed", "Repeatable Read", "Serializable"],
                "issue": ["dirty reads", "phantom reads", "lost updates"],
                "advantage": ["horizontal scalability", "flexible schema", "high availability"],
                "comparison": ["traditional RDBMS", "single-node databases", "rigid schemas"]
            },
            "devops": {
                "process": ["build and deployment", "testing", "security scanning", "infrastructure provisioning"],
                "tool": ["Jenkins", "GitHub Actions", "Terraform", "Prometheus", "Ansible"],
                "benefit": ["faster releases", "consistent environments", "early bug detection", "infrastructure visibility"],
                "issue": ["manual errors", "deployment failures", "configuration drift", "performance issues"],
                "platform": ["Kubernetes", "Docker Swarm", "ECS", "Nomad"],
                "aspect": ["container lifecycle", "service discovery", "load balancing", "auto-scaling"],
                "feature": ["rolling updates", "health checks", "resource limits", "secrets management"],
                "use_case": ["microservices", "batch jobs", "stateful applications", "edge computing"],
                "characteristic": ["reproducible", "version controlled", "auditable", "automated"],
                "metric": ["response times", "error rates", "resource utilization", "throughput"],
                "response": ["auto-remediation", "on-call notification", "scaling actions"],
                "practice": ["continuous integration", "infrastructure as code", "GitOps", "chaos engineering"],
                "approach": ["automation", "collaboration", "measurement", "sharing"]
            },
            "security": {
                "threat": ["SQL injection", "XSS attacks", "DDoS", "phishing", "ransomware"],
                "solution": ["parameterized queries", "input sanitization", "rate limiting", "security training", "backups"],
                "technique": ["defense in depth", "least privilege", "security headers", "encryption at rest"],
                "asset": ["user data", "API endpoints", "database", "file systems"],
                "method": ["OAuth 2.0", "JWT tokens", "multi-factor", "biometrics", "SSO"],
                "level": ["enterprise-grade", "high", "military-grade", "compliant"],
                "feature": ["refresh tokens", "device fingerprinting", "risk-based auth"],
                "attack": ["session hijacking", "brute force", "credential stuffing", "man-in-the-middle"],
                "algorithm": ["AES-256", "RSA", "ChaCha20", "Argon2"],
                "data_type": ["data at rest", "data in transit", "passwords", "PII"],
                "practice": ["key rotation", "HSM usage", "secure key storage"],
                "compliance": ["GDPR", "HIPAA", "PCI-DSS", "SOC 2"],
                "process": ["penetration testing", "code review", "vulnerability scanning", "threat modeling"],
                "vulnerability": ["outdated dependencies", "misconfigurations", "exposed secrets"],
                "target": ["applications", "networks", "containers", "cloud infrastructure"],
                "principle": ["never trust, always verify", "micro-segmentation", "continuous validation"],
                "entity": ["user", "device", "application", "network flow"],
                "action": ["authenticated", "authorized", "validated"],
                "access": ["granting access", "allowing connection", "permitting action"]
            },
            "cloud_computing": {
                "service": ["EC2", "Lambda", "S3", "RDS", "CloudFront"],
                "capability": ["scalable compute", "serverless execution", "object storage", "managed databases", "CDN"],
                "use_case": ["web hosting", "event processing", "data lakes", "OLTP workloads", "static assets"],
                "benefit": ["pay-per-use", "per-invocation", "durability", "automated backups", "global distribution"],
                "platform": ["AWS Lambda", "Azure Functions", "Google Cloud Functions", "Cloudflare Workers"],
                "workload": ["API backends", "data processing", "scheduled tasks", "webhooks"],
                "trigger": ["HTTP requests", "queue messages", "file uploads", "schedules"],
                "resource": ["costs", "cold starts", "memory usage", "execution time"],
                "pattern": ["event sourcing", "CQRS", "saga", "strangler fig"],
                "challenge": ["data consistency", "read/write separation", "distributed transactions", "legacy migration"],
                "providers": ["AWS and Azure", "GCP and AWS", "multi-cloud"],
                "factor": ["data sovereignty", "vendor lock-in", "cost optimization"],
                "consideration": ["networking complexity", "skill requirements", "tooling differences"],
                "technique": ["right-sizing", "reserved instances", "spot instances", "auto-scaling"],
                "metric": ["unused resources", "data transfer costs", "storage tiers"],
                "expense": ["compute costs", "storage costs", "network egress"],
                "amount": ["30-50%", "significant amounts", "up to 70%"]
            },
            "data_science": {
                "technique": ["normalization", "imputation", "encoding", "outlier detection", "feature scaling"],
                "data_issue": ["missing values", "outliers", "categorical variables", "skewed distributions"],
                "approach": ["mean/median imputation", "IQR method", "one-hot encoding", "log transformation"],
                "output": ["analysis", "model training", "predictions", "insights"],
                "method": ["regression analysis", "hypothesis testing", "ANOVA", "correlation analysis"],
                "insight": ["relationships between variables", "significant differences", "trends", "patterns"],
                "test": ["t-test", "chi-square", "Mann-Whitney U", "Kolmogorov-Smirnov"],
                "claim": ["statistical significance", "effect size", "confidence intervals"],
                "library": ["Matplotlib", "Seaborn", "Plotly", "Altair"],
                "chart_type": ["scatter plots", "heatmaps", "interactive dashboards", "time series charts"],
                "information": ["patterns", "outliers", "distributions", "relationships"],
                "framework": ["Spark", "Dask", "Flink", "Beam"],
                "scale": ["petabyte-scale data", "streaming data", "distributed datasets"],
                "capability": ["parallel processing", "real-time analytics", "fault tolerance"],
                "component": ["proper sample size", "randomization", "control groups", "clear metrics"],
                "level": ["p < 0.05", "95% confidence", "p < 0.01"],
                "validity": ["reliable conclusions", "actionable insights", "business decisions"]
            },
            "algorithms": {
                "algorithm": ["QuickSort", "Dijkstra's", "A*", "Binary Search", "Merge Sort"],
                "problem": ["sorting", "shortest path", "pathfinding", "searching", "divide and conquer"],
                "complexity": ["n log n", "V + E log V", "b^d", "log n", "n log n"],
                "technique": ["partitioning", "relaxation", "heuristics", "divide and conquer"],
                "optimization": ["in-place sorting", "early termination", "pruning", "memoization"],
                "structure": ["Hash Table", "Binary Search Tree", "Heap", "Trie", "Graph"],
                "operation": ["lookup", "insertion/deletion", "extract-min", "prefix search", "traversal"],
                "use_case": ["caching", "ordered data", "priority queues", "autocomplete", "network routing"],
                "initial": ["O(2^n)", "O(n!)", "O(n^2)", "exponential"],
                "optimized": ["O(n*W)", "O(n^2)", "O(n log n)", "polynomial"],
                "result": ["shortest paths", "minimum spanning tree", "strongly connected components", "topological order"],
                "application": ["GPS navigation", "network design", "dependency resolution", "scheduling"],
                "characteristic": ["stability", "adaptiveness", "in-place operation", "cache efficiency"],
                "condition": ["data is nearly sorted", "memory is limited", "stability matters", "parallel execution is available"]
            }
        }
        return fill_values.get(category, {})

    def generate_document(self, doc_id: int, category: str) -> Document:
        """Generate a single document."""
        import random

        template = random.choice(self.templates[category])
        fill_values = self._get_fill_values(category)

        text = template
        for key, values in fill_values.items():
            placeholder = "{" + key + "}"
            if placeholder in text:
                text = text.replace(placeholder, random.choice(values), 1)

        difficulty = random.choice(["beginner", "intermediate", "advanced"])
        year = random.randint(2020, 2024)

        return Document(
            id=f"doc_{doc_id}",
            text=text,
            metadata={
                "category": category,
                "difficulty": difficulty,
                "year": year,
                "word_count": len(text.split())
            }
        )

    def generate_dataset(self, num_documents: int = 1200) -> List[Document]:
        """Generate a dataset with specified number of documents."""
        import random
        random.seed(42)

        documents = []
        docs_per_category = num_documents // len(self.categories)

        for category in self.categories:
            for i in range(docs_per_category):
                doc_id = len(documents)
                doc = self.generate_document(doc_id, category)
                documents.append(doc)

        random.shuffle(documents)
        return documents


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Loaded model: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.model.encode(query, convert_to_numpy=True)


class ChromaDBManager:
    """Manager for ChromaDB operations."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = None
        print(f"ChromaDB initialized at: {persist_directory}")

    def create_collection(self, name: str, distance_metric: str = "cosine") -> None:
        """Create a collection with specified distance metric."""
        try:
            self.client.delete_collection(name)
        except:
            pass

        self.collection = self.client.create_collection(
            name=name,
            metadata={"hnsw:space": distance_metric}
        )
        print(f"Created collection '{name}' with {distance_metric} distance metric")

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> float:
        """Add documents to the collection."""
        start_time = time.time()

        for i in tqdm(range(0, len(documents), batch_size), desc="Indexing documents"):
            batch = documents[i:i+batch_size]

            self.collection.add(
                ids=[doc.id for doc in batch],
                embeddings=[doc.embedding for doc in batch],
                documents=[doc.text for doc in batch],
                metadatas=[doc.metadata for doc in batch]
            )

        indexing_time = time.time() - start_time
        print(f"Indexed {len(documents)} documents in {indexing_time:.2f}s")
        return indexing_time

    def search(self, query_embedding: List[float], top_k: int = 5,
               where: Optional[Dict] = None, where_document: Optional[Dict] = None) -> Tuple[List[Dict], float]:
        """Search for similar documents."""
        start_time = time.time()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )

        search_time = time.time() - start_time

        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })

        return formatted_results, search_time

    def batch_search(self, query_embeddings: List[List[float]], top_k: int = 5) -> Tuple[List[List[Dict]], float]:
        """Perform batch search for multiple queries."""
        start_time = time.time()

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        search_time = time.time() - start_time

        all_results = []
        for q_idx in range(len(query_embeddings)):
            query_results = []
            for i in range(len(results['ids'][q_idx])):
                query_results.append({
                    'id': results['ids'][q_idx][i],
                    'document': results['documents'][q_idx][i],
                    'metadata': results['metadatas'][q_idx][i],
                    'distance': results['distances'][q_idx][i]
                })
            all_results.append(query_results)

        return all_results, search_time


# Only define MilvusManager if Milvus is available
if MILVUS_AVAILABLE:
    class MilvusManager:
        """Manager for Milvus operations with ANN algorithm comparison."""

        def __init__(self, db_path: str = "./milvus_lite.db"):
            self.client = MilvusClient(db_path)
            self.collection_name = None
            print(f"Milvus Lite initialized at: {db_path}")

        def create_collection(self, name: str, dim: int = 384, index_type: str = "HNSW",
                             metric_type: str = "COSINE", index_params: Optional[Dict] = None) -> float:
            """Create a collection with specified index type."""
            if self.client.has_collection(name):
                self.client.drop_collection(name)

            if index_params is None:
                index_params = self._get_default_index_params(index_type)

            start_time = time.time()

            self.client.create_collection(
                collection_name=name,
                dimension=dim,
                metric_type=metric_type,
                index_params={
                    "index_type": index_type,
                    "params": index_params
                }
            )

            self.collection_name = name
            creation_time = time.time() - start_time

            print(f"Created collection '{name}' with {index_type} index ({metric_type})")
            print(f"Index params: {index_params}")

            return creation_time

        def _get_default_index_params(self, index_type: str) -> Dict:
            """Get default parameters for each index type."""
            params = {
                "FLAT": {},
                "IVF_FLAT": {"nlist": 128},
                "IVF_SQ8": {"nlist": 128},
                "IVF_PQ": {"nlist": 128, "m": 8, "nbits": 8},
                "HNSW": {"M": 16, "efConstruction": 200},
                "ANNOY": {"n_trees": 8}
            }
            return params.get(index_type, {})

        def add_documents(self, documents: List[Document], batch_size: int = 100) -> float:
            """Add documents to the collection."""
            start_time = time.time()

            for i in tqdm(range(0, len(documents), batch_size), desc="Indexing documents"):
                batch = documents[i:i+batch_size]

                data = [
                    {
                        "id": idx + i,
                        "vector": doc.embedding,
                        "text": doc.text,
                        "category": doc.metadata["category"],
                        "difficulty": doc.metadata["difficulty"],
                        "year": doc.metadata["year"]
                    }
                    for idx, doc in enumerate(batch)
                ]

                self.client.insert(
                    collection_name=self.collection_name,
                    data=data
                )

            indexing_time = time.time() - start_time
            print(f"Indexed {len(documents)} documents in {indexing_time:.2f}s")
            return indexing_time

        def search(self, query_embedding: List[float], top_k: int = 5,
                   search_params: Optional[Dict] = None, filter_expr: Optional[str] = None) -> Tuple[List[Dict], float]:
            """Search for similar documents."""
            start_time = time.time()

            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                search_params=search_params or {"metric_type": "COSINE"},
                filter=filter_expr,
                output_fields=["text", "category", "difficulty", "year"]
            )

            search_time = time.time() - start_time

            formatted_results = []
            for hit in results[0]:
                formatted_results.append({
                    'id': hit['id'],
                    'document': hit['entity'].get('text', ''),
                    'metadata': {
                        'category': hit['entity'].get('category', ''),
                        'difficulty': hit['entity'].get('difficulty', ''),
                        'year': hit['entity'].get('year', 0)
                    },
                    'distance': hit['distance']
                })

            return formatted_results, search_time


class HybridSearchEngine:
    """Hybrid search combining vector similarity with metadata filtering and keyword boost."""

    def __init__(self, chroma_manager: ChromaDBManager, embedding_gen: EmbeddingGenerator):
        self.chroma = chroma_manager
        self.embedding_gen = embedding_gen

    def hybrid_search(self, query: str, top_k: int = 10,
                      category_filter: Optional[str] = None,
                      difficulty_filter: Optional[str] = None,
                      year_min: Optional[int] = None,
                      keyword_boost: Optional[List[str]] = None,
                      keyword_weight: float = 0.3) -> List[Dict]:
        """Perform hybrid search with multiple signals."""
        where_filter = None
        conditions = []

        if category_filter:
            conditions.append({"category": category_filter})
        if difficulty_filter:
            conditions.append({"difficulty": difficulty_filter})
        if year_min:
            conditions.append({"year": {"$gte": year_min}})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        query_embedding = self.embedding_gen.generate_query_embedding(query)
        results, search_time = self.chroma.search(
            query_embedding.tolist(),
            top_k=top_k * 2,
            where=where_filter
        )

        if keyword_boost:
            for result in results:
                text_lower = result['document'].lower()
                keyword_score = sum(1 for kw in keyword_boost if kw.lower() in text_lower)
                keyword_score = keyword_score / len(keyword_boost)

                vector_score = 1 - result['distance']
                combined_score = (1 - keyword_weight) * vector_score + keyword_weight * keyword_score
                result['combined_score'] = combined_score
                result['keyword_matches'] = keyword_score

            results = sorted(results, key=lambda x: x.get('combined_score', 0), reverse=True)

        return results[:top_k]


class RAGSystem:
    """Complete RAG System with document retrieval and LLM-based answer generation."""

    def __init__(self, vector_db: ChromaDBManager, embedding_gen: EmbeddingGenerator,
                 llm_provider: str = "ollama", model_name: str = "llama3.2:latest"):
        self.vector_db = vector_db
        self.embedding_gen = embedding_gen
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.client = None

        if llm_provider == "ollama":
            try:
                import ollama
                # Test connection to Ollama
                ollama.list()
                self.client = ollama
                print(f"RAG System initialized with Ollama (model: {model_name})")
            except Exception as e:
                print(f"Warning: Could not connect to Ollama: {e}")
                print("Make sure Ollama is running (ollama serve)")
                print("RAG System will use fallback answer generation")
        else:
            print("RAG System initialized with fallback mode")

    def retrieve_context(self, query: str, top_k: int = 5,
                         category_filter: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant documents for the query."""
        query_embedding = self.embedding_gen.generate_query_embedding(query)
        where_filter = {"category": category_filter} if category_filter else None

        results, _ = self.vector_db.search(
            query_embedding.tolist(),
            top_k=top_k,
            where=where_filter
        )
        return results

    def prepare_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Prepare the prompt with retrieved context."""
        context_text = "\n\n".join([
            f"[{i+1}] ({doc['metadata']['category']}, {doc['metadata']['difficulty']}): {doc['document']}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = f"""You are a helpful technical assistant. Use the following context to answer the question.
If the context doesn't contain enough information, say so and provide general guidance.

Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above. Reference specific points from the context when relevant."""

        return prompt

    def generate_answer(self, query: str, top_k: int = 5,
                        category_filter: Optional[str] = None,
                        temperature: float = 0.7) -> Dict:
        """Generate answer using RAG pipeline."""
        context_docs = self.retrieve_context(query, top_k, category_filter)
        prompt = self.prepare_prompt(query, context_docs)

        try:
            if self.client and self.llm_provider == "ollama":
                response = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful technical assistant. Be concise and informative."},
                        {"role": "user", "content": prompt}
                    ],
                    options={"temperature": temperature}
                )
                answer = response['message']['content']
            else:
                answer = self._generate_local_answer(query, context_docs)
        except Exception as e:
            print(f"LLM error: {e}")
            answer = self._generate_local_answer(query, context_docs)

        return {
            "query": query,
            "answer": answer,
            "sources": context_docs,
            "num_sources": len(context_docs)
        }

    def _generate_local_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate a simple answer without LLM (fallback)."""
        if not context_docs:
            return "I couldn't find relevant information in the knowledge base."

        answer_parts = [f"Based on the available information about '{query}':\n"]

        for i, doc in enumerate(context_docs[:3], 1):
            category = doc['metadata']['category']
            answer_parts.append(f"\n{i}. From {category}: {doc['document']}")

        return "\n".join(answer_parts)


def compare_metrics_chromadb(documents: List[Document], test_queries: List[str],
                              embedding_gen: EmbeddingGenerator) -> pd.DataFrame:
    """Compare different distance metrics using ChromaDB (fallback when Milvus unavailable)."""
    metrics = ["cosine", "l2", "ip"]
    results = []

    query_embeddings = [embedding_gen.generate_query_embedding(q).tolist() for q in test_queries]

    for metric in metrics:
        print(f"\n{'='*60}")
        print(f"Testing ChromaDB with {metric} metric...")
        print(f"{'='*60}")

        chroma = ChromaDBManager(f"./chroma_test_{metric}")
        chroma.create_collection(f"test_{metric}", distance_metric=metric)

        indexing_time = chroma.add_documents(documents)

        search_times = []
        for query_emb in query_embeddings:
            _, search_time = chroma.search(query_emb, top_k=10)
            search_times.append(search_time)

        _, batch_time = chroma.batch_search(query_embeddings, top_k=10)

        results.append({
            'Index': f"ChromaDB_{metric.upper()}",
            'Index Type': "HNSW",
            'Metric': metric.upper(),
            'Indexing Time (s)': indexing_time,
            'Avg Search Time (ms)': np.mean(search_times) * 1000,
            'Batch Search Time (ms)': batch_time * 1000,
            'Queries/sec': len(test_queries) / sum(search_times)
        })

        print(f"Indexing: {indexing_time:.2f}s, Avg Search: {np.mean(search_times)*1000:.2f}ms")

    return pd.DataFrame(results)


def compare_ann_algorithms(documents: List[Document], test_queries: List[str],
                           embedding_gen: EmbeddingGenerator) -> pd.DataFrame:
    """Compare different ANN algorithms. Uses Milvus if available, otherwise ChromaDB."""

    if not MILVUS_AVAILABLE:
        print("Milvus not available. Using ChromaDB for metric comparison instead.")
        return compare_metrics_chromadb(documents, test_queries, embedding_gen)

    index_configs = [
        {"name": "FLAT", "type": "FLAT", "params": {}, "search_params": {}},
        {"name": "IVF_FLAT_64", "type": "IVF_FLAT", "params": {"nlist": 64},
         "search_params": {"nprobe": 16}},
        {"name": "IVF_FLAT_128", "type": "IVF_FLAT", "params": {"nlist": 128},
         "search_params": {"nprobe": 32}},
        {"name": "HNSW_M8", "type": "HNSW", "params": {"M": 8, "efConstruction": 100},
         "search_params": {"ef": 64}},
        {"name": "HNSW_M16", "type": "HNSW", "params": {"M": 16, "efConstruction": 200},
         "search_params": {"ef": 128}},
    ]

    results = []
    ground_truth = {}

    query_embeddings = [embedding_gen.generate_query_embedding(q).tolist() for q in test_queries]

    for config in index_configs:
        print(f"\n{'='*60}")
        print(f"Testing {config['name']}...")
        print(f"{'='*60}")

        milvus = MilvusManager(f"./milvus_{config['name'].lower()}.db")

        creation_time = milvus.create_collection(
            name=f"test_{config['name'].lower()}",
            dim=384,
            index_type=config['type'],
            index_params=config['params']
        )

        indexing_time = milvus.add_documents(documents)

        search_times = []
        all_results = []

        for i, query_emb in enumerate(query_embeddings):
            search_result, search_time = milvus.search(
                query_emb,
                top_k=10,
                search_params=config['search_params'] if config['search_params'] else None
            )
            search_times.append(search_time)
            all_results.append([r['id'] for r in search_result])

        if config['name'] == 'FLAT':
            ground_truth = {i: set(results) for i, results in enumerate(all_results)}

        recall = 1.0
        if config['name'] != 'FLAT' and ground_truth:
            recalls = []
            for i, results_ids in enumerate(all_results):
                if i in ground_truth:
                    intersection = len(set(results_ids) & ground_truth[i])
                    recalls.append(intersection / len(ground_truth[i]))
            recall = np.mean(recalls) if recalls else 1.0

        results.append({
            'Index': config['name'],
            'Index Type': config['type'],
            'Indexing Time (s)': indexing_time,
            'Avg Search Time (ms)': np.mean(search_times) * 1000,
            'Recall@10': recall,
            'Queries/sec': len(test_queries) / sum(search_times)
        })

        print(f"Indexing: {indexing_time:.2f}s, Avg Search: {np.mean(search_times)*1000:.2f}ms, Recall: {recall:.4f}")

    return pd.DataFrame(results)


def run_performance_benchmark(chroma_manager: ChromaDBManager,
                               embedding_gen: EmbeddingGenerator,
                               num_queries: int = 100) -> Dict:
    """Run comprehensive performance benchmark."""
    import random

    query_templates = [
        "How to implement {topic} in {language}?",
        "Best practices for {topic}",
        "What is {topic} and how does it work?",
        "Explain {topic} with examples",
        "Compare different approaches to {topic}"
    ]

    topics = ["caching", "indexing", "authentication", "optimization", "deployment",
              "testing", "monitoring", "scaling", "security", "performance"]
    languages = ["Python", "JavaScript", "Go", "Java", "Rust"]

    queries = []
    for _ in range(num_queries):
        template = random.choice(query_templates)
        query = template.format(
            topic=random.choice(topics),
            language=random.choice(languages)
        )
        queries.append(query)

    # Benchmark single queries
    single_query_times = []
    for query in tqdm(queries, desc="Single query benchmark"):
        query_emb = embedding_gen.generate_query_embedding(query)
        _, search_time = chroma_manager.search(query_emb.tolist(), top_k=5)
        single_query_times.append(search_time)

    # Benchmark batch queries
    batch_sizes = [10, 25, 50, 100]
    batch_results = {}

    for batch_size in batch_sizes:
        batch_queries = queries[:batch_size]
        batch_embeddings = [embedding_gen.generate_query_embedding(q).tolist() for q in batch_queries]

        start = time.time()
        _, batch_time = chroma_manager.batch_search(batch_embeddings, top_k=5)
        total_time = time.time() - start

        batch_results[batch_size] = {
            'total_time': total_time,
            'time_per_query': total_time / batch_size
        }

    # Top-k benchmark
    top_k_values = [1, 5, 10, 20, 50]
    top_k_results = {}

    test_query = queries[0]
    test_emb = embedding_gen.generate_query_embedding(test_query)

    for k in top_k_values:
        times = []
        for _ in range(10):
            _, search_time = chroma_manager.search(test_emb.tolist(), top_k=k)
            times.append(search_time)
        top_k_results[k] = np.mean(times)

    return {
        'single_query': {
            'mean_ms': np.mean(single_query_times) * 1000,
            'std_ms': np.std(single_query_times) * 1000,
            'p50_ms': np.percentile(single_query_times, 50) * 1000,
            'p95_ms': np.percentile(single_query_times, 95) * 1000,
            'p99_ms': np.percentile(single_query_times, 99) * 1000,
        },
        'batch_query': batch_results,
        'top_k_scaling': top_k_results
    }


def main():
    """Main function to run the RAG system demonstration."""
    print("=" * 80)
    print("RAG SYSTEM WITH VECTOR DATABASE COMPARISON")
    print("=" * 80)

    # 1. Generate dataset
    print("\n[1/6] Generating dataset...")
    generator = DatasetGenerator()
    documents = generator.generate_dataset(1200)
    print(f"Generated {len(documents)} documents across {len(generator.categories)} categories")

    # Show dataset distribution
    category_counts = defaultdict(int)
    for doc in documents:
        category_counts[doc.metadata['category']] += 1
    print("\nDataset distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # 2. Generate embeddings
    print("\n[2/6] Generating embeddings...")
    embedding_gen = EmbeddingGenerator()
    texts = [doc.text for doc in documents]
    embeddings = embedding_gen.generate_embeddings(texts)

    for doc, emb in zip(documents, embeddings):
        doc.embedding = emb.tolist()
    print(f"Generated embeddings shape: {embeddings.shape}")

    # 3. Setup ChromaDB
    print("\n[3/6] Setting up ChromaDB...")
    chroma_manager = ChromaDBManager()
    chroma_manager.create_collection("tech_documents", distance_metric="cosine")
    indexing_time = chroma_manager.add_documents(documents)

    # 4. Test semantic search
    print("\n[4/6] Testing semantic search...")
    test_queries = [
        "How to optimize database performance?",
        "What are the best practices for Python web development?",
        "Explain machine learning model training",
        "How to secure web applications?",
        "What is containerization and Kubernetes?"
    ]

    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH RESULTS")
    print("=" * 60)

    for query in test_queries:
        query_embedding = embedding_gen.generate_query_embedding(query)
        results, search_time = chroma_manager.search(query_embedding.tolist(), top_k=3)

        print(f"\nQuery: '{query}'")
        print(f"Search time: {search_time*1000:.2f}ms")
        print("-" * 40)

        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['metadata']['category']}] (dist: {result['distance']:.4f})")
            print(f"     {result['document'][:80]}...")

    # 5. Compare ANN algorithms
    print("\n[5/6] Comparing ANN algorithms...")
    benchmark_queries = [
        "How to optimize Python code performance?",
        "Best practices for database indexing",
        "Machine learning model deployment strategies",
        "Web security vulnerabilities and prevention",
        "Container orchestration with Kubernetes"
    ]

    comparison_df = compare_ann_algorithms(documents, benchmark_queries, embedding_gen)
    print("\n" + "=" * 80)
    print("ANN ALGORITHM COMPARISON RESULTS")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # 6. Test RAG system
    print("\n[6/6] Testing RAG system...")
    # Using Ollama with local model (default: llama3.2)
    # Change model_name to match your installed Ollama model
    rag_system = RAGSystem(chroma_manager, embedding_gen, llm_provider="ollama", model_name="llama3.2")

    print("\n" + "=" * 80)
    print("RAG SYSTEM DEMONSTRATION")
    print("=" * 80)

    rag_questions = [
        "What are the best practices for optimizing database queries?",
        "How can I implement secure authentication in web applications?",
    ]

    for question in rag_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("=" * 60)

        result = rag_system.generate_answer(question, top_k=3)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\n--- Sources ({result['num_sources']}) ---")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. [{source['metadata']['category']}] {source['document'][:60]}...")

    # Run performance benchmark
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)

    benchmark_results = run_performance_benchmark(chroma_manager, embedding_gen, num_queries=50)

    print("\n--- Single Query Performance ---")
    sq = benchmark_results['single_query']
    print(f"Mean: {sq['mean_ms']:.2f}ms")
    print(f"Std: {sq['std_ms']:.2f}ms")
    print(f"P50: {sq['p50_ms']:.2f}ms")
    print(f"P95: {sq['p95_ms']:.2f}ms")
    print(f"P99: {sq['p99_ms']:.2f}ms")

    print("\n--- Batch Query Performance ---")
    for batch_size, results in benchmark_results['batch_query'].items():
        print(f"Batch size {batch_size}: {results['total_time']*1000:.2f}ms total, "
              f"{results['time_per_query']*1000:.2f}ms per query")

    # Save results
    results_summary = {
        "dataset": {
            "num_documents": len(documents),
            "num_categories": len(generator.categories),
            "embedding_dim": embedding_gen.embedding_dim
        },
        "ann_comparison": comparison_df.to_dict('records'),
        "performance_benchmark": {
            "single_query": benchmark_results['single_query'],
            "batch_query": {str(k): v for k, v in benchmark_results['batch_query'].items()},
            "top_k_scaling": {str(k): v for k, v in benchmark_results['top_k_scaling'].items()}
        }
    }

    with open('rag_results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
Dataset:
- Total documents: {len(documents)}
- Categories: {len(generator.categories)}
- Embedding dimension: {embedding_gen.embedding_dim}

Performance (ChromaDB):
- Indexing: {indexing_time:.2f}s for {len(documents)} documents
- Average search latency: {benchmark_results['single_query']['mean_ms']:.2f}ms
- P99 latency: {benchmark_results['single_query']['p99_ms']:.2f}ms

Results saved to: rag_results_summary.json
    """)


if __name__ == "__main__":
    main()
