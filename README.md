# NLP Knowledge Discovery Platform

> AI-Powered Scientific Knowledge Discovery System using Natural Language Processing, Knowledge Graphs, Graph Neural Networks, Embeddings, and Large Language Models.

---

# Project Description

The **NLP Knowledge Discovery Platform** is a research-oriented Natural Language Processing (NLP) project that combines multiple advanced AI topics into one unified system.

The goal of this project is to transform large collections of unstructured scientific or technical text into structured and searchable knowledge.

The platform combines:

- Information Extraction & Retrieval
- Knowledge Graph Construction
- Topic Modeling
- Embeddings & Semantic Networks
- Graph Neural Networks (GNNs)
- Large Language Models (LLMs)
- Bias Detection & Debiasing

The system will process scientific papers, articles, technical reports, or research datasets and automatically:

- extract entities and relationships,
- construct knowledge graphs,
- generate semantic embeddings,
- retrieve relevant information,
- discover hidden research topics,
- apply graph-based reasoning,
- and analyze bias in NLP systems.

The final outcome will be an intelligent knowledge discovery platform with visual analytics and semantic search capabilities.

---

# Main Objectives

The project aims to:

- Build a complete NLP processing pipeline
- Extract entities and semantic relationships from documents
- Construct and analyze knowledge graphs
- Implement semantic information retrieval
- Discover hidden topics in document collections
- Generate embeddings for semantic similarity analysis
- Apply Graph Neural Networks to graph data
- Combine LLMs with Knowledge Graphs using RAG pipelines
- Evaluate bias in embeddings and LLM outputs
- Implement debiasing strategies
- Create interactive visualizations and dashboards

---

# Topics Covered

## 1. Information Extraction & Retrieval

This module converts raw text into structured information.

### Features
- Named Entity Recognition (NER)
- Relation Extraction
- Keyword Extraction
- Semantic Search
- Document Ranking
- Dense Retrieval

### Example Use Cases
- Find related scientific papers
- Extract important concepts from research articles
- Build searchable document collections

### Technologies
- spaCy
- BM25
- Sentence Transformers
- Hugging Face Transformers

---

## 2. Knowledge Graphs

This module transforms extracted information into graph structures.

### Example Nodes
- Authors
- Papers
- Institutions
- Algorithms
- Datasets
- Concepts

### Example Relationships
- `AUTHORED_BY`
- `CITES`
- `USES`
- `RELATED_TO`
- `BELONGS_TO_TOPIC`

### Features
- Graph construction
- Graph querying
- Graph visualization
- Relationship analysis

### Technologies
- NetworkX
- Neo4j
- RDFLib

---

## 3. Large Language Models + Knowledge Graphs

This module investigates how LLMs can interact with structured knowledge.

### Features
- Retrieval-Augmented Generation (RAG)
- Graph-grounded Question Answering
- Context-aware prompting
- Hallucination reduction

### Example Use Cases
- Ask questions over a knowledge graph
- Generate grounded scientific summaries
- Retrieve evidence-based answers

### Technologies
- Hugging Face Transformers
- Sentence Transformers
- Vector Databases
- RAG Pipelines

---

## 4. Topic Modeling

This module discovers hidden themes inside document collections.

### Methods
- Latent Dirichlet Allocation (LDA)
- BERTopic
- Non-negative Matrix Factorization (NMF)

### Outputs
- Topic Clusters
- Topic Trends
- Topic Evolution
- Topic Visualization

---

## 5. Embeddings & Networks

This module creates semantic vector representations of text and graphs.

### Embedding Types
- Word Embeddings
- Sentence Embeddings
- Document Embeddings
- Graph Embeddings

### Network Types
- Citation Networks
- Semantic Similarity Networks
- Co-author Networks
- Concept Networks

### Applications
- Semantic search
- Similarity analysis
- Recommendation systems
- Clustering

---

## 6. Graph Neural Networks (GNNs)

This module applies deep learning to graph structures.

### Methods
- Graph Convolutional Networks (GCN)
- GraphSAGE
- Graph Attention Networks (GAT)

### Tasks
- Node Classification
- Link Prediction
- Graph Embedding Learning

### Example Applications
- Predict missing relationships
- Detect important nodes
- Infer hidden graph structure

---

## 7. Bias and Debiasing in NLP

This module investigates fairness and responsible AI.

### Bias Types
- Gender Bias
- Cultural Bias
- Domain Bias
- Stereotypical Associations

### Debiasing Techniques
- Embedding Projection
- Balanced Sampling
- Prompt Engineering
- Adversarial Debiasing

### Goals
- Analyze fairness of embeddings
- Evaluate LLM behavior
- Reduce harmful associations

---

# Proposed Architecture

```text
Documents / PDFs / CSV Files
            |
            v
Text Extraction & Cleaning
            |
            v
Information Extraction
            |
            +----------------------+
            |                      |
            v                      v
   Knowledge Graph          Embedding Store
            |                      |
            v                      v
Graph Analysis / GNNs       Semantic Retrieval
            |                      |
            +----------+-----------+
                       |
                       v
              LLM-based QA System
                       |
                       v
        Dashboard / Visualizations / Report
```

---

# Repository Structure

```text
nlp-knowledge-discovery-platform/
│
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── graphs/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_information_extraction.ipynb
│   ├── 03_topic_modeling.ipynb
│   ├── 04_embeddings_networks.ipynb
│   ├── 05_bias_analysis.ipynb
│   └── 06_gnn_experiments.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── extraction/
│   ├── retrieval/
│   ├── knowledge_graph/
│   ├── topic_modeling/
│   ├── embeddings/
│   ├── gnn/
│   ├── llm_kg/
│   ├── bias/
│   └── visualization/
│
├── reports/
│   └── figures/
│
└── tests/
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/AngeloOttendorfer02/nlp-knowledge-discovery-platform.git
cd nlp-knowledge-discovery-platform
```

---

## Create Conda Environment

```bash
conda create -n nlp-kg python=3.10
conda activate nlp-kg
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

# Installation, Reproducibility and Smoke Tests

This section explains how a new developer can clone the repository, install the environment, prepare the required folders, and run basic checks to verify that the current pipeline works.

The commands below are intended to be run from the repository root.

---

## Recommended Python Version

Use **Python 3.10 or Python 3.11** for best compatibility with the NLP ecosystem.

Avoid very new Python versions such as Python 3.13 or Python 3.14 unless all dependencies have been verified manually, because some NLP and ML packages may not provide stable wheels for the newest Python versions yet.

Check your Python version:

```bash
python3 --version
```

---

## Option A — Setup with Python venv

```bash
git clone https://github.com/AngeloOttendorfer02/nlp-knowledge-discovery-platform.git
cd nlp-knowledge-discovery-platform

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Option B — Setup with Conda

```bash
git clone https://github.com/AngeloOttendorfer02/nlp-knowledge-discovery-platform.git
cd nlp-knowledge-discovery-platform

conda create -n nlp-kg python=3.10
conda activate nlp-kg

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Required Data and Output Folders

The repository expects the following folders to exist:

```bash
mkdir -p data/raw data/processed data/graphs reports/figures

touch data/raw/.gitkeep \
      data/processed/.gitkeep \
      data/graphs/.gitkeep \
      reports/.gitkeep \
      reports/figures/.gitkeep
```

Purpose of these folders:

- `data/raw/` — original input data, for example arXiv metadata or small test files.
- `data/processed/` — cleaned or transformed intermediate data.
- `data/graphs/` — exported knowledge graphs, for example JSON or GraphML files.
- `reports/figures/` — plots and visualizations generated during experiments.

Large datasets should not be committed directly unless explicitly agreed by the team. If the dataset is too large, document the download source and expected file path.

---

## Source Code Syntax Check

Run this first to verify that all Python files compile:

```bash
python -m compileall src
```

Expected result: no Python syntax errors.

---

## Import Smoke Test

This verifies that the implemented modules can be imported successfully.

```bash
python - <<'PY'
modules = [
    "src.preprocessing.document_loader",
    "src.preprocessing.text_cleaning",
    "src.retrieval.bm25_retriever",
    "src.retrieval.vector_store",
    "src.retrieval.embedding_retriever",
    "src.extraction.entity_extraction",
    "src.extraction.keyword_extraction",
    "src.extraction.relation_extraction",
    "src.knowledge_graph.graph_builder",
    "src.knowledge_graph.graph_queries",
    "src.knowledge_graph.graph_visualization",
    "src.embeddings.sentence_embeddings",
    "src.embeddings.network_embeddings",
    "src.topic_modeling.lda_model",
    "src.topic_modeling.bertopic_model",
]

for module in modules:
    __import__(module)
    print("OK:", module)

print("Import smoke test passed.")
PY
```

Expected result: every module prints `OK:` and the script ends with `Import smoke test passed.`

---

## Minimal End-to-End Smoke Test

This test creates a tiny artificial arXiv-like JSONL dataset and verifies the current pipeline components:

- JSONL document loading
- document object conversion
- text cleaning
- text chunking
- BM25 retrieval
- spaCy entity extraction
- co-occurrence relation extraction
- knowledge graph construction
- graph export to JSON and GraphML

```bash
python - <<'PY'
from pathlib import Path
import json

from src.preprocessing.document_loader import load_arxiv_jsonl, dataframe_to_documents
from src.preprocessing.text_cleaning import clean_text, chunk_text
from src.retrieval.bm25_retriever import BM25Retriever
from src.extraction.entity_extraction import EntityExtractor
from src.extraction.relation_extraction import CooccurrenceRelationExtractor
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder

Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/graphs").mkdir(parents=True, exist_ok=True)

sample_path = Path("data/raw/tiny_arxiv.jsonl")

records = [
    {
        "id": "001",
        "title": "Graph Neural Networks for Scientific Document Retrieval",
        "abstract": "We study graph neural networks and semantic retrieval for scientific papers.",
        "authors": "Alice Smith, Bob Miller",
        "categories": "cs.AI cs.IR",
        "update_date": "2026-01-01",
    },
    {
        "id": "002",
        "title": "Transformer Models for Information Extraction",
        "abstract": "BERT and transformer models are used for named entity recognition and relation extraction.",
        "authors": "Carol Jones",
        "categories": "cs.CL",
        "update_date": "2026-01-02",
    },
    {
        "id": "003",
        "title": "Medical Image Segmentation",
        "abstract": "This paper studies convolutional neural networks for image segmentation.",
        "authors": "David Brown",
        "categories": "cs.CV",
        "update_date": "2026-01-03",
    },
]

with sample_path.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")

df = load_arxiv_jsonl(
    str(sample_path),
    categories=["cs.AI", "cs.CL", "cs.IR"],
    sample_size=None,
)

print("Loaded documents:")
print(df[["doc_id", "title", "categories"]])

documents = dataframe_to_documents(df)
texts = [doc.text for doc in documents]
doc_ids = [doc.doc_id for doc in documents]

print("\nClean text example:")
print(clean_text(texts[0]))

print("\nChunk example:")
print(chunk_text(["a", "b", "c", "d", "e"], chunk_size=2))

retriever = BM25Retriever(tokenizer=lambda text: text.lower().split())
retriever.index(doc_ids, texts)

results = retriever.search("transformer information extraction", top_k=2)

print("\nBM25 results:")
for result in results:
    print(result.rank, result.doc_id, round(result.score, 4), result.text[:80])

entity_extractor = EntityExtractor(entity_types=["ORG", "PERSON", "GPE", "PRODUCT", "WORK_OF_ART"])
entities_per_doc = entity_extractor.extract_batch(texts)

print("\nEntities:")
for doc_id, entities in zip(doc_ids, entities_per_doc):
    print(doc_id, [(e.text, e.label) for e in entities])

relation_extractor = CooccurrenceRelationExtractor()
kg = KnowledgeGraphBuilder()

for doc, entities in zip(documents, entities_per_doc):
    kg.add_paper(
        doc_id=doc.doc_id,
        title=doc.title,
        authors=doc.authors,
        categories=doc.categories,
    )
    kg.add_entities(doc.doc_id, entities)
    kg.add_relations(relation_extractor.extract(doc.text, entities))

print("\nKG stats:")
print(kg.stats())

kg.save_json("data/graphs/tiny_graph.json")
kg.save_graphml("data/graphs/tiny_graph.graphml")

print("\nSmoke test passed.")
PY
```

Expected result:

- documents are loaded from the generated JSONL file,
- BM25 returns ranked search results,
- entities are extracted with spaCy,
- a small knowledge graph is built,
- `data/graphs/tiny_graph.json` is created,
- `data/graphs/tiny_graph.graphml` is created,
- the script ends with `Smoke test passed.`

---

## Optional Semantic Retrieval Smoke Test

This test verifies the embedding-based retrieval path.

It may download a Sentence Transformers model on the first run and can take longer than the basic smoke test.

```bash
python - <<'PY'
from src.retrieval.embedding_retriever import EmbeddingRetriever

doc_ids = ["001", "002", "003"]
texts = [
    "Graph neural networks for scientific document retrieval.",
    "Transformer models for named entity recognition and relation extraction.",
    "Medical image segmentation using convolutional networks.",
]

retriever = EmbeddingRetriever(model_name="all-MiniLM-L6-v2")
retriever.index(doc_ids, texts)

results = retriever.search("semantic search for scientific papers", top_k=2)

for result in results:
    print(result.rank, result.doc_id, round(result.score, 4), result.text)

print("Embedding retrieval smoke test passed.")
PY
```

If this fails because FAISS is missing, install it with:

```bash
pip install faiss-cpu
```

If this fails because Sentence Transformers is missing, install it with:

```bash
pip install sentence-transformers
```

Both packages should ideally be listed in `requirements.txt` if this module is part of the active project scope.

---

## Full Quick-Start Checklist

For a fresh clone, run:

```bash
git clone https://github.com/AngeloOttendorfer02/nlp-knowledge-discovery-platform.git
cd nlp-knowledge-discovery-platform

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

mkdir -p data/raw data/processed data/graphs reports/figures

touch data/raw/.gitkeep \
      data/processed/.gitkeep \
      data/graphs/.gitkeep \
      reports/.gitkeep \
      reports/figures/.gitkeep

python -m compileall src
```

Then run the import smoke test and the minimal end-to-end smoke test from the sections above.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pandas'`

Install project dependencies:

```bash
pip install -r requirements.txt
```

### `spaCy model 'en_core_web_sm' is not installed`

Download the model:

```bash
python -m spacy download en_core_web_sm
```

### `ModuleNotFoundError: No module named 'rank_bm25'`

Install the BM25 dependency:

```bash
pip install rank-bm25
```

### `ModuleNotFoundError: No module named 'faiss'`

Install FAISS CPU:

```bash
pip install faiss-cpu
```

### Sentence Transformers model downloads are slow

The first run of embedding retrieval may download `all-MiniLM-L6-v2`. This is normal.

### Problems with Python 3.13 or 3.14

Use Python 3.10 or 3.11 instead:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Current Reproducibility Status

At this stage, the repository contains reusable module implementations for:

- preprocessing,
- document loading,
- BM25 retrieval,
- embedding retrieval,
- entity extraction,
- relation extraction,
- knowledge graph construction,
- graph querying,
- graph visualization,
- topic modeling,
- semantic embeddings and network embeddings.

The smoke tests above validate that the current codebase can be installed and executed on a fresh environment. They are not yet the final scientific evaluation. The final project will still need:

- real dataset download instructions,
- complete notebooks,
- retrieval evaluation queries,
- comparison of BM25 vs semantic retrieval vs KG-enhanced retrieval,
- metrics such as Precision@k, Recall@k and MRR,
- error analysis and interpretation.

---

# Planned Workflow

## Step 1 — Data Collection
- Collect scientific papers or documents
- Load PDFs, CSV files, or text documents
- Organize datasets

## Step 2 — Text Preprocessing
- Cleaning
- Tokenization
- Lemmatization
- Stopword removal
- Chunking

## Step 3 — Information Extraction
- Extract entities
- Extract relationships
- Extract keywords

## Step 4 — Semantic Retrieval
- Create embeddings
- Build semantic search
- Rank relevant documents

## Step 5 — Knowledge Graph Construction
- Create nodes and edges
- Store graph structure
- Visualize graph

## Step 6 — Topic Modeling
- Discover hidden themes
- Analyze topic evolution
- Visualize topics

## Step 7 — Embeddings & Networks
- Generate embeddings
- Create similarity networks
- Analyze semantic relationships

## Step 8 — Graph Neural Networks
- Prepare graph dataset
- Train GNN model
- Evaluate graph reasoning

## Step 9 — LLM + KG Integration
- Build Retrieval-Augmented Generation pipeline
- Graph-grounded QA
- Context-aware prompting

## Step 10 — Bias Analysis & Debiasing
- Evaluate embedding bias
- Analyze LLM bias
- Apply debiasing methods

---

# Detailed Commit Roadmap

---

# Commit 1 — Initialize Repository

### Goals
- Create repository
- Add project description
- Add README
- Add `.gitignore`

### Tasks
- Initialize Git repository
- Create base folders
- Create initial README

---

# Commit 2 — Add Project Structure

### Goals
Create a clean modular architecture.

### Tasks
- Create `src/`
- Create `notebooks/`
- Create `data/`
- Create `reports/`
- Create `tests/`

---

# Commit 3 — Add Dependencies & Configuration

### Goals
Set up environment and dependencies.

### Tasks
- Create `requirements.txt`
- Create `config.yaml`
- Define model configurations
- Define project paths

---

# Commit 4 — Implement Data Loading & Preprocessing

### Goals
Prepare text data for NLP processing.

### Tasks
- Load PDFs and CSV files
- Clean text
- Normalize text
- Tokenize documents
- Remove stopwords

### Files
- `document_loader.py`
- `text_cleaning.py`

---

# Commit 5 — Implement Information Extraction

### Goals
Extract structured information from documents.

### Tasks
- Named Entity Recognition
- Keyword Extraction
- Relation Extraction

### Files
- `entity_extraction.py`
- `keyword_extraction.py`
- `relation_extraction.py`

---

# Commit 6 — Implement Semantic Retrieval

### Goals
Build document retrieval system.

### Tasks
- Implement BM25 retrieval
- Implement embedding-based retrieval
- Add semantic search
- Add ranking system

### Files
- `bm25_retriever.py`
- `embedding_retriever.py`

---

# Commit 7 — Build Knowledge Graph

### Goals
Convert extracted information into graph structure.

### Tasks
- Create nodes
- Create edges
- Store graph
- Implement graph queries

### Files
- `graph_builder.py`
- `graph_queries.py`

---

# Commit 8 — Add Graph Visualization

### Goals
Visualize knowledge graph structure.

### Tasks
- Create interactive graph visualization
- Export graph images
- Add network analysis

### Files
- `graph_visualization.py`

---

# Commit 9 — Implement Topic Modeling

### Goals
Discover hidden themes in documents.

### Tasks
- Implement LDA
- Implement BERTopic
- Create topic visualization

### Files
- `lda_model.py`
- `bertopic_model.py`

---

# Commit 10 — Add Embeddings & Semantic Networks

### Goals
Create semantic vector representations.

### Tasks
- Generate sentence embeddings
- Create similarity matrices
- Build semantic networks

### Files
- `sentence_embeddings.py`
- `network_embeddings.py`

---

# Commit 11 — Prepare Graph Dataset for GNNs

### Goals
Convert graph into trainable graph dataset.

### Tasks
- Create node features
- Create graph splits
- Prepare graph tensors

### Files
- `dataset_builder.py`

---

# Commit 12 — Implement Graph Neural Networks

### Goals
Train GNN models on graph data.

### Tasks
- Implement GCN
- Train GNN model
- Evaluate graph performance

### Files
- `gcn_model.py`
- `train_gnn.py`

---

# Commit 13 — Integrate LLM + Knowledge Graph

### Goals
Connect LLMs with graph retrieval.

### Tasks
- Build Retrieval-Augmented Generation pipeline
- Create graph-grounded QA
- Add contextual retrieval

### Files
- `graph_retrieval_qa.py`
- `rag_pipeline.py`

---

# Commit 14 — Add Bias Evaluation

### Goals
Analyze fairness and bias.

### Tasks
- Evaluate embedding bias
- Analyze LLM bias
- Create fairness metrics

### Files
- `embedding_bias.py`
- `llm_bias_eval.py`

---

# Commit 15 — Implement Debiasing

### Goals
Reduce harmful bias.

### Tasks
- Add embedding debiasing
- Add prompt engineering
- Compare before vs after

### Files
- `debiasing.py`

---

# Commit 16 — Add Dashboard & Visualizations

### Goals
Create user interface and analytics dashboard.

### Tasks
- Build Streamlit dashboard
- Add graph visualizations
- Add topic visualization
- Add semantic search UI

### Files
- `dashboard.py`
- `plots.py`

---

# Commit 17 — Add Tests & Refactoring

### Goals
Improve reliability and code quality.

### Tasks
- Add unit tests
- Refactor duplicated code
- Improve documentation
- Improve comments

---

# Commit 18 — Final Report & Documentation

### Goals
Prepare final submission.

### Tasks
- Add final report
- Add figures
- Finalize README
- Add presentation materials

---

# Commit 19 — Final Submission

### Goals
Prepare clean final repository.

### Tasks
- Final cleanup
- Verify notebooks
- Verify documentation
- Verify repository structure

---

# Expected Deliverables

By the end of the project:

- Complete GitHub repository
- NLP preprocessing pipeline
- Information extraction system
- Knowledge graph construction
- Semantic retrieval system
- Topic modeling analysis
- Embedding and network analysis
- Graph Neural Network experiments
- LLM + KG integration
- Bias evaluation framework
- Debiasing experiments
- Interactive dashboard
- Final report and presentation

---

# Future Improvements

Possible extensions include:

- Multilingual NLP support
- Real-time document ingestion
- Advanced relation extraction
- Larger graph databases
- Conversational graph assistants
- Scientific recommendation systems
- Explainable graph reasoning

---

# Conclusion

This project combines several advanced NLP and AI research areas into one unified knowledge discovery platform.

It demonstrates how:
- unstructured text can become structured knowledge,
- semantic retrieval improves information access,
- graph learning enables reasoning over relationships,
- and responsible AI methods help evaluate fairness and bias in modern NLP systems.
