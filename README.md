# NLP Knowledge Discovery Platform

## Overview

The NLP Knowledge Discovery Platform is an end-to-end research prototype for scientific document exploration using Natural Language Processing (NLP), Knowledge Graphs, Information Retrieval, and Semantic Search techniques.

The system processes scientific publications from the arXiv dataset, extracts structured knowledge from unstructured text, constructs a knowledge graph representing relationships between documents, authors, topics, and entities, and enables both lexical and semantic retrieval of scientific information.

The project was developed as part of a Master's degree in Artificial Intelligence Engineering and implements the concepts proposed in:

**Building a Knowledge Graph for Scientific Document Exploration using NLP and Semantic Retrieval**

---

## Project Objectives

The primary objectives of this project are:

- Extract entities, concepts, keywords, and relations from scientific publications
- Construct a knowledge graph from extracted information
- Enable graph-based exploration of scientific literature
- Implement multiple retrieval approaches for document discovery
- Compare lexical, semantic, and graph-enhanced retrieval methods
- Evaluate retrieval quality using standard Information Retrieval metrics
- Demonstrate how structured knowledge improves scientific document exploration

---

## System Architecture

```text
Raw Scientific Documents
            │
            ▼
    Text Preprocessing
            │
            ▼
 Information Extraction
 ├── Named Entities
 ├── Keywords
 └── Relations
            │
            ▼
 Knowledge Graph Construction
            │
            ▼
      Knowledge Graph
            │
     ┌──────┼──────┐
     ▼      ▼      ▼
   BM25  Semantic  KG-Retrieval
            │
            ▼
       Evaluation
```

---

## Implemented Functionality

### Data Processing

- Loading scientific documents from CSV, JSON, and JSONL formats
- Support for arXiv metadata datasets
- Dataset filtering and sampling
- Text cleaning and normalization
- Tokenization and lemmatization
- Stopword removal

### Information Extraction

- Named Entity Recognition (NER)
- TF-IDF keyword extraction
- TextRank keyword extraction
- Co-occurrence relation extraction
- Relation aggregation across documents

### Knowledge Graph Construction

- Paper nodes
- Author nodes
- Entity nodes
- Topic nodes
- Relation edges
- Graph pruning and filtering
- GraphML export
- JSON export
- Interactive graph visualization

### Information Retrieval

#### BM25 Baseline Retrieval

- Traditional lexical retrieval
- Ranking based on term frequency and inverse document frequency

#### Semantic Retrieval

- Sentence Transformer embeddings
- Embedding caching
- Vector similarity search

#### Knowledge Graph Enhanced Retrieval

- Graph-based reranking
- Entity-aware retrieval
- Semantic graph expansion

### Topic Modeling

- Latent Dirichlet Allocation (LDA)
- BERTopic integration

### Advanced Extensions

- Retrieval-Augmented Generation (RAG)
- Graph Question Answering
- Embedding Bias Analysis
- Debiasing Utilities
- Graph Neural Network Utilities
- Semantic Network Construction

### Visualization

- Streamlit dashboard
- Interactive graph exploration
- Topic visualization
- Search interface
- Network analysis

---

## Repository Structure

```text
config.yaml
run_all.py
create_queries.py

src/
├── preprocessing/
├── extraction/
├── knowledge_graph/
├── retrieval/
├── experiments/
├── evaluation/
├── topic_modeling/
├── embeddings/
├── visualization/
├── llm_kg/
├── gnn/
└── bias/

tests/

data/
├── raw/
├── processed/
└── graphs/

reports/
├── figures/
└── tables/
```

---

## Installation

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Install spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

## Running the Project

### Complete Workflow

```bash
python run_all.py --full --skip-embeddings
```

### Full Workflow Including Embeddings

```bash
python run_all.py --full
```

### Run Only the Pipeline

```bash
python -m src.pipeline.run_pipeline --skip-embeddings
```

### Explicit arXiv Dataset

```bash
python run_all.py --input data/raw/arxiv-metadata-oai-snapshot.json --full --skip-embeddings
```

### Launch Dashboard

```bash
streamlit run src/visualization/dashboard.py
```

---

## Generated Outputs

### Processed Data

```text
data/processed/
├── processed_documents.csv
├── entities.csv
├── keywords.csv
├── relations.csv
└── graph_summary.json
```

### Knowledge Graph

```text
data/graphs/
├── knowledge_graph.graphml
├── knowledge_graph.json
└── semantic_network.graphml
```

### Retrieval Experiments

```text
reports/tables/
├── bm25_results.csv
├── bm25_metrics.csv
├── semantic_results.csv
├── semantic_metrics.csv
├── kg_results.csv
└── kg_metrics.csv
```

### Visualizations

```text
reports/figures/
└── knowledge_graph.html
```

---

## Evaluation

The project evaluates retrieval quality using:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

Experiments are performed for:

1. BM25 Baseline Retrieval
2. Semantic Retrieval
3. Knowledge Graph Enhanced Retrieval

---

## Testing

The repository includes automated tests covering:

- Data preprocessing
- Information extraction
- Retrieval systems
- Knowledge graph functionality
- Evaluation metrics
- Semantic retrieval
- Graph-enhanced retrieval
- Topic modeling helpers
- Additional extension modules

Run the tests:

```bash
pytest -v
```

### Current Status

```text
50 passed
```

---

## Technologies Used

- Python
- spaCy
- pandas
- NumPy
- scikit-learn
- NetworkX
- Sentence Transformers
- Streamlit
- PyTorch
- Gensim

---

## Conclusion

The NLP Knowledge Discovery Platform successfully demonstrates how Natural Language Processing, Knowledge Graphs, and Semantic Retrieval can be combined to support scientific document exploration. The resulting system enables extraction of structured knowledge from scientific publications, graph-based representation of research concepts, and enhanced retrieval capabilities beyond traditional keyword search.

The platform provides a complete and reproducible workflow from raw scientific documents to searchable knowledge graphs and retrieval evaluation experiments.
