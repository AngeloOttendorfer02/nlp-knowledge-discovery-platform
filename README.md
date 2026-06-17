# NLP Knowledge Discovery Platform

A professional research platform for scientific document discovery using NLP, semantic retrieval, knowledge graphs, and interactive visualization.

## Overview

This repository builds a reproducible pipeline for processing arXiv-style scientific metadata. It combines:

- text preprocessing and entity extraction
- keyword extraction and relation discovery
- knowledge graph construction and export
- BM25 and semantic retrieval comparisons
- topic modeling and embedding analysis
- interactive dashboard exploration

The project is designed for research, experimentation, and prototype evaluation of document-level knowledge discovery.

## What is implemented

- Data loading from CSV, JSON, and JSONL arXiv-style metadata
- Text cleaning and preprocessing with spaCy
- Named entity extraction and keyword extraction
- Co-occurrence relation extraction
- Knowledge graph construction and export to GraphML / JSON
- BM25 retrieval baseline implementation
- Semantic retrieval using Sentence Transformers
- Knowledge graph-enhanced retrieval reranking
- Retrieval evaluation metrics and experiment runners
- LDA topic modeling and BERTopic wrapper support
- Document embedding generation and semantic similarity network construction
- Interactive Streamlit dashboard with search, topic views, clustering, and graph exploration
- Convenience runner via `python -m run_all`
- Automated tests for pipeline and retrieval components

## What still needs to be implemented

- richer node and edge typing in the knowledge graph
- more robust relation extraction beyond co-occurrence
- full retrieval ground-truth dataset and evaluation annotations
- expanded topic modeling visualizations and interpretability tools
- production-ready dataset sampling and persistence workflows
- optional GNN experiments for node/link prediction
- optional Graph + LLM integration for KG-grounded QA
- optional bias evaluation and debiasing analysis

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/AngeloOttendorfer02/nlp-knowledge-discovery-platform.git
cd nlp-knowledge-discovery-platform
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 4. Run the pipeline

```bash
python -m src.pipeline.run_pipeline
```

### 5. Start the dashboard

```bash
streamlit run src/visualization/dashboard.py
```

## Convenience commands

- Run only the pipeline:

```bash
python -m src.pipeline.run_pipeline --skip-embeddings
```

- Create local evaluation queries:

```bash
python -m src.pipeline.run_pipeline --auto-create-queries --num-auto-queries 5
```

- Run experiments after pipeline:

```bash
python -m src.pipeline.run_pipeline --run-experiments --queries data/evaluation/local_retrieval_queries.json
```

- Use the wrapper for a common workflow:

```bash
python -m run_all --run-experiments --auto-create-queries --start-dashboard
```

- Run tests:

```bash
python -m pytest -q
```

## Recommended Python version

Use Python 3.10 or 3.11.

## Project structure

- `data/raw/` — raw input data
- `data/processed/` — cleaned and intermediate outputs
- `data/graphs/` — exported graphs and filtered subgraphs
- `reports/tables/` — retrieval and evaluation results
- `artifacts/` — saved embeddings and retrieval artifacts
- `src/` — implementation modules
- `tests/` — test suite

## Dashboard capabilities

The Streamlit dashboard provides:

- BM25 search over processed documents
- LDA topic visualization
- topic-word embedding projection with nearest neighbors
- document clustering from saved embeddings
- interactive knowledge graph exploration and filtering
- filtered graph save and node export

## Data expectations

The pipeline expects arXiv-like metadata fields such as:

- `id`
- `title`
- `abstract`
- `authors`
- `categories`
- `update_date`

If no raw dataset is available, the pipeline can create a sample dataset automatically.

## Notes

- The project is research-oriented and not yet production hardened.
- Large datasets and generated artifacts should not be committed to version control.
- The dashboard and advanced visualization features are optional and depend on installed visualization packages.

## How to contribute

- Add dataset loaders and preprocessing enhancements
- Improve graph extraction and KG modeling
- Add retrieval evaluation datasets and benchmarks
- Expand dashboard visualizations and user workflows
- Add GNN, LLM, and bias analysis modules

## Contact

For implementation questions or project updates, inspect the `src/` modules and tests. Use the project structure to locate pipeline, extraction, retrieval, graph, and visualization code.

- `data/processed/` — cleaned or transformed intermediate data
- `data/graphs/` — exported knowledge graphs
- `reports/figures/` — plots and visualizations generated during experiments

Large datasets and generated files should not be committed to Git.

---

## Dataset

The project is designed around the arXiv scientific papers metadata dataset.

Expected fields:

- `id`
- `title`
- `abstract`
- `authors`
- `categories`
- `update_date`

Recommended categories for this project:

- `cs.AI`
- `cs.LG`
- `cs.CL`
- `cs.IR`

The full arXiv metadata file is very large. For normal development, a smaller filtered subset should be created first.

Recommended future command:

```bash
python scripts/create_arxiv_subset.py ^
  --input data/raw/arxiv-metadata-oai-snapshot.json ^
  --output data/raw/arxiv_cs_subset.csv ^
  --max-papers 10000
```

---

## Running the Pipeline

Run the end-to-end NLP pipeline:

```bash
python -m src.pipeline.run_pipeline --skip-embeddings
```

Run with a specific input file:

```bash
python -m src.pipeline.run_pipeline --input data/raw/arxiv_cs_subset.csv --skip-embeddings
```

Run with embeddings enabled:

```bash
python -m src.pipeline.run_pipeline --input data/raw/arxiv_cs_subset.csv
```

Run the full workflow with auto query creation and retrieval experiments:

```bash
python -m src.pipeline.run_pipeline --full
```

Create local evaluation queries from processed documents:

```bash
python -m src.pipeline.run_pipeline --auto-create-queries --num-auto-queries 5
```

Run experiments after the pipeline using a prepared query file:

```bash
python -m src.pipeline.run_pipeline --run-experiments --queries data/evaluation/local_retrieval_queries.json
```

Direct experiment runners are also available:

```bash
python -m src.experiments.run_bm25_baseline
python -m src.experiments.run_semantic_retrieval
python -m src.experiments.run_kg_enhanced_retrieval
```

The pipeline generates outputs such as:

```text
data/processed/processed_documents.csv
data/processed/entities.csv
data/processed/keywords.csv
data/processed/relations.csv
data/processed/bm25_search_results.json
data/processed/graph_summary.json
data/processed/lda_topics.json
data/graphs/knowledge_graph.graphml
data/graphs/knowledge_graph.json
data/graphs/semantic_network.graphml
reports/figures/knowledge_graph.html
reports/tables/bm25_results.csv
reports/tables/bm25_metrics.csv
reports/tables/semantic_results.csv
reports/tables/semantic_metrics.csv
reports/tables/kg_enhanced_results.csv
reports/tables/kg_enhanced_metrics.csv
```

---

## Notebooks

Implemented notebooks:

```text
notebooks/01_data_exploration.ipynb
notebooks/02_information_extraction.ipynb
notebooks/03_topic_modeling.ipynb
notebooks/04_embeddings_networks.ipynb
notebooks/05_bias_analysis.ipynb
notebooks/06_gnn_experiments.ipynb
notebooks/07_retrieval_evaluation_and_benchmarking.ipynb
```

Recommended notebook workflow:

1. Run the pipeline first
2. Load processed outputs in notebooks
3. Analyze and visualize results
4. Clear outputs before committing notebooks

Clear notebook outputs:

```bash
jupyter nbconvert --clear-output --inplace notebooks/01_data_exploration.ipynb notebooks/02_information_extraction.ipynb notebooks/03_topic_modeling.ipynb notebooks/04_embeddings_networks.ipynb notebooks/05_bias_analysis.ipynb notebooks/06_gnn_experiments.ipynb notebooks/07_retrieval_evaluation_and_benchmarking.ipynb
```

---

## Source Code Syntax Check

Run this first to verify that all Python files compile:

```bash
python -m compileall src
```

Expected result: no Python syntax errors.

---

## Running Tests

Run all tests:

```bash
pytest -v
```

Run pipeline tests only:

```bash
pytest tests/test_pipeline.py -v
```

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

---

## Optional Semantic Retrieval Smoke Test

This test verifies the embedding-based retrieval path.

It may download a Sentence Transformers model on the first run and can take longer than the basic smoke tests.

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

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Start Jupyter or Python from the repository root, or make sure the repository root is added to `sys.path`.

### `spaCy model 'en_core_web_sm' is not installed`

Run:

```bash
python -m spacy download en_core_web_sm
```

### `ModuleNotFoundError: No module named 'rank_bm25'`

Run:

```bash
pip install rank-bm25
```

### `ModuleNotFoundError: No module named 'faiss'`

Run:

```bash
pip install faiss-cpu
```

### PyTorch import errors

Reinstall PyTorch in the active environment:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Full arXiv JSON causes memory errors

Do not load the full arXiv metadata dump directly inside notebooks. Create a smaller filtered subset first.

---

# Development Roadmap

The next development phase should focus on making the project scalable, evaluable, and directly aligned with the research question.

Recommended strategy:

1. Make arXiv loading scalable
2. Create a reusable arXiv subset
3. Run BM25 baseline retrieval
4. Run semantic retrieval
5. Add retrieval metrics
6. Compare BM25 and semantic retrieval
7. Add knowledge graph-enhanced retrieval
8. Generate final analysis outputs
9. Add dashboard/report visualizations
10. Only then work on optional extensions such as GNNs, LLM + KG, and bias analysis

---

## Commit 1 — Add Scalable arXiv Subset Generation

### Goal

Avoid loading the full arXiv metadata dump into memory.

The full arXiv JSON file is too large for notebooks and normal experimentation. Instead, the project should create a smaller CSV subset containing only selected Computer Science categories.

### Implement

Create:

```text
scripts/create_arxiv_subset.py
```

The script should:

- Stream the arXiv JSON file line by line
- Filter papers by categories:
  - `cs.AI`
  - `cs.LG`
  - `cs.CL`
  - `cs.IR`
- Keep only necessary fields:
  - `id`
  - `title`
  - `abstract`
  - `authors`
  - `categories`
  - `update_date`
- Save a manageable subset to:
  - `data/raw/arxiv_cs_subset.csv`
- Support command-line arguments:
  - `--input`
  - `--output`
  - `--max-papers`
  - `--categories`

### Example Command

```bash
python scripts/create_arxiv_subset.py ^
  --input data/raw/arxiv-metadata-oai-snapshot.json ^
  --output data/raw/arxiv_cs_subset.csv ^
  --max-papers 10000
```



---

## Commit 2 — Refactor Dataset Loading for Large JSON Files

### Goal

Make `load_arxiv_jsonl()` memory-safe.

Currently, the loader can read the entire JSON/JSONL file into a list before creating a DataFrame. This can cause memory errors with the full arXiv dataset.

### Implement

Update:

```text
src/preprocessing/document_loader.py
```

Improve:

```python
load_arxiv_jsonl()
```

so that it:

- Streams records line by line
- Applies category filtering while reading
- Stops after `sample_size` matching papers
- Avoids storing millions of records in memory

### Add Tests

Add tests to:

```text
tests/test_preprocessing.py
```

Test cases:

- Loads JSONL sample correctly
- Filters by category
- Respects `sample_size`
- Produces standardized columns



---

## Commit 3 — Add Retrieval Evaluation Metrics

### Goal

Implement the evaluation metrics required in the project proposal.

The proposal mentions:

- Precision@K
- Recall@K
- Mean Reciprocal Rank

### Implement

Create:

```text
src/evaluation/__init__.py
src/evaluation/retrieval_metrics.py
```

Functions:

```python
precision_at_k(retrieved_ids, relevant_ids, k)
recall_at_k(retrieved_ids, relevant_ids, k)
mean_reciprocal_rank(results, relevance_sets)
evaluate_retrieval_run(...)
```

### Add Tests

Create:

```text
tests/test_evaluation.py
```

Test:

- Perfect ranking
- Empty relevant set
- No retrieved relevant result
- Partial match
- MRR calculation



## Commit 4 — Add Manual Relevance Evaluation Set

### Goal

Create a small evaluation dataset for retrieval experiments.

Since the arXiv dataset does not come with relevance labels for custom queries, the project needs a small manually defined evaluation set.

### Implement

Create:

```text
data/evaluation/retrieval_queries.example.json
```

or:

```text
evaluation/retrieval_queries.example.json
```

Suggested structure:

```json
[
  {
    "query": "knowledge graph semantic retrieval",
    "relevant_doc_ids": ["..."]
  },
  {
    "query": "graph neural networks for link prediction",
    "relevant_doc_ids": ["..."]
  }
]
```

Because real evaluation labels may be dataset-specific, commit an example/template file and document how to create the real one locally.



## Commit 5 — Add BM25 Baseline Experiment Runner

### Goal

Create a reproducible script for the baseline retrieval experiment.

### Implement

Create:

```text
src/experiments/run_bm25_baseline.py
```

The script should:

- Load `processed_documents.csv`
- Build a BM25 index
- Load evaluation queries
- Retrieve top-k documents
- Compute Precision@K, Recall@K, and MRR
- Save results to:
  - `reports/tables/bm25_results.csv`
  - `reports/tables/bm25_metrics.csv`

### Command

```bash
python -m src.experiments.run_bm25_baseline
```


## Commit 6 — Add Semantic Retrieval Experiment Runner

### Goal

Evaluate Sentence Transformer retrieval against BM25.

### Implement

Create:

```text
src/experiments/run_semantic_retrieval.py
```

The script should:

- Load processed documents
- Build document embeddings
- Run semantic search for evaluation queries
- Save retrieved results
- Compute the same metrics as BM25
- Save outputs to:
  - `reports/tables/semantic_results.csv`
  - `reports/tables/semantic_metrics.csv`

### Recommended Subset Size

Use a small subset first because embeddings can be slow:

```yaml
sample_size: 1000
```

Then increase to:

```yaml
sample_size: 5000
```

or:

```yaml
sample_size: 10000
```



## Commit 7 — Add BM25 vs Semantic Retrieval Comparison Notebook

### Goal

Add an analysis notebook that directly compares retrieval methods.

### Implement

Create:

```text
notebooks/07_retrieval_evaluation.ipynb
```

The notebook should include:

- BM25 results table
- Semantic retrieval results table
- Precision@K comparison
- Recall@K comparison
- MRR comparison
- Example query analysis
- Error analysis

### Visualizations

Add:

- Bar chart: BM25 vs Semantic Precision@K
- Bar chart: BM25 vs Semantic Recall@K
- Table of qualitative examples




        ...
```

### Outputs

Save:

```text
reports/tables/kg_enhanced_results.csv
reports/tables/kg_enhanced_metrics.csv
```



## Commit 9 — Add Retrieval Method Comparison Experiment

### Goal

Compare all retrieval approaches in one experiment.

Methods:

- BM25
- Semantic Retrieval
- KG-enhanced Retrieval

### Implement

Create:

```text
src/experiments/compare_retrieval_methods.py
```

The script should:

- Run all retrieval methods
- Evaluate them with the same query set
- Save a final comparison table
- Save qualitative examples

### Output

```text
reports/tables/retrieval_comparison.csv
```

Example table:

| method | precision@5 | recall@5 | mrr |
|---|---:|---:|---:|
| BM25 | ... | ... | ... |
| Semantic | ... | ... | ... |
| KG-enhanced | ... | ... | ... |



## Commit 10 — Improve Knowledge Graph Analysis

### Goal

Make the graph analysis more useful for interpretation.

### Implement

Extend:

```text
src/knowledge_graph/graph_queries.py
src/knowledge_graph/graph_visualization.py
```

Add:

- Top central concepts
- Top connected authors
- Top paper-concept relations
- Connected component summary
- Degree distribution
- PageRank table
- Community detection if feasible

### Output

```text
reports/tables/graph_statistics.csv
reports/figures/knowledge_graph_top_nodes.png
```


## Commit 11 — Add Streamlit Dashboard

### Goal

Create a simple interactive project dashboard.

### Implement

Update:

```text
src/visualization/dashboard.py
```

Dashboard pages:

- Dataset overview
- Search documents
- BM25 search
- Semantic search
- Knowledge graph statistics
- Topic modeling results
- Retrieval comparison

### Run Command

```bash
streamlit run src/visualization/dashboard.py
```



## Commit 12 — Add Final Report Tables and Figures

### Goal

Generate outputs needed for the final project report and presentation.

### Implement

Create:

```text
src/experiments/generate_report_assets.py
```

Generate:

- Dataset statistics table
- Category distribution figure
- Retrieval comparison table
- Topic modeling summaries
- Knowledge graph statistics
- Example graph visualization
- Semantic network visualization

### Output

```text
reports/tables/
reports/figures/
```

Keep generated outputs ignored if they are large, but document how to reproduce them.



## Commit 13 — Strengthen Tests

### Goal

Improve reliability before final submission.

### Implement Tests For

- `document_loader.py`
- `text_cleaning.py`
- `entity_extraction.py`
- `keyword_extraction.py`
- `relation_extraction.py`
- `bm25_retriever.py`
- `embedding_retriever.py`
- `graph_builder.py`
- `retrieval_metrics.py`
- `run_pipeline.py`

### Run

```bash
pytest -v
```



## Commit 14 — Update README with Real Usage Instructions

### Goal

Make the project understandable and reproducible.

### Update README Sections

- Project overview
- Dataset download instructions
- How to create the arXiv subset
- How to run the pipeline
- How to run retrieval experiments
- How to run notebooks
- How to run tests
- How to start dashboard
- Project structure
- Known limitations
- Future work



## Commit 15 — Optional: Add GNN Dataset Builder

### Goal

Start implementing the GNN part only after the graph pipeline is stable.

### Implement

Update:

```text
src/gnn/dataset_builder.py
```

It should:

- Load `knowledge_graph.graphml`
- Map nodes to integer IDs
- Create edge index
- Create simple node features
- Prepare a PyTorch Geometric dataset



## Commit 16 — Optional: Add Basic GCN Model

### Goal

Add a simple GNN baseline.

### Implement

Update:

```text
src/gnn/gcn_model.py
src/gnn/train_gnn.py
```

Possible tasks:

- Node type classification
- Link prediction
- Graph embedding learning



## Commit 17 — Optional: Add LLM + Knowledge Graph QA

### Goal

Add a lightweight LLM + KG module once retrieval is stable.

### Implement

Update:

```text
src/llm_kg/graph_retrieval_qa.py
src/llm_kg/rag_pipeline.py
```

Initial version can:

- Retrieve relevant documents
- Retrieve connected KG concepts
- Build context
- Generate or format an answer

If no external LLM API is used, implement a local context-building QA stub first.



## Commit 18 — Optional: Add Bias Analysis

### Goal

Implement bias and debiasing only after the core project is complete.

### Implement

Update:

```text
src/bias/embedding_bias.py
src/bias/llm_bias_eval.py
src/bias/debiasing.py
notebooks/05_bias_analysis.ipynb
```

Suggested focus:

- Bias in scientific embeddings
- Gendered word associations
- Topic/domain imbalance
- Prompt-based bias inspection



## Commit 19 — Final Cleanup

### Goal

Prepare the repository for final submission.

### Tasks

- Clear notebook outputs
- Remove temporary files
- Verify `.gitignore`
- Run tests
- Run pipeline on small subset
- Verify README commands
- Check no datasets are committed
- Check no generated reports are committed unless intentionally included

### Commands

```bash
pytest -v

jupyter nbconvert --clear-output --inplace notebooks/01_data_exploration.ipynb notebooks/02_information_extraction.ipynb notebooks/03_topic_modeling.ipynb notebooks/04_embeddings_networks.ipynb notebooks/07_retrieval_evaluation.ipynb

git status
git ls-files data
git ls-files reports
```



## Priority Plan

If time is limited, focus only on these milestones:

1. Scalable arXiv subset generation
2. Memory-safe dataset loading
3. Retrieval evaluation metrics
4. BM25 baseline experiment
5. Semantic retrieval experiment
6. KG-enhanced retrieval
7. Retrieval comparison notebook
8. README update
9. Final cleanup

These directly answer the project proposal and research question.

The GNN, LLM + KG, and bias modules are valuable extensions, but they should not distract from the core scientific document retrieval and knowledge graph evaluation workflow.

---

## Expected Deliverables

By the end of the project, the repository should include:

- Complete GitHub repository
- Scientific document preprocessing pipeline
- Information extraction system
- Knowledge graph construction
- BM25 retrieval baseline
- Semantic retrieval system
- Knowledge graph-enhanced retrieval
- Retrieval evaluation metrics
- Topic modeling analysis
- Embedding and semantic network analysis
- Final notebooks and visualizations
- Optional GNN experiments
- Optional LLM + KG integration
- Optional bias evaluation framework
- Final report and presentation materials

---

## Future Improvements

Possible extensions include:

- Multilingual NLP support
- Real-time document ingestion
- Advanced relation extraction
- Larger graph databases
- Conversational graph assistants
- Scientific recommendation systems
- Explainable graph reasoning
- Stronger entity linking
- Hybrid retrieval with BM25 + dense retrieval + graph reranking

---

## Conclusion

This project combines several advanced NLP and AI research areas into one unified knowledge discovery platform.

It demonstrates how unstructured scientific text can be transformed into structured knowledge, how semantic retrieval can improve information access, how graph-based methods can support relationship discovery, and how responsible AI methods can be added to evaluate fairness and bias in modern NLP systems.
