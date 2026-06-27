# NLP Knowledge Discovery Platform

## Overview

The **NLP Knowledge Discovery Platform** is an end-to-end research prototype for exploring and retrieving scientific publications. It combines Natural Language Processing, Information Retrieval, semantic embeddings, and a Knowledge Graph built from arXiv metadata.

The primary use case is scientific literature discovery: a student, researcher, or R&D team can search a collection of papers and compare three retrieval strategies:

1. **BM25** for lexical keyword matching;
2. **Semantic Retrieval** using Sentence Transformer embeddings;
3. **Knowledge Graph-Enhanced Retrieval** for graph-aware reranking.

The central research question is:

> Can information from a scientific Knowledge Graph improve document discovery compared with lexical and semantic retrieval alone?

This repository is a reproducible academic prototype. It is not intended to replace a production search engine such as Google Scholar.

---

## Main Capabilities

### Data preparation and preprocessing

- Loading arXiv metadata from CSV, JSON, and JSONL files
- Filtering and reproducible subset creation
- Text normalization and preprocessing
- spaCy-based linguistic processing

### Information extraction

- Named Entity Recognition
- TF-IDF and TextRank keyword extraction
- Co-occurrence relation extraction
- Aggregation of extracted concepts and relations

### Knowledge Graph

- Typed `PAPER`, `AUTHOR`, `TOPIC`, and `CONCEPT` nodes
- `AUTHORED_BY`, `BELONGS_TO_TOPIC`, `MENTIONS`, and relation edges
- GraphML and node-link JSON export
- Interactive visualization and graph filtering
- Graph statistics and report-ready tables

### Retrieval and evaluation

- BM25 baseline retrieval
- Sentence Transformer semantic retrieval
- Knowledge Graph-enhanced reranking
- Precision@K, Recall@K, and Mean Reciprocal Rank
- Query-level ranking comparison
- Synthetic silver-standard evaluation
- One manually labelled qualitative case study

### Topic modeling and extensions

- LDA topic modeling
- BERTopic integration
- Document and network embeddings
- Streamlit dashboard
- Prototype modules for GNN, bias analysis, and LLM/Knowledge Graph experiments

The advanced modules are exploratory extensions. The core evaluated workflow is preprocessing, extraction, graph construction, BM25, semantic retrieval, KG-enhanced retrieval, and evaluation.

---

## Architecture

```text
arXiv Metadata
      |
      v
Document Loading and Text Preprocessing
      |
      v
Entity, Keyword, and Relation Extraction
      |
      v
Knowledge Graph Construction
      |
      +----------------------+----------------------+
      |                      |                      |
      v                      v                      v
   BM25 Retrieval      Semantic Retrieval      Graph Evidence
      |                      |                      |
      +----------------------+----------+-----------+
                                        |
                                        v
                           KG-Enhanced Reranking
                                        |
                                        v
                         Evaluation, Reports, Dashboard
```

---

## Repository Structure

```text
.
├── config.yaml
├── run_all.py
├── create_queries.py
├── prepare_and_install_dep.txt
├── requirements.txt
├── scripts/
│   ├── create_arxiv_subset.py
│   └── run_full_validation.sh
├── src/
│   ├── preprocessing/
│   ├── extraction/
│   ├── knowledge_graph/
│   ├── retrieval/
│   ├── evaluation/
│   ├── experiments/
│   ├── topic_modeling/
│   ├── embeddings/
│   ├── visualization/
│   ├── gnn/
│   ├── bias/
│   └── llm_kg/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_information_extraction.ipynb
│   ├── 03_topic_modeling.ipynb
│   ├── 04_embeddings_networks.ipynb
│   ├── 05_bias_analysis.ipynb
│   ├── 06_gnn_experiments.ipynb
│   └── 07_retrieval_evaluation_and_benchmarking.ipynb
├── tests/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── evaluation/
│   └── graphs/
├── reports/
│   ├── tables/
│   ├── figures/
│   └── manual_test/
└── logs/
    └── e2e/
```

Large datasets, generated reports, embeddings, images, and logs are ignored by Git.

---

## Recommended Environment

- Linux
- Conda or Miniforge
- Python 3.11

Create and activate the environment:

```bash
conda create -n nlp-kg python=3.11 -y
conda activate nlp-kg
```

Install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install kaggle
python -m spacy download en_core_web_sm
```

Optional Jupyter kernel registration:

```bash
python -m ipykernel install \
  --user \
  --name nlp-kg \
  --display-name "Python (nlp-kg)"
```

Detailed setup and troubleshooting instructions are available in [`prepare_and_install_dep.txt`](prepare_and_install_dep.txt).

---

## Quick Start: Complete Reproducible Validation

After cloning the repository and activating the Conda environment:

```bash
chmod +x scripts/run_full_validation.sh
./scripts/run_full_validation.sh
```

This is the recommended command for a complete local validation. It:

- checks the active Conda environment and Python interpreter;
- installs or validates dependencies;
- validates the spaCy model;
- runs static compilation and all automated tests;
- downloads the public Cornell arXiv metadata file when it is missing;
- creates a reproducible 500-paper subset from `cs.AI`, `cs.LG`, `cs.CL`, and `cs.IR`;
- runs the complete NLP and Knowledge Graph pipeline;
- evaluates BM25, semantic, and KG-enhanced retrieval;
- generates report tables and figures using real experiment metrics;
- runs the manually labelled qualitative case study;
- prints metric and ranking summaries;
- writes a complete log to `logs/e2e/`;
- starts the Streamlit dashboard.

Useful alternatives:

```bash
# Complete validation without the dashboard
./scripts/run_full_validation.sh --no-dashboard

# Skip dependency installation in an already prepared environment
./scripts/run_full_validation.sh --skip-install --no-dashboard

# Keep existing generated outputs
./scripts/run_full_validation.sh --keep-outputs --no-dashboard

# Show all supported options
./scripts/run_full_validation.sh --help
```

### arXiv dataset download

The complete metadata file is approximately 5 GB. The script stores it at:

```text
data/raw/arxiv-metadata-oai-snapshot.json
```

The file is reused when it already exists. The dataset is public and normally downloads without an API token, although Kaggle may request authentication in some environments or after service-side policy or rate-limit changes.

---

## Running the Workflow Directly

Run the complete pipeline and all retrieval experiments on an existing subset:

```bash
python run_all.py \
  --input data/raw/arxiv_real_500.csv \
  --full \
  --skip-embeddings \
  --num-auto-queries 10
```

`--skip-embeddings` skips the additional semantic-network generation step. It does not disable the semantic retrieval experiment.

Run the pipeline without the experiment suite:

```bash
python -m src.pipeline.run_pipeline \
  --input data/raw/arxiv_real_500.csv \
  --skip-embeddings
```

Launch the dashboard:

```bash
python -m streamlit run src/visualization/dashboard.py
```

---

## Retrieval Experiments

### BM25

```bash
python -m src.experiments.run_bm25_baseline \
  --documents data/processed/processed_documents.csv \
  --queries data/evaluation/local_retrieval_queries.json \
  --output-dir reports/tables \
  --top-k 5 10
```

### Semantic Retrieval

```bash
python -m src.experiments.run_semantic_retrieval \
  --documents data/processed/processed_documents.csv \
  --queries data/evaluation/local_retrieval_queries.json \
  --output-dir reports/tables \
  --top-k 5 10
```

### Knowledge Graph-Enhanced Retrieval

```bash
python -m src.experiments.run_kg_enhanced_retrieval \
  --documents data/processed/processed_documents.csv \
  --queries data/evaluation/local_retrieval_queries.json \
  --graph data/graphs/knowledge_graph.graphml \
  --keywords data/processed/keywords.csv \
  --output-dir reports/tables \
  --base-method semantic \
  --top-k 5 10
```

---

## Report Assets

After the experiments have created the three real metric files, generate the report tables and figures:

```bash
python -m src.experiments.generate_report_assets \
  --processed-documents data/processed/processed_documents.csv \
  --entities data/processed/entities.csv \
  --keywords data/processed/keywords.csv \
  --relations data/processed/relations.csv \
  --graph data/graphs/knowledge_graph.json \
  --output-dir reports \
  --retrieval-metrics-path reports/tables/bm25_metrics.csv \
  --retrieval-metrics-path reports/tables/semantic_metrics.csv \
  --retrieval-metrics-path reports/tables/kg_enhanced_metrics.csv
```

The report generator requires real experiment metrics. It does not create comparison figures from hardcoded fallback values.

---

## Generated Outputs

### Processed data

```text
data/processed/
├── processed_documents.csv
├── entities.csv
├── keywords.csv
├── relations.csv
├── lda_topics.json
├── graph_summary.json
└── bm25_search_results.json
```

### Knowledge Graph

```text
data/graphs/
├── knowledge_graph.graphml
└── knowledge_graph.json
```

### Retrieval experiment outputs

```text
reports/tables/
├── bm25_results.csv
├── bm25_metrics.csv
├── semantic_results.csv
├── semantic_metrics.csv
├── kg_enhanced_results.csv
└── kg_enhanced_metrics.csv
```

### Report tables

```text
reports/tables/
├── dataset_statistics.csv
├── graph_statistics.csv
├── top_authors.csv
├── top_concepts.csv
├── topic_summary.csv
├── retrieval_benchmark_comparison.csv
├── retrieval_ranking_change_summary.csv
└── retrieval_summary.csv
```

### Report figures

```text
reports/figures/
├── category_distribution.png
├── degree_distribution.png
├── knowledge_graph.html
├── knowledge_graph_top_nodes.png
├── retrieval_metrics_comparison.png
├── automatic_precision_at_k.png
├── automatic_recall_at_k.png
└── automatic_mrr.png
```

### Manual qualitative case study

```text
reports/manual_test/
├── bm25_results.csv
├── bm25_metrics.csv
├── semantic_results.csv
├── semantic_metrics.csv
├── kg_enhanced_results.csv
├── kg_enhanced_metrics.csv
├── manual_retrieval_benchmark.csv
└── manual_ranking_comparison.csv
```

Some Notebook 07 outputs are generated only after running that notebook.

---

## Evaluation Methodology

The project reports:

- **Precision@K**: the fraction of the top-K results that are relevant;
- **Recall@K**: the fraction of known relevant documents retrieved within the top K;
- **Mean Reciprocal Rank**: the reciprocal rank of the first relevant result, averaged over queries.

### Synthetic silver-standard evaluation

`run_all.py --full` can create local evaluation queries from the processed corpus. These queries are deterministic and useful for reproducible system comparison, but they are not independent human relevance judgements. Results from this setup must therefore be described as a **synthetic silver-standard evaluation**.

### Manual qualitative case study

The validation script also runs one manually labelled query over real arXiv papers. This case study demonstrates how graph-based reranking can change the ranking and recover a relevant document for a specific query.

Because it contains only one manually labelled query, it is a qualitative demonstration rather than statistically conclusive evidence of general superiority.

---

## Notebook Workflow

The notebooks provide the analysis and presentation layer. The implementation remains in `src/`, while the notebooks read generated artifacts and visualize or interpret them.

- `01_data_exploration.ipynb`: dataset inspection
- `02_information_extraction.ipynb`: entities, keywords, and relations
- `03_topic_modeling.ipynb`: topic modeling
- `04_embeddings_networks.ipynb`: embeddings and semantic networks
- `05_bias_analysis.ipynb`: exploratory bias analysis
- `06_gnn_experiments.ipynb`: exploratory graph neural network analysis
- `07_retrieval_evaluation_and_benchmarking.ipynb`: final retrieval comparison and case-study analysis

Before running Notebook 07, generate the experiment outputs with:

```bash
./scripts/run_full_validation.sh --no-dashboard
```

Then open JupyterLab, select the `nlp-kg` kernel, and run all cells:

```bash
jupyter lab
```

Notebook 07:

- validates all required CSV files;
- compares the three methods at each K;
- creates report-ready metric plots;
- performs query-level ranking analysis;
- compares Semantic and KG-enhanced rankings per query;
- analyzes the manual case when its outputs exist;
- clearly separates silver-standard findings from the qualitative manual result.

---

## Testing

Run all automated tests:

```bash
python -m pytest -q
```

Run static compilation:

```bash
python -m compileall -q src tests scripts run_all.py create_queries.py
```

Current verified status:

```text
60 passed
```

The test suite covers preprocessing, extraction, retrieval, evaluation metrics, pipeline behavior, Knowledge Graph reranking, report asset generation, and regression cases.

---

## Reproducibility and Limitations

- The default validation subset is deterministic for the configured seed and source snapshot.
- Sentence Transformer model weights may be downloaded during the first semantic run.
- Generated files are intentionally excluded from Git and should be regenerated locally.
- The automatic relevance labels are synthetic silver-standard labels.
- The manual result is a single-query qualitative case study.
- The GNN, bias, and LLM/Knowledge Graph modules are exploratory extensions rather than the core evaluated contribution.
- This is a research prototype, not a production-hardened search service.

---

## Conclusion

The project provides a complete and reproducible research workflow from real arXiv metadata to structured information extraction, Knowledge Graph construction, three retrieval strategies, quantitative evaluation, report assets, notebooks, and an interactive dashboard.

Its main contribution is the controlled comparison of lexical, semantic, and graph-enhanced scientific document retrieval, together with transparent evaluation limitations and a reproducible end-to-end validation procedure.
