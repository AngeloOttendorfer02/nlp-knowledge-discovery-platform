#!/usr/bin/env bash
# ==============================================================================
# NLP Knowledge Discovery Platform - Full End-to-End Validation Runner
# ==============================================================================
# Purpose
# -------
# Run the complete project workflow with one command after:
#   1. cloning the Git repository;
#   2. activating a Conda environment.
#
# The script performs:
#   - environment and repository validation;
#   - dependency installation/checks;
#   - compileall and pytest;
#   - spaCy model validation/download;
#   - arXiv metadata download from Kaggle when missing;
#   - creation of a reproducible 500-paper real-world subset;
#   - cleanup of previously generated outputs;
#   - full NLP pipeline execution;
#   - BM25, semantic, and KG-enhanced automatic evaluation;
#   - report table and figure generation using real metrics only;
#   - one manually labelled retrieval case study;
#   - metric and ranking summaries;
#   - optional Streamlit dashboard startup.
#
# Logging
# -------
# All output is shown in the terminal and written to:
#   logs/e2e/full_validation_<timestamp>.log
#
# Default usage
# -------------
#   conda activate nlp-kg
#   chmod +x scripts/run_full_validation.sh
#   ./scripts/run_full_validation.sh
#
# Non-interactive validation without starting Streamlit
# -----------------------------------------------------
#   ./scripts/run_full_validation.sh --no-dashboard
#
# Reuse an already prepared local dataset and skip package installation
# ----------------------------------------------------------------------
#   ./scripts/run_full_validation.sh --skip-install --no-dashboard
# ==============================================================================

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_NAME="$(basename "$0")"
VALIDATION_COMPLETE=0

# ----------------------------- Default configuration ---------------------------
INSTALL_DEPS=1
START_DASHBOARD=1
CLEAN_OUTPUTS=1
RUN_MANUAL_CASE=1
FORCE_SUBSET=0

SUBSET_SIZE=500
AUTO_QUERY_COUNT=10
SEED=42

RAW_DATASET="data/raw/arxiv-metadata-oai-snapshot.json"
SUBSET_DATASET="data/raw/arxiv_real_500.csv"
MANUAL_QUERIES="data/evaluation/manual_retrieval_queries.json"
MANUAL_OUTPUT_DIR="reports/manual_test"

ARXIV_CATEGORIES=("cs.AI" "cs.LG" "cs.CL" "cs.IR")

# The manual case is intentionally based on papers that are present in the
# reproducible 500-paper subset created from the Cornell arXiv metadata dump.
MANUAL_RELEVANT_IDS=("706.4375" "711.3128" "712.3298")
MANUAL_QUERY="semantic information retrieval using natural language processing, entity ranking and network analysis"

usage() {
    cat <<EOF
Usage: ./${SCRIPT_NAME} [options]

Options:
  --no-dashboard       Run all tests but do not start the Streamlit dashboard.
  --skip-install       Do not run pip install commands; only validate imports.
  --keep-outputs       Keep existing generated outputs instead of cleaning them.
  --skip-manual        Skip the manually labelled retrieval case study.
  --force-subset       Recreate the arXiv subset even if it already exists.
  --subset-size N      Number of matching arXiv papers in the subset (default: 500).
  --auto-queries N     Number of synthetic silver-standard queries (default: 10).
  --raw-dataset PATH   Path to the full arXiv metadata JSONL file.
  --subset PATH        Path for the generated CSV subset.
  -h, --help           Show this help message.

Examples:
  ./${SCRIPT_NAME}
  ./${SCRIPT_NAME} --no-dashboard
  ./${SCRIPT_NAME} --skip-install --keep-outputs --no-dashboard
EOF
}

while (($# > 0)); do
    case "$1" in
        --no-dashboard)
            START_DASHBOARD=0
            shift
            ;;
        --skip-install)
            INSTALL_DEPS=0
            shift
            ;;
        --keep-outputs)
            CLEAN_OUTPUTS=0
            shift
            ;;
        --skip-manual)
            RUN_MANUAL_CASE=0
            shift
            ;;
        --force-subset)
            FORCE_SUBSET=1
            shift
            ;;
        --subset-size)
            [[ $# -ge 2 ]] || { echo "ERROR: --subset-size requires a value." >&2; exit 2; }
            SUBSET_SIZE="$2"
            shift 2
            ;;
        --auto-queries)
            [[ $# -ge 2 ]] || { echo "ERROR: --auto-queries requires a value." >&2; exit 2; }
            AUTO_QUERY_COUNT="$2"
            shift 2
            ;;
        --raw-dataset)
            [[ $# -ge 2 ]] || { echo "ERROR: --raw-dataset requires a path." >&2; exit 2; }
            RAW_DATASET="$2"
            shift 2
            ;;
        --subset)
            [[ $# -ge 2 ]] || { echo "ERROR: --subset requires a path." >&2; exit 2; }
            SUBSET_DATASET="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ "$SUBSET_SIZE" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: --subset-size must be a positive integer." >&2
    exit 2
}

[[ "$AUTO_QUERY_COUNT" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: --auto-queries must be a positive integer." >&2
    exit 2
}

# Resolve the repository root. The script may live either in the repository root
# or in the repository's scripts/ directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_all.py" ]]; then
    REPO_ROOT="$SCRIPT_DIR"
elif [[ -f "${SCRIPT_DIR}/../run_all.py" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "ERROR: Could not locate repository root. Expected run_all.py next to the script or one directory above it." >&2
    exit 1
fi
cd "$REPO_ROOT"

mkdir -p logs/e2e
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="logs/e2e/full_validation_${RUN_TIMESTAMP}.log"
LATEST_LOG="logs/e2e/latest.log"

# Show every message in the terminal and persist the same output in a log file.
touch "$LOG_FILE"
ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LOG"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
    local exit_code=$?
    local line_no=${BASH_LINENO[0]:-unknown}
    local command=${BASH_COMMAND:-unknown}
    echo
    echo "================================================================================"
    echo "VALIDATION FAILED"
    echo "Exit code : ${exit_code}"
    echo "Line      : ${line_no}"
    echo "Command   : ${command}"
    echo "Log file  : ${REPO_ROOT}/${LOG_FILE}"
    echo "================================================================================"
    exit "$exit_code"
}

on_interrupt() {
    echo
    if [[ "$VALIDATION_COMPLETE" -eq 1 ]]; then
        echo "Dashboard stopped by the user. Automated validation had already completed successfully."
        echo "Log file: ${REPO_ROOT}/${LOG_FILE}"
        exit 0
    fi

    echo "Validation interrupted by the user."
    echo "Log file: ${REPO_ROOT}/${LOG_FILE}"
    exit 130
}

trap on_error ERR
trap on_interrupt INT TERM

section() {
    local title="$1"
    echo
    echo "================================================================================"
    echo "$title"
    echo "================================================================================"
}

run_cmd() {
    echo
    printf '+ '
    printf '%q ' "$@"
    echo
    "$@"
}

require_file() {
    local path="$1"
    [[ -s "$path" ]] || {
        echo "ERROR: Required file is missing or empty: $path" >&2
        return 1
    }
}

section "1/14 - Run metadata and environment validation"
echo "Started at       : $(date --iso-8601=seconds)"
echo "Repository root : $REPO_ROOT"
echo "Log file        : $REPO_ROOT/$LOG_FILE"
echo "User            : ${USER:-unknown}"
echo "Host            : $(hostname)"
echo "Conda env       : ${CONDA_DEFAULT_ENV:-not-active}"
echo "Conda prefix    : ${CONDA_PREFIX:-not-active}"
echo "Python          : $(command -v python || true)"
echo "Python version  : $(python --version 2>&1 || true)"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: No active Conda environment detected. Run 'conda activate <environment>' first." >&2
    exit 1
fi

if [[ "$(command -v python)" != "${CONDA_PREFIX}"/* ]]; then
    echo "ERROR: The active Python interpreter is not inside CONDA_PREFIX." >&2
    echo "Python      : $(command -v python)" >&2
    echo "CONDA_PREFIX: ${CONDA_PREFIX}" >&2
    exit 1
fi

require_file "requirements.txt"
require_file "run_all.py"
require_file "scripts/create_arxiv_subset.py"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Git branch       : $(git branch --show-current)"
    echo "Git commit       : $(git rev-parse HEAD)"
    echo "Git short commit : $(git rev-parse --short HEAD)"
    echo "Git status:"
    git status --short || true
else
    echo "WARNING: This directory is not recognized as a Git worktree."
fi

section "2/14 - Dependency installation and validation"
if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    run_cmd python -m pip install -r requirements.txt

    if ! command -v kaggle >/dev/null 2>&1; then
        run_cmd python -m pip install kaggle
    fi
else
    echo "Dependency installation skipped by --skip-install."
fi

run_cmd python -m pip check

run_cmd python - <<'PY'
import importlib
import sys

required_modules = {
    "pandas": "pandas",
    "networkx": "networkx",
    "spacy": "spacy",
    "sklearn": "scikit-learn",
    "sentence_transformers": "sentence-transformers",
    "streamlit": "streamlit",
    "yaml": "PyYAML",
}

missing = []
for module_name, package_name in required_modules.items():
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"OK: {module_name} {version}")
    except Exception as exc:  # noqa: BLE001
        missing.append((module_name, package_name, str(exc)))

if missing:
    print("\nMissing or broken dependencies:", file=sys.stderr)
    for module_name, package_name, error in missing:
        print(f"- {module_name} ({package_name}): {error}", file=sys.stderr)
    raise SystemExit(1)
PY

section "3/14 - spaCy language model validation"
if python - <<'PY'
import spacy
spacy.load("en_core_web_sm")
print("spaCy model 'en_core_web_sm' is available.")
PY
then
    :
else
    echo "spaCy model 'en_core_web_sm' is missing. Downloading it now..."
    run_cmd python -m spacy download en_core_web_sm
fi

section "4/14 - Static compilation and automated tests"
run_cmd python -m compileall -q src tests scripts run_all.py create_queries.py
run_cmd python -m pytest -q

section "5/14 - Real arXiv metadata acquisition"
mkdir -p "$(dirname "$RAW_DATASET")"

if [[ -s "$RAW_DATASET" ]]; then
    echo "Using existing arXiv metadata file: $RAW_DATASET"
    ls -lh "$RAW_DATASET"
else
    if ! command -v kaggle >/dev/null 2>&1; then
        echo "ERROR: Kaggle CLI is not available. Re-run without --skip-install or install it with 'python -m pip install kaggle'." >&2
        exit 1
    fi

    echo "The arXiv metadata dump is not present and will now be downloaded."
    echo "Expected download size: approximately 5 GB."
    echo "Kaggle credentials must already be configured for the current user."

    run_cmd kaggle datasets download \
        -d Cornell-University/arxiv \
        -f arxiv-metadata-oai-snapshot.json \
        -p "$(dirname "$RAW_DATASET")" \
        --unzip

    require_file "$RAW_DATASET"
    ls -lh "$RAW_DATASET"
fi

section "6/14 - Reproducible real-world arXiv subset creation"
if [[ "$FORCE_SUBSET" -eq 1 || ! -s "$SUBSET_DATASET" ]]; then
    run_cmd python scripts/create_arxiv_subset.py \
        --input "$RAW_DATASET" \
        --output "$SUBSET_DATASET" \
        --max-papers "$SUBSET_SIZE" \
        --categories "${ARXIV_CATEGORIES[@]}" \
        --seed "$SEED"
else
    echo "Using existing subset: $SUBSET_DATASET"
fi

require_file "$SUBSET_DATASET"

run_cmd python - "$SUBSET_DATASET" "$SUBSET_SIZE" <<'PY'
from pathlib import Path
import sys

import pandas as pd

path = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
df = pd.read_csv(path, dtype={"id": "string"})

required_columns = {"id", "title", "abstract", "authors", "categories", "update_date"}
missing = sorted(required_columns.difference(df.columns))
if missing:
    raise SystemExit(f"Subset is missing required columns: {missing}")

if len(df) != expected_rows:
    raise SystemExit(f"Expected {expected_rows} rows, found {len(df)}")

if df["id"].isna().any() or df["id"].duplicated().any():
    raise SystemExit("Subset contains missing or duplicate document IDs")

print(f"Subset validation passed: {len(df)} documents")
print(f"Columns: {list(df.columns)}")
print("\nFirst 10 documents:")
print(df[["id", "title", "categories"]].head(10).to_string(index=False))
PY

section "7/14 - Cleanup of previously generated outputs"
if [[ "$CLEAN_OUTPUTS" -eq 1 ]]; then
    for directory in data/processed data/graphs reports/tables reports/figures; do
        mkdir -p "$directory"
        find "$directory" -type f ! -name '.gitkeep' -delete
    done

    rm -rf artifacts/semantic_retrieval "$MANUAL_OUTPUT_DIR"
    mkdir -p artifacts/semantic_retrieval "$MANUAL_OUTPUT_DIR"
    echo "Generated outputs were cleaned."
else
    echo "Existing outputs were preserved because --keep-outputs was supplied."
    mkdir -p "$MANUAL_OUTPUT_DIR"
fi

section "8/14 - Full NLP and retrieval workflow"
run_cmd python run_all.py \
    --input "$SUBSET_DATASET" \
    --full \
    --skip-embeddings \
    --num-auto-queries "$AUTO_QUERY_COUNT"

section "9/14 - Expected artifact validation"
EXPECTED_FILES=(
    "data/processed/processed_documents.csv"
    "data/processed/entities.csv"
    "data/processed/keywords.csv"
    "data/processed/relations.csv"
    "data/processed/lda_topics.json"
    "data/graphs/knowledge_graph.graphml"
    "data/graphs/knowledge_graph.json"
    "reports/tables/bm25_results.csv"
    "reports/tables/bm25_metrics.csv"
    "reports/tables/semantic_results.csv"
    "reports/tables/semantic_metrics.csv"
    "reports/tables/kg_enhanced_results.csv"
    "reports/tables/kg_enhanced_metrics.csv"
)

for path in "${EXPECTED_FILES[@]}"; do
    require_file "$path"
    echo "OK: $path"
done

section "10/14 - Automatic synthetic silver-standard metric summary"
run_cmd python - <<'PY'
from pathlib import Path

import pandas as pd

metric_files = [
    Path("reports/tables/bm25_metrics.csv"),
    Path("reports/tables/semantic_metrics.csv"),
    Path("reports/tables/kg_enhanced_metrics.csv"),
]

for path in metric_files:
    df = pd.read_csv(path)
    required = {"method", "k", "precision_at_k", "recall_at_k", "mrr", "num_queries"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")

    metric_columns = ["precision_at_k", "recall_at_k", "mrr"]
    if df[metric_columns].isna().any().any():
        raise SystemExit(f"{path} contains NaN metrics")

    if not ((df[metric_columns] >= 0.0) & (df[metric_columns] <= 1.0)).all().all():
        raise SystemExit(f"{path} contains metrics outside [0, 1]")

    print("\n" + "=" * 90)
    print(path)
    print("=" * 90)
    print(df.to_string(index=False))

print("\nNOTE: These automatically generated queries form a synthetic silver-standard evaluation.")
PY

section "11/14 - Report tables and figures generated from real metrics"
run_cmd python -m src.experiments.generate_report_assets \
    --processed-documents data/processed/processed_documents.csv \
    --entities data/processed/entities.csv \
    --keywords data/processed/keywords.csv \
    --relations data/processed/relations.csv \
    --graph data/graphs/knowledge_graph.json \
    --output-dir reports \
    --retrieval-metrics-path reports/tables/bm25_metrics.csv \
    --retrieval-metrics-path reports/tables/semantic_metrics.csv \
    --retrieval-metrics-path reports/tables/kg_enhanced_metrics.csv

REPORT_FILES=(
    "reports/tables/dataset_statistics.csv"
    "reports/tables/graph_statistics.csv"
    "reports/tables/top_authors.csv"
    "reports/tables/top_concepts.csv"
    "reports/tables/topic_summary.csv"
    "reports/figures/category_distribution.png"
    "reports/figures/degree_distribution.png"
    "reports/figures/knowledge_graph.html"
    "reports/figures/knowledge_graph_top_nodes.png"
    "reports/figures/retrieval_metrics_comparison.png"
)

for path in "${REPORT_FILES[@]}"; do
    require_file "$path"
    echo "OK: $path"
done

section "12/14 - Manually labelled real-world retrieval case study"
if [[ "$RUN_MANUAL_CASE" -eq 1 ]]; then
    mkdir -p "$(dirname "$MANUAL_QUERIES")" "$MANUAL_OUTPUT_DIR"

    cat > "$MANUAL_QUERIES" <<JSON
[
  {
    "query_id": "manual_q1",
    "query": "${MANUAL_QUERY}",
    "relevant_doc_ids": [
      "${MANUAL_RELEVANT_IDS[0]}",
      "${MANUAL_RELEVANT_IDS[1]}",
      "${MANUAL_RELEVANT_IDS[2]}"
    ],
    "notes": "Documents selected manually after reading their titles and abstracts. This is a qualitative case study."
  }
]
JSON

    if python - "$MANUAL_QUERIES" <<'PY'
import json
import sys

import pandas as pd

query_path = sys.argv[1]
documents = pd.read_csv(
    "data/processed/processed_documents.csv",
    dtype={"doc_id": "string"},
)
known_ids = set(documents["doc_id"].astype(str).str.strip())

with open(query_path, encoding="utf-8") as handle:
    queries = json.load(handle)

missing = []
for query in queries:
    print(f"Query: {query['query']}")
    for doc_id in query.get("relevant_doc_ids", []):
        doc_id = str(doc_id).strip()
        status = "OK" if doc_id in known_ids else "MISSING"
        print(f"  {doc_id}: {status}")
        if status == "MISSING":
            missing.append(doc_id)

if missing:
    print(f"Manual case cannot run because IDs are missing: {missing}", file=sys.stderr)
    raise SystemExit(3)
PY
    then
        rm -f "$MANUAL_OUTPUT_DIR"/*.csv

        run_cmd python -m src.experiments.run_bm25_baseline \
            --documents data/processed/processed_documents.csv \
            --queries "$MANUAL_QUERIES" \
            --output-dir "$MANUAL_OUTPUT_DIR" \
            --top-k 5 10

        run_cmd python -m src.experiments.run_semantic_retrieval \
            --documents data/processed/processed_documents.csv \
            --queries "$MANUAL_QUERIES" \
            --output-dir "$MANUAL_OUTPUT_DIR" \
            --top-k 5 10

        run_cmd python -m src.experiments.run_kg_enhanced_retrieval \
            --documents data/processed/processed_documents.csv \
            --queries "$MANUAL_QUERIES" \
            --graph data/graphs/knowledge_graph.graphml \
            --keywords data/processed/keywords.csv \
            --output-dir "$MANUAL_OUTPUT_DIR" \
            --base-method semantic \
            --top-k 5 10

        run_cmd python - <<'PY'
from pathlib import Path

import pandas as pd

output_dir = Path("reports/manual_test")
relevant = {"706.4375", "711.3128", "712.3298"}

documents = pd.read_csv(
    "data/processed/processed_documents.csv",
    dtype={"doc_id": "string"},
)[["doc_id", "title"]]

print("\nMANUAL CASE METRICS")
for path in sorted(output_dir.glob("*_metrics.csv")):
    metrics = pd.read_csv(path)
    print("\n" + "=" * 100)
    print(path.name)
    print("=" * 100)
    print(metrics.to_string(index=False))

print("\nMANUAL CASE RANKINGS")
for path in sorted(output_dir.glob("*_results.csv")):
    results = pd.read_csv(path, dtype={"doc_id": "string"})
    results = results.merge(documents, on="doc_id", how="left")
    results["relevant"] = results["doc_id"].isin(relevant)

    columns = [
        column
        for column in ["rank", "doc_id", "score", "relevant", "title"]
        if column in results.columns
    ]

    print("\n" + "=" * 110)
    print(path.name)
    print("=" * 110)
    print(results[columns].head(10).to_string(index=False))

metric_paths = {
    "BM25": output_dir / "bm25_metrics.csv",
    "Semantic": output_dir / "semantic_metrics.csv",
    "KG-enhanced": output_dir / "kg_enhanced_metrics.csv",
}

recall_at_5 = {}
for method, path in metric_paths.items():
    df = pd.read_csv(path)
    row = df.loc[df["k"] == 5]
    if row.empty:
        raise SystemExit(f"Missing k=5 metric row in {path}")
    recall_at_5[method] = float(row.iloc[0]["recall_at_k"])

print("\nMANUAL CASE INTERPRETATION")
for method, value in recall_at_5.items():
    print(f"- {method} Recall@5: {value:.6f}")

semantic_recall = recall_at_5["Semantic"]
kg_recall = recall_at_5["KG-enhanced"]
if kg_recall > semantic_recall:
    print(
        f"Observation: KG-enhanced improved Recall@5 over Semantic by "
        f"{kg_recall - semantic_recall:.6f}."
    )
elif kg_recall == semantic_recall:
    print("Observation: KG-enhanced and Semantic achieved the same Recall@5.")
else:
    print(
        "Observation: KG-enhanced did not improve Recall@5 for this run. "
        "This is a case-study result, not a pipeline failure."
    )

print(
    "Limitation: one manually labelled query is a qualitative case study and "
    "is not statistically conclusive."
)
PY
    else
        echo "WARNING: The fixed manual case was skipped because its document IDs are not present in this subset."
        echo "This can happen if the upstream arXiv snapshot or subset size changes."
    fi
else
    echo "Manual retrieval case skipped by --skip-manual."
fi

section "13/14 - Final generated-file inventory"
find data/processed data/graphs reports -maxdepth 2 -type f | sort

section "14/14 - Final validation summary"
echo "Automated validation completed successfully."
echo "Completed at      : $(date --iso-8601=seconds)"
echo "Repository commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Subset           : $SUBSET_DATASET (${SUBSET_SIZE} documents requested)"
echo "Automatic queries: $AUTO_QUERY_COUNT"
echo "Report directory : $REPO_ROOT/reports"
echo "Log file         : $REPO_ROOT/$LOG_FILE"
echo "Latest log link  : $REPO_ROOT/$LATEST_LOG"

VALIDATION_COMPLETE=1

if [[ "$START_DASHBOARD" -eq 1 ]]; then
    section "Starting Streamlit dashboard"
    echo "The automated validation is complete."
    echo "Open the Local URL printed by Streamlit in your browser."
    echo "Press Ctrl+C to stop the dashboard; the script will then exit successfully."
    run_cmd python -m streamlit run src/visualization/dashboard.py
else
    echo "Dashboard startup skipped by --no-dashboard."
fi
