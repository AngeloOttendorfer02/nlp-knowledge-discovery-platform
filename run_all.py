"""Convenience wrapper for the full NLP Knowledge Discovery workflow.

This script intentionally stays thin. The canonical project workflow lives in:

    src.pipeline.run_pipeline

Recommended usage:

    python run_all.py --full --skip-embeddings

Run tests first:

    python run_all.py --run-tests

Run pipeline + experiments + dashboard:

    python run_all.py --full --skip-embeddings --start-dashboard
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], *, check: bool = True) -> int:
    print("\n" + "=" * 100)
    print("Running:", " ".join(command))
    print("=" * 100)

    completed = subprocess.run(command)

    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            + " ".join(command)
        )

    return completed.returncode


def ensure_spacy_model(model: str = "en_core_web_sm") -> None:
    try:
        import spacy  # type: ignore

        try:
            spacy.load(model)
            print(f"spaCy model '{model}' is available.")
            return
        except Exception:
            print(f"spaCy model '{model}' not found. Downloading...")
            run_command([sys.executable, "-m", "spacy", "download", model])

    except ImportError as exc:
        raise RuntimeError(
            "spaCy is not installed. Install project dependencies first:\n"
            "    pip install -r requirements.txt"
        ) from exc


def build_pipeline_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.pipeline.run_pipeline",
    ]

    if args.input:
        command.extend(["--input", args.input])

    if args.query:
        command.extend(["--query", args.query])

    if args.skip_embeddings:
        command.append("--skip-embeddings")

    if args.full:
        command.append("--full")

    if args.run_experiments:
        command.append("--run-experiments")

    if args.auto_create_queries:
        command.append("--auto-create-queries")

    if args.queries:
        command.extend(["--queries", args.queries])

    command.extend(["--num-auto-queries", str(args.num_auto_queries)])

    return command


def launch_dashboard() -> None:
    dashboard_path = Path("src/visualization/dashboard.py")

    if not dashboard_path.exists():
        raise FileNotFoundError(f"Dashboard file not found: {dashboard_path}")

    run_command(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NLP Knowledge Discovery project workflow."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional input CSV, JSON, or JSONL dataset.",
    )

    parser.add_argument(
        "--query",
        type=str,
        default="knowledge graphs and language models",
        help="Example BM25 query used during pipeline execution.",
    )

    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip pipeline semantic-network creation for faster execution.",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run pipeline, auto-create evaluation queries, and run all experiments.",
    )

    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run retrieval experiments after the pipeline.",
    )

    parser.add_argument(
        "--auto-create-queries",
        action="store_true",
        help="Create local evaluation queries from processed documents.",
    )

    parser.add_argument(
        "--queries",
        type=str,
        default="data/evaluation/local_retrieval_queries.json",
        help="Evaluation query JSON path.",
    )

    parser.add_argument(
        "--num-auto-queries",
        type=int,
        default=10,
        help="Number of automatic local evaluation queries.",
    )

    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run pytest before the workflow.",
    )

    parser.add_argument(
        "--start-dashboard",
        action="store_true",
        help="Launch the Streamlit dashboard after the workflow.",
    )

    parser.add_argument(
        "--skip-spacy-check",
        action="store_true",
        help="Do not check/download the spaCy model.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.skip_spacy_check:
        ensure_spacy_model()

    if args.run_tests:
        run_command([sys.executable, "-m", "pytest", "-v"])

    command = build_pipeline_command(args)
    run_command(command)

    if args.start_dashboard:
        launch_dashboard()

    print("\nWorkflow completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())