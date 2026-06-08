"""
Project path and configuration utilities.

This module centralizes project-root detection, config loading, and commonly
used project directories. It is used by notebooks, scripts, and pipeline code.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    """
    Centralized access to project directories and generated artifacts.
    """

    project_root: Path
    config_path: Path

    data_raw: Path
    data_processed: Path
    data_graphs: Path

    reports: Path
    figures: Path

    # ------------------------------------------------------------------
    # Raw datasets
    # ------------------------------------------------------------------

    @property
    def sample_dataset(self) -> Path:
        return self.data_raw / "sample_documents.csv"

    @property
    def arxiv_dataset(self) -> Path:
        return self.data_raw / "arxiv-metadata-oai-snapshot.json"

    # ------------------------------------------------------------------
    # Processed outputs
    # ------------------------------------------------------------------

    @property
    def processed_documents(self) -> Path:
        return self.data_processed / "processed_documents.csv"

    @property
    def entities(self) -> Path:
        return self.data_processed / "entities.csv"

    @property
    def keywords(self) -> Path:
        return self.data_processed / "keywords.csv"

    @property
    def relations(self) -> Path:
        return self.data_processed / "relations.csv"

    @property
    def bm25_results(self) -> Path:
        return self.data_processed / "bm25_results.csv"

    # ------------------------------------------------------------------
    # Graph outputs
    # ------------------------------------------------------------------

    @property
    def knowledge_graph_graphml(self) -> Path:
        return self.data_graphs / "knowledge_graph.graphml"

    @property
    def knowledge_graph_json(self) -> Path:
        return self.data_graphs / "knowledge_graph.json"

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    @property
    def knowledge_graph_html(self) -> Path:
        return self.figures / "knowledge_graph.html"

    @property
    def topic_distribution_png(self) -> Path:
        return self.figures / "topic_distribution.png"

    @property
    def similarity_network_png(self) -> Path:
        return self.figures / "similarity_network.png"


def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Find the repository root by searching for src/ and config.yaml.
    """
    current = (start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "src").exists() and (candidate / "config.yaml").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find project root containing src/ and config.yaml."
    )


def add_project_root_to_path(project_root: Optional[Path] = None) -> Path:
    """
    Add the project root to sys.path if necessary.
    """
    root = project_root or find_project_root()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    return root


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load config.yaml.
    """
    project_root = find_project_root()
    path = config_path or project_root / "config.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_project_paths(config: Optional[Dict[str, Any]] = None) -> ProjectPaths:
    """
    Build a ProjectPaths object and ensure directories exist.
    """
    project_root = find_project_root()
    add_project_root_to_path(project_root)

    cfg = config or load_config()
    paths = cfg.get("paths", {})

    project_paths = ProjectPaths(
        project_root=project_root,
        config_path=project_root / "config.yaml",
        data_raw=project_root / paths.get("data_raw", "data/raw"),
        data_processed=project_root / paths.get("data_processed", "data/processed"),
        data_graphs=project_root / paths.get("data_graphs", "data/graphs"),
        reports=project_root / paths.get("reports", "reports"),
        figures=project_root / paths.get("figures", "reports/figures"),
    )

    for directory in [
        project_paths.data_raw,
        project_paths.data_processed,
        project_paths.data_graphs,
        project_paths.reports,
        project_paths.figures,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    return project_paths


def print_project_context(paths: ProjectPaths) -> None:
    """
    Print useful project information.
    """
    print("Project root:", paths.project_root)
    print("Config:", paths.config_path)
    print("Raw data:", paths.data_raw)
    print("Processed data:", paths.data_processed)
    print("Graphs:", paths.data_graphs)
    print("Reports:", paths.reports)
    print("Figures:", paths.figures)