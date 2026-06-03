"""
document_loader.py — Load scientific documents from multiple sources.

The platform's primary dataset is the arXiv metadata collection (titles,
abstracts, authors, categories, publication dates). This module provides a
small, well-typed loading layer that returns a uniform pandas DataFrame
regardless of the input format (CSV, JSON-lines, or a directory of PDFs).

A lightweight :class:`Document` dataclass is also provided for code paths
that prefer working with objects rather than DataFrame rows.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Document:
    """
    A single scientific document with its core metadata.

    Attributes
    ----------
    doc_id : str
        Unique identifier (e.g. arXiv id or running index).
    title : str
        Paper title.
    abstract : str
        Paper abstract / body text.
    authors : list of str
        Author names.
    categories : list of str
        arXiv subject categories (e.g. "cs.CL").
    date : str
        Publication / update date.
    """

    doc_id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    date: str = ""

    @property
    def text(self) -> str:
        """Concatenate title and abstract into a single searchable string."""
        return f"{self.title}. {self.abstract}".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_authors(value) -> List[str]:
    """Normalize an authors field into a clean list of names."""
    if isinstance(value, list):
        return [str(a).strip() for a in value if str(a).strip()]
    if isinstance(value, str) and value.strip():
        # arXiv authors are typically comma- or 'and'-separated
        parts = value.replace(" and ", ",").split(",")
        return [p.strip() for p in parts if p.strip()]
    return []


def _split_categories(value) -> List[str]:
    """Normalize a categories field into a clean list of category codes."""
    if isinstance(value, list):
        return [str(c).strip() for c in value if str(c).strip()]
    if isinstance(value, str) and value.strip():
        # arXiv categories are space-separated (e.g. "cs.CL cs.AI")
        return [c.strip() for c in value.split() if c.strip()]
    return []


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_arxiv_csv(
    path: str,
    sample_size: Optional[int] = None,
    categories: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load the arXiv metadata from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    sample_size : int, optional
        If given, randomly sample this many rows (after category filtering).
    categories : list of str, optional
        Keep only papers whose category list intersects this set.
    seed : int
        Random seed used when sampling.

    Returns
    -------
    pd.DataFrame
        Columns: doc_id, title, abstract, authors, categories, date.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"arXiv CSV not found: {path}")

    df = pd.read_csv(path)
    df = _standardize_dataframe(df)

    if categories:
        wanted = set(categories)
        mask = df["categories"].apply(lambda cats: bool(wanted.intersection(cats)))
        df = df[mask].reset_index(drop=True)

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    return df


def load_arxiv_jsonl(
    path: str,
    sample_size: Optional[int] = None,
    categories: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load the arXiv metadata from a JSON-lines file (one JSON object per line).

    This is the format of the official arXiv Kaggle dump.

    Parameters
    ----------
    path : str
        Path to the .json / .jsonl file.
    sample_size : int, optional
        If given, randomly sample this many rows (after category filtering).
    categories : list of str, optional
        Keep only papers whose category list intersects this set.
    seed : int
        Random seed used when sampling.

    Returns
    -------
    pd.DataFrame
        Standardized document DataFrame.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"arXiv JSONL not found: {path}")

    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    df = _standardize_dataframe(df)

    if categories:
        wanted = set(categories)
        mask = df["categories"].apply(lambda cats: bool(wanted.intersection(cats)))
        df = df[mask].reset_index(drop=True)

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    return df


def load_pdf(path: str) -> str:
    """
    Extract raw text from a single PDF file.

    Uses pdfplumber, which is already part of the project's PDF-handling stack.

    Parameters
    ----------
    path : str
        Path to the PDF file.

    Returns
    -------
    str
        Concatenated text of all pages.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    import pdfplumber

    pages: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def load_documents_from_dir(directory: str, extension: str = ".pdf") -> pd.DataFrame:
    """
    Load every document with a given extension from a directory of PDFs.

    Parameters
    ----------
    directory : str
        Folder containing the documents.
    extension : str
        File extension to match (default ".pdf").

    Returns
    -------
    pd.DataFrame
        Standardized document DataFrame (title taken from the filename).
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")

    rows: List[Dict] = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(extension.lower()):
            full_path = os.path.join(directory, filename)
            text = load_pdf(full_path) if extension == ".pdf" else _read_text(full_path)
            rows.append(
                {
                    "doc_id": filename,
                    "title": os.path.splitext(filename)[0],
                    "abstract": text,
                    "authors": [],
                    "categories": [],
                    "date": "",
                }
            )

    return pd.DataFrame(rows)


def _read_text(path: str) -> str:
    """Read a plain-text file as UTF-8."""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.read().strip()


# ---------------------------------------------------------------------------
# DataFrame <-> Document conversion
# ---------------------------------------------------------------------------

def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map a raw arXiv DataFrame onto the platform's standard schema.

    Missing columns are filled with sensible defaults so that downstream code
    can always rely on the same set of fields.

    Returns
    -------
    pd.DataFrame
        Columns: doc_id, title, abstract, authors, categories, date.
    """
    df = df.copy()

    # Resolve an id column (arXiv uses "id"); otherwise use the row index
    if "id" in df.columns:
        doc_ids = df["id"].astype(str)
    elif "doc_id" in df.columns:
        doc_ids = df["doc_id"].astype(str)
    else:
        doc_ids = pd.Series([str(i) for i in range(len(df))])

    # Resolve a date column from common arXiv field names
    date_col = next((c for c in ["update_date", "date", "published"] if c in df.columns), None)

    standardized = pd.DataFrame(
        {
            "doc_id": doc_ids,
            "title": df.get("title", pd.Series([""] * len(df))).fillna("").astype(str),
            "abstract": df.get("abstract", pd.Series([""] * len(df))).fillna("").astype(str),
            "authors": df.get("authors", pd.Series([""] * len(df))).apply(_split_authors),
            "categories": df.get("categories", pd.Series([""] * len(df))).apply(_split_categories),
            "date": (df[date_col].fillna("").astype(str) if date_col else pd.Series([""] * len(df))),
        }
    )

    return standardized.reset_index(drop=True)


def dataframe_to_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convert a standardized DataFrame into a list of :class:`Document` objects.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by one of the loaders.

    Returns
    -------
    list of Document
    """
    documents: List[Document] = []
    for _, row in df.iterrows():
        documents.append(
            Document(
                doc_id=str(row["doc_id"]),
                title=str(row["title"]),
                abstract=str(row["abstract"]),
                authors=list(row["authors"]),
                categories=list(row["categories"]),
                date=str(row["date"]),
            )
        )
    return documents
