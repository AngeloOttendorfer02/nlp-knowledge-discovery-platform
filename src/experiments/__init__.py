"""Reproducible retrieval experiment runners and report generators."""

from importlib import import_module

__all__ = ["generate_report_assets"]


def __getattr__(name: str):
    if name == "generate_report_assets":
        return import_module(".generate_report_assets", __name__).generate_report_assets
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
