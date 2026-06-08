from pathlib import Path


def test_pipeline_entrypoint_exists():
    assert Path("src/pipeline/run_pipeline.py").exists()


def test_pipeline_imports():
    from src.pipeline.run_pipeline import load_config, ensure_directories

    assert callable(load_config)
    assert callable(ensure_directories)