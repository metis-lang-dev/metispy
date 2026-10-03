import pytest


@pytest.fixture(autouse=True)
def _no_user_config(monkeypatch):
    """Hermetic runs: the developer's metis dot file (llm endpoint,
    ollama host/model) must never leak into test behavior. A
    set-but-empty METIS_CONFIG pins the config search to nothing."""
    monkeypatch.setenv("METIS_CONFIG", "")
    from metis.loop import config
    config.load(reload=True)
    yield
    config.load(reload=True)
