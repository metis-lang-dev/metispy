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


@pytest.fixture
def _oracle_required():
    """METIS_ORACLE_REQUIRED=1 and no wayplan ll2pg oracle: the host-only
    gate (marker `oracle`) fails instead of skipping (tests/oracle.py)."""
    from oracle import ORACLE
    pytest.fail(f"METIS_ORACLE_REQUIRED=1 but the wayplan ll2pg oracle is not at {ORACLE} (set METIS_ORACLE_PATH)")
