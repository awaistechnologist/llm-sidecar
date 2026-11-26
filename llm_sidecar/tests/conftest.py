import pytest


@pytest.fixture(autouse=True)
def _private_data_dir(monkeypatch, tmp_path):
    """Keep account readings (claude-windows.json, claude-api-spend.json)
    out of the real data directory, in both directions: a test must not
    write the user's, and the user's must not decide a test. A real reading
    saying the subscription is out would make every claude/ test fail."""
    from llm_sidecar import ledger
    monkeypatch.setattr(ledger, "DATA_DIR", tmp_path / "data")
    # Same for Anthropic credentials: a real `ant auth login` on this
    # machine must not switch claude-api/ on in a test, or reach the API.
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "anthropic"))
    # Settings are now saved when changed through the daemon, and the CLI
    # also tells a running daemon. Neither may touch the real ones.
    from llm_sidecar import cli, config
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config" / "config.json")
    monkeypatch.setattr(cli, "_tell_daemon", lambda *a, **k: False)
