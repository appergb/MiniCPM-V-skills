"""Smoke test: package importable, version exposed."""
import minicpm_v_local


def test_version_exposed():
    assert minicpm_v_local.__version__ == "0.1.0"


def test_cli_entrypoint_callable():
    from minicpm_v_local.cli import main
    assert callable(main)
