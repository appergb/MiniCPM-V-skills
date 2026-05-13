# tests/unit/test_paths.py
from minicpm_v_local import paths

def test_config_path_under_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert paths.config_dir() == tmp_path / ".config" / "minicpm-v-local"

def test_state_path():
    assert paths.state_file().name == "state.json"
    assert "minicpm-v-local" in str(paths.state_file())

def test_cache_dir_per_backend():
    assert "mlx" in str(paths.cache_dir("mlx"))
    assert "cpu" in str(paths.cache_dir("cpu"))

def test_lock_files_under_run():
    assert paths.cli_lock().parent == paths.run_dir()
    assert paths.download_lock().parent == paths.run_dir()
