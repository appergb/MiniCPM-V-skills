# tests/unit/test_config.py
from minicpm_v_local.config import Config, load

def test_defaults():
    cfg = Config.defaults()
    assert cfg.backend == "auto"
    assert cfg.quant == "4bit"
    assert cfg.idle_timeout == 300
    assert cfg.max_lifetime == 1800
    assert cfg.isolation is False

def test_env_overrides_toml(monkeypatch, tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('backend = "cpu"\nidle_timeout = 60\n')
    monkeypatch.setenv("MINICPM_IDLE_TIMEOUT", "120")
    cfg = load(toml_path=toml, cli_overrides={})
    assert cfg.backend == "cpu"
    assert cfg.idle_timeout == 120

def test_cli_overrides_env(monkeypatch, tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('backend = "cpu"\n')
    monkeypatch.setenv("MINICPM_BACKEND", "cuda")
    cfg = load(toml_path=toml, cli_overrides={"backend": "mlx"})
    assert cfg.backend == "mlx"

def test_missing_toml_returns_defaults(tmp_path):
    cfg = load(toml_path=tmp_path / "nope.toml", cli_overrides={})
    assert cfg.backend == "auto"

def test_video_section_nested(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text('[video]\nscene_threshold = 0.5\nmax_frames = 30\n')
    cfg = load(toml_path=toml, cli_overrides={})
    assert cfg.video.scene_threshold == 0.5
    assert cfg.video.max_frames == 30
