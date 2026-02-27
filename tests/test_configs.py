"""Smoke tests for training configuration dataclasses.

Heavy ML dependencies (unsloth, bitsandbytes, wandb) and the environment-actor
imports triggered by ``src.evaluation._constants`` are replaced with
lightweight mocks *before* the script modules are loaded.  This makes the
config tests fast and self-contained — no GPU or large model weights required.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Pre-patch heavy ML deps so script-level imports succeed without hardware.
# sys.modules.setdefault() leaves real packages in place if already imported
# (e.g. when running inside the affine conda environment).
#
# bitsandbytes requires special treatment: transformers uses
# importlib.util.find_spec() to detect it, which reads __spec__ from the
# entry already in sys.modules.  A plain MagicMock auto-creates __spec__ as
# another MagicMock, causing a ValueError.  Setting __spec__ = None signals
# "package not found" and lets transformers continue gracefully.
# ---------------------------------------------------------------------------
_MOCK_NAMES = ["unsloth", "wandb"]
for _name in _MOCK_NAMES:
    sys.modules.setdefault(_name, MagicMock())

if "bitsandbytes" not in sys.modules:
    _bnb_mock = MagicMock()
    _bnb_mock.__spec__ = None   # prevent ValueError in importlib.util.find_spec
    sys.modules["bitsandbytes"] = _bnb_mock

ROOT = Path(__file__).parent.parent


def _load_script(rel_path: str, module_name: str):
    """Load a script file as a Python module via importlib."""
    spec = importlib.util.spec_from_file_location(module_name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    # Make src.evaluation a mock *for this load* so _constants.py is not
    # executed (it imports AffineFoundation environment actors).
    _sentinel = object()
    _prev = sys.modules.get("src.evaluation", _sentinel)
    sys.modules["src.evaluation"] = MagicMock()
    try:
        spec.loader.exec_module(mod)
    finally:
        if _prev is _sentinel:
            sys.modules.pop("src.evaluation", None)
        else:
            sys.modules["src.evaluation"] = _prev
    return mod


# ---------------------------------------------------------------------------
# Load script modules once per session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tig_trainer_mod():
    return _load_script("scripts/rl/tig_trainer.py", "tig_trainer")


@pytest.fixture(scope="session")
def tig_trainer_v2_mod():
    return _load_script("scripts/rl/tig_trainer_v2.py", "tig_trainer_v2")


@pytest.fixture(scope="session")
def dapo_mod():
    return _load_script("scripts/rl/dapo.py", "dapo")


@pytest.fixture(scope="session")
def sft_mod():
    return _load_script("scripts/sft/sft_reasoning_trainer.py", "sft_reasoning_trainer")


# ---------------------------------------------------------------------------
# TiGConfig (tig_trainer.py)
# ---------------------------------------------------------------------------
class TestTiGConfig:
    def test_instantiates_with_no_args(self, tig_trainer_mod):
        cfg = tig_trainer_mod.TiGConfig()
        assert cfg is not None

    def test_field_types(self, tig_trainer_mod):
        cfg = tig_trainer_mod.TiGConfig()
        assert isinstance(cfg.model_name, str)
        assert isinstance(cfg.lora_r, int)
        assert isinstance(cfg.learning_rate, float)
        assert isinstance(cfg.num_epochs, int)
        assert isinstance(cfg.use_wandb, bool)

    def test_keyword_override(self, tig_trainer_mod):
        cfg = tig_trainer_mod.TiGConfig(model_name="Qwen/Qwen3-14B", lora_r=32)
        assert cfg.model_name == "Qwen/Qwen3-14B"
        assert cfg.lora_r == 32

    def test_mutable_default_independence(self, tig_trainer_mod):
        """lora_target_modules default (None) must be independent per instance."""
        a = tig_trainer_mod.TiGConfig()
        b = tig_trainer_mod.TiGConfig()
        assert a.lora_target_modules is b.lora_target_modules  # both None is fine


# ---------------------------------------------------------------------------
# TiGConfig (tig_trainer_v2.py)
# ---------------------------------------------------------------------------
class TestTiGConfigV2:
    def test_instantiates_with_no_args(self, tig_trainer_v2_mod):
        cfg = tig_trainer_v2_mod.TiGConfig()
        assert cfg is not None

    def test_v2_has_no_batch_size(self, tig_trainer_v2_mod):
        """v2 removes batch_size — it must not exist on the config."""
        cfg = tig_trainer_v2_mod.TiGConfig()
        assert not hasattr(cfg, "batch_size")

    def test_field_types(self, tig_trainer_v2_mod):
        cfg = tig_trainer_v2_mod.TiGConfig()
        assert isinstance(cfg.num_generations, int)
        assert isinstance(cfg.temperature, float)
        assert isinstance(cfg.clip_epsilon, float)


# ---------------------------------------------------------------------------
# DapoConfig (dapo.py)
# ---------------------------------------------------------------------------
class TestDapoConfig:
    def test_instantiates_with_no_args(self, dapo_mod):
        cfg = dapo_mod.DapoConfig()
        assert cfg is not None

    def test_field_types(self, dapo_mod):
        cfg = dapo_mod.DapoConfig()
        assert isinstance(cfg.model_name, str)
        assert isinstance(cfg.lora_r, int)
        assert isinstance(cfg.learning_rate, float)
        assert isinstance(cfg.clip_epsilon, float)
        assert isinstance(cfg.use_wandb, bool)

    def test_wandb_config_mutable_default(self, dapo_mod):
        """Each DapoConfig instance must receive its own wandb_config dict."""
        a = dapo_mod.DapoConfig()
        b = dapo_mod.DapoConfig()
        assert a.wandb_config is not b.wandb_config

    def test_wandb_config_has_expected_keys(self, dapo_mod):
        cfg = dapo_mod.DapoConfig()
        assert "dataset" in cfg.wandb_config
        assert "note" in cfg.wandb_config


# ---------------------------------------------------------------------------
# SFTReasoningConfig (sft_reasoning_trainer.py)
# ---------------------------------------------------------------------------
class TestSFTReasoningConfig:
    def test_instantiates_with_no_args(self, sft_mod):
        cfg = sft_mod.SFTReasoningConfig()
        assert cfg is not None

    def test_field_types(self, sft_mod):
        cfg = sft_mod.SFTReasoningConfig()
        assert isinstance(cfg.model_name, str)
        assert isinstance(cfg.lora_r, int)
        assert isinstance(cfg.learning_rate, float)
        assert isinstance(cfg.num_epochs, int)
        assert isinstance(cfg.use_wandb, bool)
        assert isinstance(cfg.seed, int)

    def test_keyword_override(self, sft_mod):
        cfg = sft_mod.SFTReasoningConfig(model_name="Qwen/Qwen3-7B", num_epochs=5)
        assert cfg.model_name == "Qwen/Qwen3-7B"
        assert cfg.num_epochs == 5

