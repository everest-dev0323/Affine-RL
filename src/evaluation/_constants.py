"""Shared constants and environment actor mappings for evaluation modules.

This module is responsible for:
- Adding affinetes sub-environment packages to ``sys.path`` so they can be
  imported without individual editable installs.
- Importing and exposing every Actor class used across the evaluation package.
- Defining the mapping from environment name → Actor class and all AgentGym
  port / round defaults.
"""

import importlib
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Submodule path setup
#
# affinetes sub-environments are not always installed as top-level packages,
# so we add their directories to sys.path here, once, at import time.
# ---------------------------------------------------------------------------

_submodule_base = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../AffineFoundation/affinetes")
)

sys.path.insert(0, os.path.join(_submodule_base, "affinetes/templates"))
sys.path.insert(0, os.path.join(_submodule_base, "environments/trace"))
sys.path.insert(0, os.path.join(_submodule_base, "environments/openspiel"))
sys.path.insert(0, os.path.join(_submodule_base, "environments/primeintellect/lgc-v2"))
sys.path.insert(0, os.path.join(_submodule_base, "environments/arc_gen"))

# ---------------------------------------------------------------------------
# Actor imports
#
# lgc-v2 uses a hyphenated directory name that prevents normal dot-import
# syntax, so we load it via importlib.
# ---------------------------------------------------------------------------

_lgc2_module = importlib.import_module(
    "AffineFoundation.affinetes.environments.primeintellect.lgc-v2.env"
)
LGC2Actor = _lgc2_module.Actor

from AffineFoundation.affinetes.environments.trace.env import Actor as PrintActor      # noqa: E402
from AffineFoundation.affinetes.environments.openspiel.env import Actor as GameActor   # noqa: E402
from AffineFoundation.affinetes.environments.arc_gen.env import Actor as ArcGenActor   # noqa: E402

# ---------------------------------------------------------------------------
# ACTOR_MAPPING
#
# Maps the lowercase environment name used in evaluation calls to the
# corresponding Actor class.  Add new environments here.
# ---------------------------------------------------------------------------

ACTOR_MAPPING = {
    "lgc-v2":  LGC2Actor,
    "print":   PrintActor,
    "game":    GameActor,
    "arc-gen": ArcGenActor,
}

# ---------------------------------------------------------------------------
# AgentGym port mapping  (values come from .env, with sensible defaults)
# ---------------------------------------------------------------------------

AGENTGYM_PORTS: dict = {
    "webshop":   int(os.getenv("AGENTGYM_WEBSHOP_PORT",   36001)),
    "alfworld":  int(os.getenv("AGENTGYM_ALFWORLD_PORT",  36002)),
    "sciworld":  int(os.getenv("AGENTGYM_SCIWORLD_PORT",  36003)),
    "textcraft": int(os.getenv("AGENTGYM_TEXTCRAFT_PORT", 36004)),
    "babyai":    int(os.getenv("AGENTGYM_BABYAI_PORT",    36005)),
}

# Default maximum interaction rounds per AgentGym environment
AGENTGYM_MAX_ROUNDS: dict = {
    "webshop":   10,
    "alfworld":  30,
    "babyai":    20,
    "sciworld":  20,
    "textcraft": 20,
}

# Environments that require a ``task_type`` parameter during evaluation
ENVS_WITH_TASK_TYPE: frozenset = frozenset({"abd-v2", "ded-v2", "sat", "abd", "ded"})

