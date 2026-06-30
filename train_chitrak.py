"""Wrapper that registers Chitrak gym environments then runs Isaac Lab's train.py."""
import sys
import os
import runpy

# ensure chitrak_isaac is importable — add the workspace root (parent of this package)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chitrak_isaac  # noqa: F401, E402 — triggers gym.register() for Chitrak envs

# delegate to Isaac Lab's standard train.py
_train = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "IsaacLab",
    "scripts", "reinforcement_learning", "rsl_rl", "train.py",
))

# runpy doesn't add the script dir to sys.path — do it manually so cli_args is found
sys.path.insert(0, os.path.dirname(_train))

runpy.run_path(_train, run_name="__main__")
