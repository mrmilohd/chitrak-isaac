"""Wrapper that registers Chitrak gym environments then runs Isaac Lab's play.py."""
import sys
import os
import runpy

# ensure chitrak_isaac is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chitrak_isaac  # noqa: F401, E402 — triggers gym.register() for Chitrak envs

_play = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "IsaacLab",
    "scripts", "reinforcement_learning", "rsl_rl", "play.py",
))

sys.path.insert(0, os.path.dirname(_play))

runpy.run_path(_play, run_name="__main__")
