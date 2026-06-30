#!/usr/bin/env bash
# Roll out a Chitrak checkpoint headless in Isaac Lab, log its trajectory,
# then render it to .mp4 via MuJoCo (avoids Isaac Sim's RTX renderer, which
# hangs on the T4's lack of RT cores).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK="Isaac-Velocity-Flat-Chitrak-Play-v0"
NUM_STEPS=300
NUM_ENVS=1
OUT_DIR="$SCRIPT_DIR/replay_mp4"
FPS=50

usage() {
  echo "Usage: $0 --checkpoint <path/to/model_XXXX.pt> [--task NAME] [--num_steps N] [--num_envs N] [--out_dir DIR] [--fps N]"
  exit 1
}

CHECKPOINT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2;;
    --task) TASK="$2"; shift 2;;
    --num_steps) NUM_STEPS="$2"; shift 2;;
    --num_envs) NUM_ENVS="$2"; shift 2;;
    --out_dir) OUT_DIR="$2"; shift 2;;
    --fps) FPS="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

[[ -z "$CHECKPOINT" ]] && usage
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT"; exit 1; }

ISAACLAB_DIR="$(cd "$SCRIPT_DIR/../IsaacLab" && pwd)"
PY311=/usr/lib/python-build-standalone/3.11/bin/python3.11

mkdir -p "$OUT_DIR"
TAG="$(basename "$CHECKPOINT" .pt)"
NPZ="$OUT_DIR/${TAG}_traj.npz"
MP4="$OUT_DIR/${TAG}_replay.mp4"

echo "[1/2] Rolling out '$TAG' headless (no rendering) -> $NPZ"
cd "$ISAACLAB_DIR"
CONDA_PREFIX="" VIRTUAL_ENV=/usr/lib/python-build-standalone/3.11 OMNI_KIT_ACCEPT_EULA=Y \
  ./isaaclab.sh -p "$SCRIPT_DIR/play_log_chitrak.py" \
  --task="$TASK" --headless \
  --checkpoint "$CHECKPOINT" --num_envs "$NUM_ENVS" --num_steps "$NUM_STEPS" --out "$NPZ"

echo "[2/2] Rendering MuJoCo replay -> $MP4"
cd "$SCRIPT_DIR/mujoco"
MUJOCO_GL=egl "$PY311" replay_to_mp4.py --traj "$NPZ" --out "$MP4" --fps "$FPS"

echo "Done: $MP4"
