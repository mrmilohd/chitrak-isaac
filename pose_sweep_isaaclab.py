"""Sweep candidate standing poses directly in Isaac Lab's PhysX to find a pose
where the knee doesn't saturate at its 2.5 Nm limit while holding still. One
Isaac Sim launch, many resets -- much cheaper than relaunching per pose.

IMPORTANT: poses are parameterized via the SAME 2-link planar IK used by the
real robot's ik_solver.py (target foot z-depth -> hip_pitch, knee angles),
not by sweeping hip_pitch/knee independently. An earlier attempt swept the
two angles independently and got a nonsensical result (height DECREASING as
knee bend decreased) -- that's because hip_pitch and knee are coupled; moving
one without the other doesn't trace a real "leg extension" path. Using the
actual IK formula ties them together correctly, exactly mirroring how the
real hardware's gait stack derives joint angles from a desired foot position.

Action formula (JointPositionActionCfg, use_default_offset=True):
    target = scale * action + default_joint_pos
so to command an arbitrary target pose without touching chitrak.py, we solve
    action = (target - default_joint_pos) / scale
and step with that constant action long enough to settle.
"""
import argparse
import math
import os
import sys

import numpy as np

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chitrak_isaac  # noqa: F401, E402

parser = argparse.ArgumentParser()
parser.add_argument("--settle_steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from chitrak_isaac.tasks.velocity.chitrak_flat_env_cfg import ChitrakFlatEnvCfg_PLAY  # noqa: E402

env_cfg = ChitrakFlatEnvCfg_PLAY()
env_cfg.scene.num_envs = 1
env_cfg.observations.policy.enable_corruption = False
env_cfg.events.reset_base.params = {
    "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
    "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
}

env = gym.make("Isaac-Velocity-Flat-Chitrak-Play-v0", cfg=env_cfg, render_mode=None)
env.reset()

robot = env.unwrapped.scene["robot"]
joint_names = list(robot.data.joint_names)
default_joint_pos = robot.data.default_joint_pos[0].cpu().numpy().copy()
SCALE = env_cfg.actions.joint_pos.scale  # 0.25, set in chitrak_rough_env_cfg.py
print(f"[INFO]: joint order: {joint_names}")
print(f"[INFO]: default_joint_pos: {default_joint_pos}")
print(f"[INFO]: action scale: {SCALE}")

name_to_idx = {n: i for i, n in enumerate(joint_names)}
knee_idx = [name_to_idx[n] for n in ["fr_knee_joint", "fl_knee_joint", "br_knee_joint", "bl_knee_joint"]]
hip_pitch_idx = [name_to_idx[n] for n in ["fr_hip_pitch_joint", "fl_hip_pitch_joint",
                                            "br_hip_pitch_joint", "bl_hip_pitch_joint"]]

# Candidate (hip_pitch_mag, knee_mag) pairs where direct forward-kinematics
# (chitrak_isaac/mujoco/chitrak_floating.xml) confirmed the foot lands
# approximately under the hip (minimizing horizontal lever arm), at
# increasing standing height. Found by FK sweep, not by inverting IK (which
# has a degenerate near-hip singularity that gave nonsensical results).
# Current chitrak.py pose (0.5, 1.0) is NOT in this list -- FK showed its
# foot sits ~2cm forward of the hip, an avoidable inefficiency.
FOOT_UNDER_HIP_CANDIDATES = [
    (1.4, 2.8), (1.3, 2.6), (1.2, 2.4), (1.0, 2.0), (0.8, 1.6),
]

# sign convention per chitrak.py: fr/br positive hip_pitch & negative knee; fl/bl mirrored
SIGN = {"fr": +1, "br": +1, "fl": -1, "bl": -1}


def make_target(hip_pitch_mag, knee_mag):
    target = np.zeros(12)
    for leg, sign in SIGN.items():
        target[name_to_idx[f"{leg}_hip_pitch_joint"]] = sign * hip_pitch_mag
        target[name_to_idx[f"{leg}_knee_joint"]] = -sign * knee_mag
    return target


def run_pose(hip_pitch_mag, knee_mag):
    target = make_target(hip_pitch_mag, knee_mag)
    action = (target - default_joint_pos) / SCALE
    action_t = torch.tensor(action, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)

    env.reset()
    for _ in range(args_cli.settle_steps):
        env.step(action_t)

    torque = robot.data.applied_torque[0].cpu().numpy()
    actual_pos = robot.data.joint_pos[0].cpu().numpy()
    z = robot.data.root_pos_w[0, 2].item()

    max_knee_torque = np.max(np.abs(torque[knee_idx]))
    max_hip_pitch_torque = np.max(np.abs(torque[hip_pitch_idx]))
    knee_pos_err = np.max(np.abs(target[knee_idx] - actual_pos[knee_idx]))
    saturated = max_knee_torque >= 2.49
    return z, max_knee_torque, max_hip_pitch_torque, knee_pos_err, saturated


print(f"\n{'hip_pitch':>10} {'knee':>6} {'height_z':>9} {'max_knee_tau':>13} {'max_hip_tau':>12} {'knee_pos_err':>13} {'saturated':>10}")
results = []
for hip_pitch_mag, knee_mag in FOOT_UNDER_HIP_CANDIDATES:
    z, knee_tau, hip_tau, knee_err, sat = run_pose(hip_pitch_mag, knee_mag)
    results.append((hip_pitch_mag, knee_mag, z, knee_tau, hip_tau, knee_err, sat))
    flag = "  <-- SAT" if sat else ("  ok" if knee_err < 0.02 else "  partial")
    print(f"{hip_pitch_mag:10.2f} {knee_mag:6.2f} {z:9.4f} {knee_tau:13.3f} {hip_tau:12.3f} {knee_err:13.4f}{flag}")

print("\n[INFO]: Feasible poses (knee reaches within 0.02 rad of target, not saturated):")
for hip_pitch_mag, knee_mag, z, knee_tau, hip_tau, knee_err, sat in results:
    if knee_err < 0.02 and not sat:
        print(f"  hip_pitch={hip_pitch_mag:.2f} knee={knee_mag:.2f}  height={z:.4f}  "
              f"knee_tau={knee_tau:.3f}  hip_tau={hip_tau:.3f}")

env.close()
simulation_app.close()
