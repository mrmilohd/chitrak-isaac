"""SUPERSEDED -- see best_case_pose_v2_isaaclab.py instead.

Renders the 'theoretical best case' standing pose -- straight leg, foot
under hip, ~97% of max leg extension (hip_pitch=1.4, knee=2.8) -- found by
the FK sweep to be the most torque-favorable configuration possible.

KNOWN FLAW: the robot spawns at chitrak.py's actual init pose (hip_pitch=0.5,
knee=1.0, z=0.17) and is then immediately commanded to snap to the very
different target pose used here on the first step. The resulting chaotic
bouncing seen in this script's output is confounded by that snap transient
-- it's not a clean test of "can 2.5 Nm hold this pose," it also includes
"can it survive an instantaneous large joint-angle jump." best_case_pose_v2
_isaaclab.py fixes this by overriding init_state so the robot spawns
directly in the target pose (init == commanded, zero snap), which is the
trustworthy version of this test. Kept here for the historical record of
why that fix was needed, not as a script to draw conclusions from."""
import argparse
import os
import sys

import numpy as np

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chitrak_isaac  # noqa: F401, E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=250)
parser.add_argument("--out", type=str, default="/tmp/chitrak_best_case.npz")
parser.add_argument("--hip_pitch", type=float, default=1.4)
parser.add_argument("--knee", type=float, default=2.8)
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
SCALE = env_cfg.actions.joint_pos.scale
name_to_idx = {n: i for i, n in enumerate(joint_names)}

SIGN = {"fr": +1, "br": +1, "fl": -1, "bl": -1}
target = np.zeros(12)
for leg, sign in SIGN.items():
    target[name_to_idx[f"{leg}_hip_pitch_joint"]] = sign * args_cli.hip_pitch
    target[name_to_idx[f"{leg}_knee_joint"]] = -sign * args_cli.knee

action = (target - default_joint_pos) / SCALE
action_t = torch.tensor(action, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
print(f"[INFO]: target pose: hip_pitch=±{args_cli.hip_pitch} knee=∓{args_cli.knee}")

z_log, torque_log = [], []
joint_pos_log, root_pos_log, root_quat_log = [], [], []
for step in range(args_cli.num_steps):
    env.step(action_t)
    z_log.append(robot.data.root_pos_w[0, 2].item())
    torque_log.append(robot.data.applied_torque[0].cpu().numpy().copy())
    joint_pos_log.append(robot.data.joint_pos[0].cpu().numpy().copy())
    root_pos_log.append(robot.data.root_pos_w[0].cpu().numpy().copy())
    root_quat_log.append(robot.data.root_quat_w[0].cpu().numpy().copy())
    if step % 20 == 0:
        print(f"step {step:4d}  z={z_log[-1]:.4f}")

print(f"\nInitial torso z: {z_log[0]:.4f} m")
print(f"Final torso z:   {z_log[-1]:.4f} m")
final_torque = torque_log[-1]
for name, tau in zip(joint_names, final_torque):
    print(f"  {name:22s} {tau:7.3f} Nm")

np.savez(
    args_cli.out,
    z=np.array(z_log), torque=np.array(torque_log), joint_names=np.array(joint_names),
    joint_pos=np.array(joint_pos_log),
    root_pos=np.array(root_pos_log),
    root_quat=np.array(root_quat_log),
    dt=env.unwrapped.step_dt,
)
print(f"\n[INFO]: Saved to {args_cli.out}")

env.close()
simulation_app.close()
