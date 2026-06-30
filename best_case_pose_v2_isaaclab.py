"""Render the 'theoretical best case' standing pose with INIT = COMMANDED --
the robot spawns directly in the target straight-leg pose (no snap-transient
from a different starting configuration), then holds it with zero action.
This isolates "can 2.5 Nm hold this pose once already there" from "can it
survive snapping into this pose from somewhere else" (which confounded the
first attempt, best_case_pose_isaaclab.py)."""
import argparse
import os
import sys

import numpy as np

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chitrak_isaac  # noqa: F401, E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=250)
parser.add_argument("--out", type=str, default="/tmp/chitrak_best_case_v2.npz")
parser.add_argument("--hip_pitch", type=float, default=1.4)
parser.add_argument("--knee", type=float, default=2.8)
parser.add_argument("--spawn_height", type=float, default=0.302)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from chitrak_isaac.tasks.velocity.chitrak_flat_env_cfg import ChitrakFlatEnvCfg_PLAY  # noqa: E402

SIGN = {"fr": +1, "br": +1, "fl": -1, "bl": -1}
new_joint_pos = {".*_hip_roll_joint": 0.0}
for leg, sign in SIGN.items():
    new_joint_pos[f"{leg}_hip_pitch_joint"] = sign * args_cli.hip_pitch
    new_joint_pos[f"{leg}_knee_joint"] = -sign * args_cli.knee

env_cfg = ChitrakFlatEnvCfg_PLAY()
env_cfg.scene.num_envs = 1
env_cfg.observations.policy.enable_corruption = False
env_cfg.events.reset_base.params = {
    "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
    "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
}
# override init_state so the robot SPAWNS in the target pose -- init == commanded
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args_cli.spawn_height)
env_cfg.scene.robot.init_state.joint_pos = new_joint_pos
# reset_robot_joints uses position_range scaling on default_joint_pos -- keep
# it at (1.0, 1.0) (exact, no randomization), inherited from chitrak_rough_env_cfg.py

print(f"[INFO]: overridden init_state.pos = {env_cfg.scene.robot.init_state.pos}")
print(f"[INFO]: overridden init_state.joint_pos = {new_joint_pos}")

env = gym.make("Isaac-Velocity-Flat-Chitrak-Play-v0", cfg=env_cfg, render_mode=None)
env.reset()

robot = env.unwrapped.scene["robot"]
joint_names = list(robot.data.joint_names)
print(f"[INFO]: default_joint_pos after override: {robot.data.default_joint_pos[0].cpu().numpy()}")
print(f"[INFO]: actual spawned joint_pos: {robot.data.joint_pos[0].cpu().numpy()}")
print(f"[INFO]: actual spawned root z: {robot.data.root_pos_w[0, 2].item():.4f}")

num_actions = env.unwrapped.action_manager.total_action_dim
zero_action = torch.zeros((1, num_actions), device=env.unwrapped.device)

z_log, torque_log = [], []
joint_pos_log, root_pos_log, root_quat_log = [], [], []
for step in range(args_cli.num_steps):
    env.step(zero_action)
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
print("\nFinal-step applied torque:")
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
