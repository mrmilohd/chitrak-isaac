"""Static holding-torque test using Isaac Lab's ACTUAL PhysX physics directly
(no RL policy, no MuJoCo) -- settles the cross-simulator fidelity question by
testing in the exact engine training uses. Steps the Chitrak env with a
constant zero action (= hold the init pose, the actuator's default offset)
and logs torso height + per-joint applied torque to see whether gravity alone
makes it collapse."""
import argparse
import os
import sys

import numpy as np

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chitrak_isaac  # noqa: F401, E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--out", type=str, default="/tmp/chitrak_static_holding.npz")
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
print(f"[INFO]: joint order: {joint_names}")

actuator = robot.actuators["legs"]
print(f"[INFO]: actuator class: {type(actuator).__name__}")
print(f"[INFO]: runtime stiffness tensor: {actuator.stiffness[0].cpu().numpy()}")
print(f"[INFO]: runtime damping tensor:   {actuator.damping[0].cpu().numpy()}")
print(f"[INFO]: runtime effort_limit:     {actuator.effort_limit[0].cpu().numpy()}")
print(f"[INFO]: runtime velocity_limit:   {actuator.velocity_limit[0].cpu().numpy()}")
if hasattr(actuator, "_saturation_effort"):
    print(f"[INFO]: runtime saturation_effort: {actuator._saturation_effort}")

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
        print(f"step {step:4d}  z={z_log[-1]:.4f}  "
              f"torque={np.round(torque_log[-1], 3)}")

z_log = np.array(z_log)
torque_log = np.array(torque_log)

print(f"\nInitial torso z: {z_log[0]:.4f} m")
print(f"Final torso z:   {z_log[-1]:.4f} m")
print(f"Min torso z:     {np.min(z_log):.4f} m")

steady = torque_log[-50:]
print(f"\nSteady-state (last 50 steps) peak |applied torque| per joint:")
for j, name in enumerate(joint_names):
    peak = np.max(np.abs(steady[:, j]))
    mean = np.mean(np.abs(steady[:, j]))
    flag = "  <-- AT/NEAR 2.5 Nm LIMIT" if peak >= 2.4 else ""
    print(f"  {name:22s} peak={peak:6.3f} Nm  mean={mean:6.3f} Nm{flag}")

print(f"\nSettled (actual, not commanded) joint positions:")
settled_joint_pos = robot.data.joint_pos[0].cpu().numpy()
for name, pos in zip(joint_names, settled_joint_pos):
    print(f"  {name:22s} {pos:7.4f} rad")
print(f"\nSettled root pos: {robot.data.root_pos_w[0].cpu().numpy()}")
print(f"Settled root quat (wxyz): {robot.data.root_quat_w[0].cpu().numpy()}")
print(f"\nSettled joint velocities:")
settled_joint_vel = robot.data.joint_vel[0].cpu().numpy()
for name, vel in zip(joint_names, settled_joint_vel):
    print(f"  {name:22s} {vel:8.5f} rad/s")
print(f"Settled root lin/ang vel: {robot.data.root_lin_vel_w[0].cpu().numpy()}  {robot.data.root_ang_vel_w[0].cpu().numpy()}")

np.savez(
    args_cli.out,
    z=z_log, torque=torque_log, joint_names=np.array(joint_names),
    settled_joint_pos=settled_joint_pos,
    joint_pos=np.array(joint_pos_log),
    root_pos=np.array(root_pos_log),
    root_quat=np.array(root_quat_log),
    dt=env.unwrapped.step_dt,
)
print(f"\n[INFO]: Saved to {args_cli.out}")

env.close()
simulation_app.close()
