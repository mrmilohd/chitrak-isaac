"""Run a trained Chitrak checkpoint headless (no rendering) and log the joint/base
trajectory to .npz for offline replay in MuJoCo. Avoids Isaac Sim's RTX renderer
entirely -- works around the T4's lack of RT cores."""

import argparse
import os
import sys

import numpy as np

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chitrak_isaac  # noqa: F401, E402 -- triggers gym.register() for Chitrak envs

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "IsaacLab",
    "scripts", "reinforcement_learning", "rsl_rl",
))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Play a Chitrak checkpoint and log trajectory.")
parser.add_argument("--num_steps", type=int, default=300, help="Number of steps to log.")
parser.add_argument("--out", type=str, default="/tmp/chitrak_traj.npz", help="Output npz path.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402, F401

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import importlib.metadata as metadata  # noqa: E402

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = (
        cli_args_checkpoint_path()
        if args_cli.checkpoint
        else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    )
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.data.joint_names)
    print(f"[INFO]: Logging joints in order: {joint_names}")

    joint_pos_log = []
    root_pos_log = []
    root_quat_log = []
    torque_log = []

    obs = env.get_observations()
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

        joint_pos_log.append(robot.data.joint_pos[0].cpu().numpy().copy())
        root_pos_log.append(robot.data.root_pos_w[0].cpu().numpy().copy())
        root_quat_log.append(robot.data.root_quat_w[0].cpu().numpy().copy())
        torque_log.append(robot.data.applied_torque[0].cpu().numpy().copy())

    torque_arr = np.array(torque_log)
    print(f"\n[INFO]: Torque stats over {len(torque_log)} steps (post-saturation, real applied torque):")
    for j, name in enumerate(joint_names):
        peak = np.max(np.abs(torque_arr[:, j]))
        mean = np.mean(np.abs(torque_arr[:, j]))
        frac_saturated = np.mean(np.abs(torque_arr[:, j]) >= 2.45)
        flag = "  <-- frequently saturated" if frac_saturated > 0.1 else ""
        print(f"  {name:22s} peak={peak:6.3f} Nm  mean={mean:6.3f} Nm  "
              f"%time>=2.45Nm={frac_saturated*100:5.1f}%{flag}")

    np.savez(
        args_cli.out,
        joint_names=np.array(joint_names),
        joint_pos=np.array(joint_pos_log),
        root_pos=np.array(root_pos_log),
        root_quat=np.array(root_quat_log),
        torque=torque_arr,
        dt=env.unwrapped.step_dt,
    )
    print(f"[INFO]: Saved {len(joint_pos_log)} steps to {args_cli.out}")

    env.close()


def cli_args_checkpoint_path():
    from isaaclab.utils.assets import retrieve_file_path
    return retrieve_file_path(args_cli.checkpoint)


if __name__ == "__main__":
    main()
    simulation_app.close()
