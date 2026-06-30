"""Replay a logged Isaac Lab trajectory (.npz) in MuJoCo and save an .mp4.
Maps joints by name since Isaac Lab's joint order differs from the URDF order."""
import argparse

import imageio.v2 as imageio
import mujoco
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--traj", type=str, default="/tmp/chitrak_traj.npz")
parser.add_argument("--model", type=str, default=__file__.replace("replay_to_mp4.py", "chitrak_floating.xml"))
parser.add_argument("--out", type=str, default="/tmp/chitrak_replay.mp4")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--fps", type=int, default=50)
args = parser.parse_args()

data = np.load(args.traj, allow_pickle=True)
joint_names = [str(n) for n in data["joint_names"]]
joint_pos = data["joint_pos"]  # (T, 12)
root_pos = data["root_pos"]  # (T, 3)
root_quat = data["root_quat"]  # (T, 4) wxyz

m = mujoco.MjModel.from_xml_path(args.model)
d = mujoco.MjData(m)

# build name -> qpos index map (skip the freejoint's 7 dof at the front)
qpos_idx = {}
for name in joint_names:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid == -1:
        raise ValueError(f"joint {name} not found in mujoco model")
    qpos_idx[name] = m.jnt_qposadr[jid]

renderer = mujoco.Renderer(m, height=args.height, width=args.width)
cam = mujoco.MjvCamera()
cam.distance = 0.8
cam.azimuth = 120
cam.elevation = -20

frames = []
T = joint_pos.shape[0]
for t in range(T):
    d.qpos[0:3] = root_pos[t]
    d.qpos[3:7] = root_quat[t]
    for i, name in enumerate(joint_names):
        d.qpos[qpos_idx[name]] = joint_pos[t, i]
    mujoco.mj_forward(m, d)
    cam.lookat = root_pos[t]
    renderer.update_scene(d, camera=cam)
    frames.append(renderer.render().copy())

imageio.mimsave(args.out, frames, fps=args.fps)
print(f"[INFO]: Saved {len(frames)} frames to {args.out}")
