"""RETRACTED -- see ../static_holding_isaaclab.py instead (TRAINING_ANALYSIS.md
section 4a/4b for the full writeup).

Independent of any RL policy: command the robot to hold a fixed standing pose
forever (ctrl = pose, i.e. always-zero residual action) using PD gains and a
torque limit nominally matching Isaac Lab's DCMotorCfg (stiffness=10,
damping=0.5, effort_limit=saturation_effort=2.5 Nm), and see whether gravity
alone makes the legs buckle.

The zero-penetration bisection below (auto-computing the initial torso height
per pose, rather than guessing) was a real fix for an initial-impact artifact
-- but it was NOT sufficient to make this script trustworthy. A follow-up
check (running with the 2.5 Nm cap removed entirely) found the system never
settles to a calm state even with unlimited torque -- after 10s of sim time
it still had nonzero velocity and wildly asymmetric joint torques across a
mirror-symmetric stance. That's a MuJoCo contact-stiffness/timestep
instability artifact in this particular reproduction, not real physics.
Conclusion: this MuJoCo setup is not a faithful enough reproduction of Isaac
Lab's PhysX contact/actuator behavior to trust quantitatively. Retracted in
favor of testing directly in Isaac Lab's actual PhysX engine
(static_holding_isaaclab.py), which removes the cross-simulator fidelity
question entirely. Kept here for the historical record only.
"""
import sys

import mujoco
import numpy as np

JOINT_NAMES = [
    "fr_hip_roll_joint", "fr_hip_pitch_joint", "fr_knee_joint",
    "fl_hip_roll_joint", "fl_hip_pitch_joint", "fl_knee_joint",
    "br_hip_roll_joint", "br_hip_pitch_joint", "br_knee_joint",
    "bl_hip_roll_joint", "bl_hip_pitch_joint", "bl_knee_joint",
]

# Two pose options:
#  "chitrak_cfg" = the pose currently in chitrak_isaac/robots/chitrak.py (hip_pitch=0.5, knee=1.0)
#  "ik_default"  = the EXACT pose produced by the real, hand-derived ik_solver.py's default foot
#                  target (x=0, y=0, z=-0.15), passed through joint_angles_translator.py's mapping.
#                  This is the original ROS2 stack's actual neutral/standing pose, not a guess.
POSE_NAME = sys.argv[1] if len(sys.argv) > 1 else "chitrak_cfg"

POSES = {
    "chitrak_cfg": {
        "fr_hip_roll_joint": 0.0, "fl_hip_roll_joint": 0.0, "br_hip_roll_joint": 0.0, "bl_hip_roll_joint": 0.0,
        "fr_hip_pitch_joint": 0.5, "br_hip_pitch_joint": 0.5,
        "fl_hip_pitch_joint": -0.5, "bl_hip_pitch_joint": -0.5,
        "fr_knee_joint": -1.0, "br_knee_joint": -1.0,
        "fl_knee_joint": 1.0, "bl_knee_joint": 1.0,
    },
    "ik_default": {
        "fr_hip_roll_joint": 0.0, "fl_hip_roll_joint": 0.0, "br_hip_roll_joint": 0.0, "bl_hip_roll_joint": 0.0,
        "fr_hip_pitch_joint": 0.4482, "br_hip_pitch_joint": 0.4482,
        "fl_hip_pitch_joint": -0.4482, "bl_hip_pitch_joint": -0.4482,
        "fr_knee_joint": -1.1226, "br_knee_joint": -1.1226,
        "fl_knee_joint": 1.1226, "bl_knee_joint": 1.1226,
    },
}
INIT_POSE = POSES[POSE_NAME]
print(f"[INFO]: Testing pose '{POSE_NAME}': {INIT_POSE}")

m = mujoco.MjModel.from_xml_path(
    "/teamspace/studios/this_studio/chitrak_isaac/mujoco/chitrak_floating.xml"
)
d = mujoco.MjData(m)

joint_qpos_idx = {}
actuator_idx = {}
for name in JOINT_NAMES:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_qpos_idx[name] = m.jnt_qposadr[jid]
    actuator_idx[name] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def min_contact_dist_at_height(z):
    mujoco.mj_resetData(m, d)
    d.qpos[2] = z
    for name, val in INIT_POSE.items():
        d.qpos[joint_qpos_idx[name]] = val
    mujoco.mj_forward(m, d)
    if d.ncon == 0:
        return None  # no contacts yet -> floating above ground
    return min(c.dist for c in d.contact)


# bisect for the zero-penetration height for this exact pose
lo, hi = 0.05, 0.30
for _ in range(60):
    mid = (lo + hi) / 2
    dist = min_contact_dist_at_height(mid)
    if dist is None or dist > 0:
        hi = mid
    else:
        lo = mid
init_height = hi + 0.001  # tiny clearance margin, avoids contact ambiguity at t=0
print(f"[INFO]: Auto-computed zero-penetration height for this pose: {hi:.4f} m "
      f"(using {init_height:.4f} m as start)")

mujoco.mj_resetData(m, d)
d.qpos[2] = init_height
for name in JOINT_NAMES:
    d.qpos[joint_qpos_idx[name]] = INIT_POSE[name]
    d.ctrl[actuator_idx[name]] = INIT_POSE[name]
mujoco.mj_forward(m, d)

DURATION_S = 3.0
n_steps = int(DURATION_S / m.opt.timestep)

z_log, t_log = [], []
torque_log = []  # per-joint, not just max-abs

for step in range(n_steps):
    mujoco.mj_step(m, d)
    if step % 10 == 0:
        t_log.append(step * m.opt.timestep)
        z_log.append(d.qpos[2])
        torque_log.append([d.actuator_force[actuator_idx[n]] for n in JOINT_NAMES])

z_log = np.array(z_log)
t_log = np.array(t_log)
torque_log = np.array(torque_log)  # (T, 12)
saturated_log = np.sum(np.abs(torque_log) >= 2.49, axis=1)

print(f"\n{'t (s)':>6} {'torso z (m)':>12} {'max |torque| (Nm)':>18} {'# joints saturated':>20}")
for i in range(0, len(t_log), max(1, len(t_log) // 20)):
    print(f"{t_log[i]:6.2f} {z_log[i]:12.4f} {np.max(np.abs(torque_log[i])):18.3f} {saturated_log[i]:20d}")

print(f"\nInitial torso z: {z_log[0]:.4f} m")
print(f"Final torso z:   {z_log[-1]:.4f} m")
print(f"Min torso z:     {np.min(z_log):.4f} m")

# steady-state = last 0.5s, well past any initial transient
steady_mask = t_log >= (DURATION_S - 0.5)
steady_torque = torque_log[steady_mask]
print(f"\nSteady-state (last 0.5s) peak |torque| per joint:")
for j, name in enumerate(JOINT_NAMES):
    peak = np.max(np.abs(steady_torque[:, j]))
    mean = np.mean(np.abs(steady_torque[:, j]))
    flag = "  <-- SATURATED" if peak >= 2.49 else ""
    print(f"  {name:22s} peak={peak:6.3f} Nm  mean={mean:6.3f} Nm{flag}")

print(f"\nSteady-state torso z range: [{z_log[steady_mask].min():.4f}, {z_log[steady_mask].max():.4f}] m"
      f"  (started at {init_height:.4f} m)")
