from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from chitrak_isaac.robots.chitrak import CHITRAK_CFG


@configclass
class ChitrakRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # robot
        self.scene.robot = CHITRAK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # action scale — same as Go1
        self.actions.joint_pos.scale = 0.25

        # task: stand upright only, no locomotion. Force every env's velocity
        # command to zero (rather than removing the tracking reward terms) —
        # track_lin_vel_xy_exp / track_ang_vel_z_exp then become pure
        # "minimize actual velocity" rewards, since the target is always 0.
        self.commands.base_velocity.rel_standing_envs = 1.0

        # events
        self.events.push_robot = None
        self.events.base_com = None
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_link"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_calf_link"
        self.rewards.feet_air_time.weight = 0.01
        # anti-collapse signal: penalize thigh-on-ground contact (the leg has
        # buckled). Previously disabled (copied verbatim from Go1's template,
        # where it's fine since Go1 doesn't tend to collapse) -- re-enabled
        # and remapped to Chitrak's actual link names, since without this a
        # collapsed crouch was reward-neutral and never terminated the
        # episode (see TRAINING_ANALYSIS.md section 5).
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh_link"
        self.rewards.dof_torques_l2.weight = -1.0e-4
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7
        # height-maintenance: symmetric L2 kernel around the standing height
        # (chitrak.py's init pose, 0.17m) -- penalizes BOTH sinking below and
        # jumping above the target equally, so there's no incentive to hop;
        # the only optimum is to sit at the target height.
        # History: weight=-300 settled at h=0.126m with nearly every joint
        # saturated at the real 2.5 Nm limit. weight=-1000 (same 2.5 Nm
        # limit) reached h=0.174m via an asymmetric leg strategy. Reverted to
        # -300 here specifically to pair with a temporary effort_limit bump
        # to Go1's spec (23.7 Nm, see chitrak.py) -- isolates whether the
        # original weight=-300 failure was a torque-budget problem (fixed by
        # more torque) or a reward-strength problem (would need -1000
        # regardless of torque).
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-300.0,
            params={"target_height": 0.17},
        )

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
