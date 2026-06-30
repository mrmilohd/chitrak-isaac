import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

CHITRAK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # IMPORTANT: usd_merged/, not usd/. The original usd/chitrak.usd was converted with
        # merge_fixed_joints=False, which left the URDF's empty `<link name="base_link"/>`
        # (no inertial, meant to be a massless reference frame -- all real mass is on the
        # fixed-jointed torso_link) as a SEPARATE rigid body. PhysX can't have a body with
        # undefined mass, so it silently defaulted base_link to 1.0 kg -- a phantom ~77% mass
        # overestimate (2.3045 kg simulated vs. the real 1.3045 kg design) that was present in
        # every test and training run before this was caught. Re-converted with
        # `--merge-joints` (merge_fixed_joints=True), which merges torso_link's mass/inertia
        # into base_link (keeping the root's name) instead of creating a separate body for it --
        # verified via direct `root_physx_view.get_masses()` query to total exactly 1.3045 kg,
        # matching the URDF's per-link masses summed directly. See CLAUDE.md and
        # chitrak_isaac/VALIDATION_REPORT.md for the full investigation.
        usd_path="/teamspace/studios/this_studio/chitrak/chitrak_description/usd/chitrak.usd",  # TEMP revert for playback compat, see below
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.17),
        joint_pos={
            ".*_hip_roll_joint": 0.0,
            "fr_hip_pitch_joint": 0.5,   # axis Y negated, range [0, 3.14]
            "br_hip_pitch_joint": 0.5,
            "fl_hip_pitch_joint": -0.5,  # axis Y positive, range [-3.14, 0]
            "bl_hip_pitch_joint": -0.5,
            "fr_knee_joint": -1.0,
            "br_knee_joint": -1.0,
            "fl_knee_joint": 1.0,
            "bl_knee_joint": 1.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            # TEMP revert for playback compat -- model_1498.pt was trained under 23.7 Nm + old USD
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=8.0,
            stiffness={".*": 10.0},
            damping={".*": 0.5},
            friction=0.0,
        ),
    },
)
