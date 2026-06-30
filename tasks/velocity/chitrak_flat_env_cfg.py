from isaaclab.utils import configclass

from .chitrak_rough_env_cfg import ChitrakRoughEnvCfg


@configclass
class ChitrakFlatEnvCfg(ChitrakRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no curriculum
        self.curriculum.terrain_levels = None

        # disable events that deadlock on headless T4 before first sim step
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None

        # flat terrain reward overrides
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.feet_air_time.weight = 0.25


@configclass
class ChitrakFlatEnvCfg_PLAY(ChitrakFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
