from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def knee_near_ground(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_calf_link"),
) -> torch.Tensor:
    """Geometric proxy for "is a knee dragging on the ground" -- counts how many
    of the selected bodies (default: the 4 calf_link bodies, whose own origin
    coincides with the knee joint's pivot point by URDF/joint convention,
    invariant to the joint's own current angle) have a world-frame height
    below ``threshold``.

    This is NOT a contact-force check. Chitrak's calf_link contains both the
    shin shaft AND the foot-tip as sub-geoms of the SAME rigid body, so a
    body-level contact sensor (isaaclab.envs.mdp.undesired_contacts) cannot
    distinguish "foot tip bearing normal standing load" from "knee/shin
    dragging" -- both register as contact force on the same body. This
    height-based check sidesteps that by looking at the knee pivot's
    position directly, independent of contact/force.

    Threshold derivation (chitrak_isaac/mujoco/chitrak_floating.xml FK sweep,
    865 configs filtered to realistic standing/gait geometry -- foot-drop
    0.10-0.20m matching observed training heights, foot under the hip, sane
    non-inverted knee): knee height above ground never goes below ~0.071m in
    any plausible gait. Observed COLLAPSED legs in actual training (one leg
    idle/dragging) measured 0.0235-0.0382m. Default threshold=0.05m sits in
    the ~3cm gap between these with margin on both sides -- should never
    trigger during genuine standing/walking, should always trigger on a
    genuinely dragging leg.

    Returns the count of bodies below threshold per env (0-4), to be used
    with a negative weight -- i.e. behaves like a per-leg boolean contact
    penalty (matches isaaclab.envs.mdp.undesired_contacts' weight scale of
    -1.0 per triggered body, for consistency).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, num_bodies)
    return torch.sum((heights < threshold).float(), dim=1)
