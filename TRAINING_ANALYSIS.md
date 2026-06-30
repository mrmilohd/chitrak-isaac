# Chitrak Flat-Terrain Locomotion — Current Implementation & Diagnosis

Run analyzed: `IsaacLab/logs/rsl_rl/chitrak_flat/2026-06-29_17-43-48/` (model_550.pt, model_800.pt, model_1100.pt)

**Symptom:** the torso starts at the spawn height (0.168 m) and collapses to ~0.078 m within
0.4 s (20 steps) of every rollout, then stays pancaked there. By iteration 1100 the policy never
recovers; at iteration 550 it still oscillates (0.07–0.16 m) with some XY drift. The robot never
learns to stand, let alone walk. This document lays out the exact config as currently written,
flags which parts are stock Isaac Lab / copied from Go1 / Chitrak-specific, and identifies the
most likely root causes with supporting numbers.

---

## 0. Chitrak vs. Unitree Go1 — differences at a glance

The Chitrak config (`chitrak_isaac/`) was built by copying the structure of Isaac Lab's stock
Go1 config (`isaaclab_tasks/.../config/go1/`) almost line-for-line. The physical robot and the
actuator model behind it, however, are very different — several config values that are correct
for Go1 do not automatically transfer. This table consolidates every difference found; the
sections referenced below give the detailed reasoning.

| Aspect | Chitrak | Unitree Go1 (Isaac Lab `UNITREE_GO1_CFG`) | Note |
|---|---|---|---|
| Total mass | ~1.5 kg (`CLAUDE.md`) | ~12 kg (Unitree spec sheet; not re-verified in this session) | Go1 is ~8x heavier |
| DOF | 12 (hip_roll, hip_pitch, knee × 4 legs) | 12 (hip, thigh, calf × 4 legs) | same topology |
| Standing height (sim) | 0.17 m (`pos=(0,0,0.17)`) | 0.4 m (`pos=(0,0,0.4)`) | Go1 ~2.4x taller |
| Init joint pose | per-leg mirrored `hip_pitch=±0.5`, `knee=∓1.0` (derived from Chitrak's own URDF limits) | `hip=±0.1`, `F.*_thigh=0.8`, `R.*_thigh=1.0`, `calf=-1.5` | both Chitrak-specific and Go1-specific, not interchangeable |
| **Actuator model class** | `DCMotorCfg` — ideal PD + linear torque/speed saturation curve, **no learned dynamics** | `ActuatorNetMLPCfg` — torque output is a **learned neural net** trained on real Go1 hardware sysid data (`unitree_go1.pt`), input = (pos error, vel) history | fundamentally different actuator simulation — Go1's actuator net captures real motor nonlinearities (friction, backlash, current limiting dynamics) that a simple PD model does not. This is *not* a copy/paste issue, just a fact: there's no equivalent learned actuator net for Chitrak's motors (would require real hardware sysid data Chitrak doesn't have), so DCMotor is the right available choice — but it means Chitrak's sim dynamics are inherently more idealized/optimistic than Go1's. |
| `effort_limit` / `saturation_effort` | 2.5 Nm / 2.5 Nm | 23.7 Nm / 23.7 Nm | Go1 motors are ~9.5x stronger in absolute torque |
| `velocity_limit` | 8.0 rad/s | 30.0 rad/s | Go1 motors ~3.75x faster |
| `stiffness` (explicit PD gain) | 10.0 Nm/rad (all joints) | N/A — baked into the learned actuator net, no explicit stiffness param | not a directly comparable number; Chitrak's DCMotor needs this value to be hand-tuned, Go1's net effectively has whatever "stiffness" the real hardware exhibited when the net was trained |
| `action_scale` | 0.25 (copied from Go1's `rough_env_cfg.py` comment "reduce action scale") | 0.25 | **same number, very different consequence at the knee specifically** — see §3/§4b: holding the init pose with zero action already uses 90–100% of the knee's 2.5 Nm budget, so `stiffness × scale = 2.5 Nm` leaves ~0% margin there. Hip joints use only 0.02–0.6 Nm even at zero action, so the same arithmetic doesn't bite for them. A1/Go2 (`25 × 0.25 = 6.25 Nm` against 23.5–33.5 Nm limits) have 70-80% headroom robot-wide. Go1 itself isn't a fair stiffness comparison (learned net, no explicit stiffness), but the *blind copy of the scale value* without re-deriving it against Chitrak's actuator is the issue. |
| `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` weights | 1.5 / 0.75 | 1.5 / 0.75 | copied verbatim, no Chitrak-specific re-derivation |
| `dof_torques_l2` weight | -1.0e-4 | -2.0e-4 | Chitrak penalizes torque usage half as much per Nm — but Chitrak's torques are also ~9.5x smaller in absolute Nm, so the *effective* penalty per %-of-max-torque is actually much smaller for Chitrak than for Go1 (unclear if intentional) |
| `flat_orientation_l2` weight (flat terrain) | -5.0 | -2.5 | Chitrak penalizes tilt 2x harder than Go1; no derivation found for why |
| `undesired_contacts` (thigh-on-ground penalty) | disabled (`None`) | disabled (`None`) | copied verbatim — see §5 for why this is more consequential for Chitrak |
| `feet_air_time` body name | `.*_calf_link` | `.*_foot` | necessary remap — Chitrak's URDF has no separate foot link, the calf link tip is the ground-contact point |
| `base_contact` termination body name | `torso_link` | `trunk` | necessary remap, same logic, different link name |
| Episode length / `decimation` / `sim.dt` | 20 s / 4 / 0.005 s (all stock, unmodified) | 20 s / 4 / 0.005 s | identical, inherited from `LocomotionVelocityRoughEnvCfg` |
| PPO hidden dims / `entropy_coef` | [128,128,128] / 0.01 | [128,128,128] / not explicitly verified in this session for Go1's own agent cfg | both small networks appropriate for low-dim 48-obs/12-act task |

**Bottom line:** topology and high-level config structure are copied faithfully and correctly
remapped (link names, joint names, episode settings). The numbers that silently don't transfer
are the ones tied to **absolute actuator strength** (`action_scale`, and by extension the
stiffness/effort_limit headroom margin in §3) and the **actuator model itself** (idealized PD vs.
learned net) — both real, structural differences between a from-scratch DCMotor robot and Go1's
hardware-fitted actuator net, not Chitrak-specific authoring mistakes.

---

## 1. Actuator model — `chitrak_isaac/robots/chitrak.py`

```python
actuators={
    "legs": DCMotorCfg(
        joint_names_expr=[".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
        effort_limit=2.5,       # Nm, continuous torque
        saturation_effort=2.5,  # Nm, stall torque — EQUAL to effort_limit
        velocity_limit=8.0,     # rad/s
        stiffness={".*": 10.0}, # Nm/rad — single value for all 12 joints
        damping={".*": 0.5},    # Nm·s/rad
        friction=0.0,
    ),
}
```

`DCMotor` (`isaaclab/actuators/actuator_pd.py`) computes:

```
torque_cmd = stiffness * (target_pos - current_pos) - damping * current_vel
torque_applied = clip(torque_cmd, tau_min(vel), tau_max(vel))
```

Because `saturation_effort == effort_limit` here, the torque–speed curve is flat: ±2.5 Nm at any
velocity up to 8 rad/s, then it rolls off to 0. There is no "boost" region above continuous
torque — this matches the real hardware spec in `CLAUDE.md` (effort=2.5 Nm is the *real* motor
limit, not a placeholder), so the limit itself is correct for sim-to-real fidelity.

**These numbers are real / intentional. The problem is how they interact with action scale (§3).**

### Comparison with stock Isaac Lab quadrupeds using the same `DCMotorCfg` actuator class

| Robot | stiffness (Nm/rad) | damping | effort_limit (Nm) | velocity_limit | action_scale |
|---|---|---|---|---|---|
| **Chitrak** | 10.0 | 0.5 | **2.5** | 8.0 | 0.25 |
| Unitree A1 (`unitree.py`) | 25.0 | 0.5 | 33.5 | 21.0 | 0.25 (per A1 cfg) |
| Unitree Go2 (`unitree.py`) | 25.0 | 0.5 | 23.5 | 30.0 | 0.25 |

(Unitree Go1's actual `UNITREE_GO1_CFG` in Isaac Lab uses a learned `ActuatorNetMLPCfg`, not
`DCMotorCfg` — it isn't a fair stiffness/effort_limit comparison. A1 and Go2 are the right
reference points since they share Chitrak's exact actuator class.)

---

## 2. Robot config & init pose — `chitrak_isaac/robots/chitrak.py`

```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.17),
    joint_pos={
        ".*_hip_roll_joint": 0.0,
        "fr_hip_pitch_joint": 0.5, "br_hip_pitch_joint": 0.5,    # axis -Y, range [0, 3.14]
        "fl_hip_pitch_joint": -0.5, "bl_hip_pitch_joint": -0.5,  # axis +Y, range [-3.14, 0]
        "fr_knee_joint": -1.0, "br_knee_joint": -1.0,
        "fl_knee_joint": 1.0, "bl_knee_joint": 1.0,
    },
),
soft_joint_pos_limit_factor=0.9,
```
`enabled_self_collisions=False` (self-collision disabled — Chitrak's legs are close enough to
the torso, with `enabled_self_collisions=True` it might generate spurious contact forces; this
is a reasonable choice, not a likely cause).

This is **not directly copied from Go1** — it's derived from Chitrak's own URDF joint ranges
(documented in `CLAUDE.md`), correctly per-leg-mirrored.

---

## 3. Action scale — interacts with §1, this is the core finding

`chitrak_rough_env_cfg.py`:
```python
self.actions.joint_pos.scale = 0.25   # comment: "same as Go1"
```

Stock `ActionsCfg` (`velocity_env_cfg.py`): `target = scale * action + default_joint_pos`,
`action ∈ [-1, 1]` (post `tanh`/policy output, roughly bounded). So the maximum per-step position
command excursion from the default pose is **±0.25 rad**.

**Torque required to fully use that range:** `stiffness × scale = 10.0 × 0.25 = 2.5 Nm`
— exactly `effort_limit`. At the full action range, the motor is already saturated by the
position-tracking term alone, with **zero torque budget left** for absorbing contact impacts or
correcting disturbances. §4b shows this matters most at the knee specifically: holding the init
pose with *zero* action already uses 90–100% of the knee's torque budget (2.27–2.50 Nm of 2.5 Nm),
so any nonzero knee action from an exploring policy pushes it straight into saturation. Hip joints
are not in this situation — they use only 0.02–0.6 Nm even at zero action, so the same `scale=0.25`
headroom math doesn't bite there in practice.

Contrast with A1/Go2: `stiffness × scale = 25.0 × 0.25 = 6.25 Nm` against an effort_limit of
23.5–33.5 Nm → only ~20–27% of the torque budget is used at full action excursion, leaving
**70–80% headroom**. Chitrak's knee joints have **~0% headroom** by construction (hip joints are
fine).

This was carried over verbatim from Go1's `rough_env_cfg.py` (`self.actions.joint_pos.scale =
0.25  # reduce action scale`) without re-deriving it against Chitrak's much weaker
`effort_limit=2.5` (vs. Go1/A1/Go2's 23.5–33.5 Nm).

**Suggested fix to test:** apply a per-joint stiffness override rather than a single global value
— raise knee `stiffness`/`effort_limit` specifically (hip joints don't need it, see §4b) — or
lower `action_scale` for the knee joints specifically via a per-joint action scale if Isaac Lab's
`JointPositionActionCfg` supports it, so `stiffness × scale` for the knee drops well under its
2.5 Nm cap.

---

## 4. Static holding-torque sanity check

**This section was tested twice. The first attempt (MuJoCo) was methodologically flawed and its
headline conclusion is retracted below. The second attempt (direct Isaac Lab/PhysX test) is the
trustworthy result and supersedes it.**

### 4a. First attempt (MuJoCo) — retracted

Built a floating-base MJCF (`chitrak_isaac/mujoco/chitrak_floating.xml`) with 12 `<position>`
actuators nominally matching Isaac Lab's `DCMotorCfg` (`kp=10`, `kv=0.5`, `forcerange=±2.5`), held
the init pose with constant `ctrl`, and initially found all 12 joints saturating and the torso
collapsing to ~0 m. **On closer inspection this result is not reliable:**
- The original 0.17 m spawn height was ~1.3 cm too low for this exact pose, causing foot/ground
  *penetration* at t=0 — the contact solver's corrective impulse in the first timestep loaded
  every joint (including hip_roll, which has no reason to carry load in a symmetric stance),
  contaminating the reading.
- After fixing that (auto-bisecting to the exact zero-penetration height), the result was
  unchanged — still full collapse. But a follow-up check via `mj_inverse` showed the "required
  torque" is *extremely* sensitive to the assumed ground penetration depth (0.05 Nm at zero
  penetration vs. >20 Nm at 2 cm penetration) — MuJoCo's penalty-based contact model means
  "required torque" isn't a single well-defined number without first finding the model's own
  true equilibrium penetration depth.
- Attempting to find that equilibrium by running with **unlimited** torque (no 2.5 Nm cap) and
  letting it settle: it never reached a calm state. After 10 s of sim time it still had nonzero
  velocity and wildly asymmetric joint torques (up to ±22 Nm) across what should be a
  mirror-symmetric stance — i.e. it was still bouncing/tipping, not settling. That's a strong
  signal of a MuJoCo contact-stiffness/timestep instability artifact in this particular
  reproduction (default contact stiffness vs. `kp=10`/`timestep=0.005s` apparently don't pair
  well), not a physically meaningful result.

**Conclusion: this MuJoCo setup was not a faithful enough reproduction of Isaac Lab's PhysX
contact/actuator behavior to trust quantitatively.** Retracted. (Scripts left in
`chitrak_isaac/mujoco/static_holding_test.py` for reference, with this caveat.)

### 4b. Second attempt — direct test in Isaac Lab's actual PhysX (authoritative)

Rather than fight cross-simulator fidelity, tested directly in the real engine training uses:
`chitrak_isaac/static_holding_isaaclab.py` creates the actual `Isaac-Velocity-Flat-Chitrak-Play-v0`
env (1 instance, randomization disabled, base reset pinned to the exact spawn pose) and steps it
with a **constant zero action** for 300 steps — `JointPositionActionCfg` with `use_default_offset
=True` means `action=0` commands exactly the init pose, i.e. "just hold still," the easiest
possible task, with zero RL policy involved. Reads `robot.data.applied_torque` directly (the
actual torque PhysX's `DCMotor` model applied after clipping).

```
Initial torso z: 0.1700 m
Final torso z:   0.1202 m   <- settles here and stays, calm steady-state (not bouncing)
Min torso z:     0.1202 m

Steady-state (last 50 steps) peak |applied torque| per joint:
  bl_hip_roll_joint      0.121 Nm     fl_hip_roll_joint   0.022 Nm
  br_hip_roll_joint      0.329 Nm     fr_hip_roll_joint   0.121 Nm
  bl_hip_pitch_joint     0.434 Nm     fl_hip_pitch_joint  0.607 Nm
  br_hip_pitch_joint     0.163 Nm     fr_hip_pitch_joint  0.222 Nm
  bl_knee_joint          2.272 Nm     fl_knee_joint       2.500 Nm  <-- SATURATED
  br_knee_joint          2.348 Nm     fr_knee_joint       2.500 Nm  <-- SATURATED
```

**This is a much more specific and moderate finding than §4a's retracted result:**
- The robot does **not** catastrophically collapse with zero action — it sags ~5 cm from the
  commanded 0.17 m to a new, genuinely stable equilibrium at 0.12 m, and holds there calmly
  (peak == mean torque per joint — no oscillation).
- **Hip joints have large headroom**: 0.02–0.6 Nm used out of a 2.5 Nm budget (75–99% headroom).
  §3's "zero headroom from `stiffness × action_scale`" argument is real as an action-budget
  calculation, but it overstates the problem at the hip — in practice the hip joints aren't anywhere
  near their torque limit even holding still.
- **Knee joints are the actual bottleneck**: 2 of 4 are fully saturated at exactly 2.500 Nm, the
  other 2 are at 2.27–2.35 Nm (within ~10% of the limit). This is a real, narrow margin problem
  specifically at the knee, not a robot-wide "can't stand at all" problem.
- This 5 cm sag-to-a-new-equilibrium, by itself, is **not** enough to explain the full collapse to
  0.078 m seen in the actual trained-policy rollouts (§ symptom, top of doc) — that requires an
  additional mechanism. The leading candidate is §5: with `undesired_contacts` disabled, once the
  policy's own actions push the marginal knees past this 0.12 m baseline sag (which a learning,
  exploring policy applying nonzero actions absolutely will, per §3's zero-headroom math), there's
  no reward signal pulling it back, so it free-falls further into the unpenalized crouch.

**Revised picture:** the knee joints have a real, narrow, but not totally-incapacitating torque
margin problem (~successfully holds 70% of the way to the target height, not 0%). The missing
anti-collapse reward (§5) is what turns a moderate "knees sag under static load" issue into the
total collapse actually observed during training.

---

## 5. Rewards — `chitrak_rough_env_cfg.py` + `chitrak_flat_env_cfg.py`

| Term | Stock weight (`velocity_env_cfg.py`) | Go1 (`rough_env_cfg.py`/`flat_env_cfg.py`) | Chitrak | Source of Chitrak value |
|---|---|---|---|---|
| `track_lin_vel_xy_exp` | 1.0 | 1.5 | **1.5** | copied from Go1 |
| `track_ang_vel_z_exp` | 0.5 | 0.75 | **0.75** | copied from Go1 |
| `lin_vel_z_l2` | -2.0 | -2.0 (inherited) | -2.0 | stock, unchanged |
| `ang_vel_xy_l2` | -0.05 | -0.05 (inherited) | -0.05 | stock, unchanged |
| `dof_torques_l2` | -1.0e-5 | -2.0e-4 | **-1.0e-4** | between stock and Go1, no derivation found |
| `dof_acc_l2` | -2.5e-7 | -2.5e-7 | -2.5e-7 | copied from Go1 (same as stock) |
| `action_rate_l2` | -0.01 | -0.01 (inherited) | -0.01 | stock, unchanged |
| `feet_air_time` | 0.125, body=`.*FOOT` | 0.01 (rough) / 0.25 (flat), body=`.*_foot` | 0.01 (rough) / **0.25** (flat), body=`.*_calf_link` | copied from Go1; body remapped since Chitrak has no separate foot link |
| `undesired_contacts` (thigh contact penalty) | -1.0, body=`.*THIGH` | **None (disabled)** | **None (disabled)** | copied from Go1 verbatim |
| `flat_orientation_l2` | 0.0 (disabled) | -2.5 (flat only) | **-5.0** (flat only) | Chitrak-specific, 2x Go1's value, no derivation found |
| `dof_pos_limits` | 0.0 (disabled) | 0.0 (inherited, disabled) | 0.0 (disabled) | stock, unchanged |

**Key finding: `undesired_contacts` is disabled, exactly as in Go1's stock config — but this
matters far more for Chitrak.** This reward normally penalizes thigh-link contact with the
ground (a proxy for "the leg has buckled / robot is dragging on its knees"). Disabling it is fine
for Go1 because Go1's actuators are strong enough (relative to its mass) that it doesn't tend to
collapse into a stable crouch in the first place. Chitrak, with zero torque headroom (§3), *can*
buckle, and with this penalty disabled **there is no reward signal at all telling it that
collapsing is bad** — as long as the `torso_link` itself doesn't register >1 N contact force
(the `base_contact` termination threshold), the episode keeps running.

This is consistent with what we observed directly: in the model_1100 trajectory log, torso height
drops to 0.078 m at step ~20 and **never resets back up to 0.168 m for the rest of the 300-step
rollout** — i.e. `base_contact` never triggered. The robot found a stable, non-terminating,
not-actively-penalized resting crouch (thighs/calves on the ground, torso just barely clear of
the 1 N threshold) and has no gradient pushing it back to standing. There is also no
`base_height_l2`-style reward in the stock `RewardsCfg` at all (Isaac Lab's default template
relies entirely on `flat_orientation_l2` + `track_lin_vel_xy_exp` + termination to keep the robot
upright) — for Go1-scale robots that's apparently enough; for Chitrak it evidently is not.

**Suggested fix to test:** re-enable `undesired_contacts` remapped to Chitrak's actual link names
(`.*_thigh_link` and/or `.*_calf_link` contact, excluding the foot tip), or add an explicit
height-maintenance reward term.

---

## 6. Events / domain randomization — `chitrak_rough_env_cfg.py` + `chitrak_flat_env_cfg.py`

| Event | Stock | Chitrak (rough) | Chitrak (flat) | Why |
|---|---|---|---|---|
| `physics_material` (startup) | randomize robot friction | unchanged | **disabled** | T4 GPU deadlock workaround (see `CLAUDE.md`) |
| `add_base_mass` (startup) | ±5 kg on `base` | remapped to `torso_link` | **disabled** | T4 GPU deadlock workaround |
| `base_com` (startup) | ±0.05/0.05/0.01 m | **disabled** | disabled (inherited) | copied from Go1 verbatim |
| `base_external_force_torque` (reset) | 0 force/torque (no-op by default) | remapped to `torso_link` | **disabled** | T4 GPU deadlock workaround |
| `reset_base` (reset) | ±0.5 m xy, ±π yaw, ±0.5 vel | same pose range, **velocity forced to 0** | inherited | copied from Go1 verbatim |
| `reset_robot_joints` (reset) | scale (0.5, 1.5) — randomizes joints ±50% | **scale (1.0, 1.0) — no randomization** | inherited | copied from Go1 verbatim |
| `push_robot` (interval) | random xy velocity push every 10–15s | **disabled** | disabled (inherited) | copied from Go1 verbatim |

So at flat-training time, the *only* active domain randomization is the reset-base position/yaw
range; joint positions reset to the exact init pose every episode, and there are no mass/friction/
push perturbations at all. This makes the training easier, not harder — it rules out
domain-randomization-induced instability as a cause. The collapse is a deterministic consequence
of the dynamics + reward shape, not noise.

---

## 7. Observations — `velocity_env_cfg.py` (stock, unmodified by Chitrak)

48-dim flat-terrain policy observation (no deviation from stock Go1/Anymal-C template):

| Term | Dim | Noise |
|---|---|---|
| `base_lin_vel` | 3 | ±0.1 |
| `base_ang_vel` | 3 | ±0.2 |
| `projected_gravity` | 3 | ±0.05 |
| `velocity_commands` | 3 | none |
| `joint_pos_rel` | 12 | ±0.01 |
| `joint_vel_rel` | 12 | ±1.5 |
| `last_action` | 12 | none |
| `height_scan` | — | **removed** (flat terrain, set to `None`) |

Nothing Chitrak-specific here — this is verified correct by construction (stock code path, only
the height scanner is removed for flat terrain, exactly like Go1's flat config).

---

## 8. Terminations — stock logic, Chitrak only remaps body name

```python
time_out = DoneTerm(func=mdp.time_out, time_out=True)                      # stock, unchanged
base_contact = DoneTerm(func=mdp.illegal_contact,
    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0})
```
`body_names` remapped from stock's `"base"` to Chitrak's actual link name `"torso_link"` — correct,
necessary remap, not a bug. `threshold=1.0` N is stock, unchanged. As discussed in §5, this
threshold is the reason the collapsed-crouch state never terminates — the torso link can rest at
0.078 m without registering 1 N of contact force as long as the legs (not the torso) are doing the
actual ground contact.

---

## 9. PPO / RSL-RL agent — `tasks/velocity/agents/rsl_rl_ppo_cfg.py`

| Param | Chitrak | Go1 (Isaac Lab stock, Anymal-style) |
|---|---|---|
| `actor/critic_hidden_dims` | [128, 128, 128] | [128, 128, 128] (flat override) |
| `num_steps_per_env` | 24 | 24 |
| `learning_rate` | 1e-3, adaptive KL=0.01 | 1e-3, adaptive KL=0.01 |
| `entropy_coef` | 0.01 | 0.005 (stock Anymal-C value; Go1 not explicitly checked) |
| `clip_param` | 0.2 | 0.2 |
| `gamma` / `lam` | 0.99 / 0.95 | 0.99 / 0.95 |
| `max_iterations` (cfg default, overridden via CLI to 1500 for the actual run) | 500 | 300 (Anymal-C flat stock) |
| `num_envs` (actual run, via CLI) | 1024 | matches Go1 stock launch command in `CLAUDE.md` |

Nothing structurally wrong here — standard PPO hyperparameters consistent with the rest of Isaac
Lab's zoo. `entropy_coef=0.01` (2x the Anymal-C stock 0.005) slightly favors more exploration,
which is not a likely cause of collapse-and-stay (if anything it should make a stuck local optimum
*less* sticky, not more).

---

## 10. Ranked hypotheses for the collapse

1. **(CONFIRMED, §4b — direct Isaac Lab/PhysX test) Knee joints have a narrow, real torque
   margin problem.** With zero RL action (just holding the init pose), the robot settles calmly
   ~5 cm lower than commanded (0.17 m → 0.12 m); 2 of 4 knee joints are fully saturated at exactly
   2.5 Nm, the other 2 are within ~10% of saturation. Hip joints have 75–99% headroom and are not
   the bottleneck. This alone causes a moderate sag, not the full collapse seen in training.
2. **(§5, now the more important factor) No anti-collapse reward signal.** `undesired_contacts`
   is disabled (copied from Go1's template) and there's no height-maintenance reward. Once a
   learning/exploring policy's nonzero actions push the already-marginal knees past the 0.12 m
   static-sag baseline (§3's zero-action-headroom math applies once the policy outputs anything
   other than exactly zero), there is no reward signal pulling it back up, and no early
   termination (torso doesn't register >1 N until it's nearly on the ground, §8) — so a transient
   knee-torque deficit during exploration can compound into the full 0.078 m collapse seen in
   model_1100, with ~980 reward-neutral crouching steps per episode reinforcing "stay down."
3. **(§3) Zero action-scale headroom on top of an already-marginal joint.** `stiffness(10) ×
   action_scale(0.25) = 2.5 Nm = effort_limit` means any nonzero knee action immediately saturates
   a joint that's already at 90–100% load just holding still (§4b). This compounds #1, it doesn't
   independently cause it.

**Recommended next steps, in order:**
1. Target the knee joints specifically — they're the only joints with a real margin problem (§4b):
   - **Reduce knee bend in the standing pose** (currently `∓1.0 rad`) toward something closer to
     upright, shrinking gravity's moment arm at the knee specifically (hip joints already have
     plenty of headroom, no change needed there).
   - Or **raise knee `stiffness`/`effort_limit`** specifically (per-joint `DCMotorCfg` already
     supports a dict, e.g. `stiffness={".*_knee_joint": 20.0, ".*": 10.0}`) if real hardware
     permits more than 2.5 Nm stall torque at the knee.
   - Re-run `chitrak_isaac/static_holding_isaaclab.py` after each change until the knee no longer
     saturates and the torso holds within ~1 cm of 0.17 m indefinitely.
2. Re-enable `undesired_contacts` remapped to `.*_thigh_link`/`.*_calf_link` (excluding the
   foot-tip geometry) so any future transient knee-torque deficit is actively penalized instead of
   reward-neutral — this is what currently lets a moderate sag turn into a permanent collapse.
3. Retrain a short run (e.g. 300 iterations) and re-check the trajectory log/MuJoCo replay before
   committing to a full 1500-iteration run.
