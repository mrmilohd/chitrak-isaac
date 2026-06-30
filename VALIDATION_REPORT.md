# Validation Report — Chitrak Stand-Upright Implementation

Audit performed after implementing the stand-upright-only task conversion and running two
500-iteration training tests. Every claim below was re-verified directly against running code or
logs during this pass — not recalled from earlier conversation — with the exact check shown.

---

## 1. Actuator config — VERIFIED CORRECT

`chitrak_isaac/robots/chitrak.py`'s `DCMotorCfg` (`stiffness=10, damping=0.5, effort_limit=2.5,
saturation_effort=2.5, velocity_limit=8.0`) was cross-checked against the **live runtime tensors**
queried directly from a running env (not just read from source):

```
[INFO]: actuator class: DCMotor
[INFO]: runtime stiffness tensor: [10. 10. 10. 10. 10. 10. 10. 10. 10. 10. 10. 10.]
[INFO]: runtime damping tensor:   [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]
[INFO]: runtime effort_limit:     [2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5]
[INFO]: runtime velocity_limit:   [8. 8. 8. 8. 8. 8. 8. 8. 8. 8. 8. 8.]
[INFO]: runtime saturation_effort: 2.5
```

All 12 joints match exactly. This also confirms `convert_urdf.py`'s baked-in USD default joint
drive gains (`stiffness=100, damping=1.0`, found in `chitrak/.../usd/config.yaml`) are **not**
leaking through — `DCMotorCfg`'s override fully takes effect.

## 2. Reward config — VERIFIED CORRECT, one false alarm resolved

`chitrak_isaac/tasks/velocity/chitrak_rough_env_cfg.py`'s three task-conversion changes, checked
against the actual printed reward table from a live training run:

```
|   0   | track_lin_vel_xy_exp |      1.5 |
...
|   8   | undesired_contacts   |     -1.0 |   <- re-enabled, stock weight (only body_names overridden)
...
|  11   | base_height          |  -1000.0 |   <- new term, confirmed present and correctly weighted
```

- **`undesired_contacts.body_names = ".*_thigh_link"`**: confirmed functionally live, not just
  configured — `Episode_Reward/undesired_contacts` produces genuine nonzero values throughout
  training (`0.0 → -0.074 → -0.0015` etc.), which is only possible if the regex is matching real
  bodies with real contact events. A silently-unmatched regex would show a permanent `0.0000`.
- **`base_height_l2(target_height=0.17)`**: confirmed present in the live reward table at both
  tested weights (-300, then -1000), and its printed `Episode_Reward` values move sensibly with
  actual measured height across training (cross-checked against `root_pos_w[:,2]` directly via the
  per-joint torque/posture scripts — see §4).
- **`rel_standing_envs=1.0`**: this needed real investigation, not just inspection. Querying
  `vel_command_b` immediately after `env.reset()` showed nonzero random values
  (`[0.344, 0.043, -0.234]`), which looked like a bug at first. **Root cause traced to source**:
  `CommandManager.reset()` calls `_resample_command()` (sets random values + the `is_standing_env`
  flag) but does **not** call `_update_command()` (the function that actually zeros the command for
  standing envs) — that only runs inside `compute()`, which fires during `env.step()`, not
  `env.reset()`. Verified by checking again after one `env.step()`: `vel_command_b` was exactly
  `[0,0,0]` for all 8 test envs, and stayed zero across 50 more steps. **Conclusion: the fix is
  correct in practice** — the reward function only ever observes `vel_command_b` during step-time
  reward computation, after `_update_command()` has already run, so training was never exposed to
  the transient pre-override values. This is a timing subtlety worth remembering if re-checked
  naively in the future (checking right after `reset()` will look broken when it isn't).

## 3. Diagnostic/test scripts — audited, two stale-documentation issues fixed

All Python diagnostic scripts re-parsed for syntax validity (`ast.parse`) — all pass. Beyond
syntax, two files had documentation that no longer matched their actual content (a direct result
of being patched multiple times via inline edits during the investigation) and have been
corrected:

- **`pose_sweep_isaaclab.py`**: header docstring described an "IK-based" pose parameterization,
  but the script's actual `FOOT_UNDER_HIP_CANDIDATES` list (and `make_target`/`run_pose` functions)
  use direct forward-kinematics measurement, a different and later methodology. Also had an unused
  `import math` left over from the removed IK approach. Both fixed — docstring now describes the
  actual FK-based method and its history, unused import removed.
- **`mujoco/static_holding_test.py`**: described its zero-penetration bisection fix as if that
  settled the test's reliability, without mentioning the **later finding that it didn't** (an
  uncapped-torque follow-up showed the system never settles even with unlimited torque — a MuJoCo
  contact-stiffness/timestep artifact). Added an explicit `RETRACTED` notice pointing to
  `static_holding_isaaclab.py` as the trustworthy replacement.
- **`best_case_pose_isaaclab.py`**: presented its result as "the most torque-favorable
  configuration" with no mention of the known snap-transient confound (robot spawns at one pose,
  instantly commanded to a very different one). Added a `SUPERSEDED` notice pointing to
  `best_case_pose_v2_isaaclab.py` (which fixes this via `init_state` override so init==commanded).

`replay_to_mp4.py` confirmed robust to the new `torque` field added to `play_log_chitrak.py`'s
output — it only reads `joint_names`/`joint_pos`/`root_pos`/`root_quat`, ignoring extra keys.

**`TRAINING_ANALYSIS.md` updated** (new §11) — its §10 conclusion ("no pose holds 0.17m within
torque budget, confirmed across 5 pose-selection strategies") is **directly contradicted** by the
actual `weight=-1000` training run, which reached 0.174m via an asymmetric multi-leg strategy none
of the manual symmetric pose sweeps tried. This is now documented as a correction, not silently
left stale. See §4 below for the same finding in this report's own terms.

## 4. Training runs — both verified clean, with one quirk

| | Run 1 (`base_height` w=-300) | Run 2 (`base_height` w=-1000) |
|---|---|---|
| Log dir | `2026-06-30_14-45-33` | `2026-06-30_15-03-26` |
| Wall time | 592.06s | 605.86s |
| Checkpoints | 11 files, `model_0.pt`...`model_499.pt`, all load correctly (`iter=499`) | same |
| Local log errors/tracebacks | none | none |
| Final height | 0.126m | **0.174m** |
| Final `base_contact` rate | 5.3% (local) / 4.2% (W&B summary) | 5.8% (local) / 5.3% (W&B summary) |
| W&B run | [vhxtyppy](https://wandb.ai/aadibhatt2007-iit-roorkee/chitrak-locomotion/runs/vhxtyppy) | [asha4xzq](https://wandb.ai/aadibhatt2007-iit-roorkee/chitrak-locomotion/runs/asha4xzq) |

**Quirk found and explained:** both W&B runs show `state: crashed` in the API rather than
`finished`. Investigated directly — `scan_history()` shows 489/491 of the expected ~500 logged
points present, and the final summary metrics match the local log's tail values closely (e.g. Run
2's local final `base_height=-0.2184` vs. W&B summary `-0.2087` — same ballpark, off by a few
iterations' worth of drift). **Conclusion: this is Isaac Sim's known abrupt process teardown**
(same Vulkan-teardown/segfault pattern seen throughout this session on `simulation_app.close()`)
disconnecting the W&B session before a clean `finished` handshake, not a real training failure or
meaningful data loss. Flagging this so it isn't misread as "the run crashed" by anyone checking
the W&B dashboard status field directly — the actual run data is intact and usable.

**Per-joint torque verified for both finals** via `play_log_chitrac.py`'s applied-torque logging
(added during this work, reading `robot.data.applied_torque` — the real post-saturation value,
traced to `DCMotor._clip_effort` in Isaac Lab source):

- Run 1: 8 of 12 joints saturated 95-99.6% of the time — uniform near-maximal effort, height
  essentially unchanged from the original passive-collapse baseline (0.120m).
- Run 2: asymmetric — one leg (`fl`) does almost no work (0.2-0.8 Nm, never saturates) while three
  legs' hip_pitch+knee and one hip_roll sit saturated 63-99% of the time. Height reaches 0.174m,
  geometrically a forward-leaning stretch (front legs extended ~0.087-0.089m knee height, back-right
  crouched at 0.028m).

This is genuinely new information relative to the pre-training static analysis — none of the
hand-designed pose sweeps tried an asymmetric per-leg strategy, only symmetric ones.

## 5. Known open items (not yet resolved, listed for transparency)

- Run 2's solution is not viable on real hardware as-is (3 of 4 legs at ~99% continuous-torque
  duty cycle indefinitely would overheat real motors). Whether a more evenly-distributed solution
  can also reach ~0.17m is untested. `dof_torques_l2` (`-1e-4`, confirmed inert relative to
  `base_height`'s `-1000` via the printed reward table) is the proposed next lever — untested.
- The front/back, left/right torque asymmetry in Run 2 doesn't have a confirmed causal explanation
  (discussed as possibly settling-order/symmetry-breaking during training, not verified).
- `static_holding_test.py` (MuJoCo) remains retracted; no MuJoCo-based test in this codebase should
  be trusted quantitatively for torque/contact magnitude — only `static_holding_isaaclab.py` and
  other direct-PhysX scripts are validated for that purpose. MuJoCo is still trustworthy for the
  *replay/visualization* pipeline (`replay_to_mp4.py`), which doesn't depend on its contact/actuator
  fidelity, only correct forward kinematics for rendering joint angles already computed elsewhere.
