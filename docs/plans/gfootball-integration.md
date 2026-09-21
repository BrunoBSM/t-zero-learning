# Plan: Google Research Football (GFootball) as a t-zero environment

Branch: `feature/gfootball-env` (off `main` @ 2b0d805). Status markers: `[ ]` todo,
`[x]` done, `[~]` in progress, `[!]` blocked/decision needed.

## Goal and constraints

Make GFootball academy scenarios runnable through the standard t-zero path
(`python train.py --config <cfg>`) so it can serve as the **final class
project** for the RL course. Scope of *this* plan is the environment port only.
A discrete-action PPO baseline is wanted but is a **separate** follow-up
(not part of this branch).

Hard constraints from the course:

- Students run **mostly on their own machines** (Linux/macOS/Windows mix).
  Past classes lost most of the project time getting the env to run. The
  deliverable therefore includes a **ready-to-use Dockerfile / Apptainer def**
  so a student is training within minutes, not days.
- Existing DQN (`algorithms/dqn.py`, `num_envs=1`, `discrete_control` stack)
  must work on it unchanged: flat `Box` observation, `Discrete` action.

Environment facts (verified 2026-09-21):

- github.com/google-research/football **archived Aug 2026, read-only**. Last
  release **v2.10.2 (2021)**. `setup.py` pins `gym<=0.21.0`; no gymnasium,
  no numpy 2, no Python ≥3.11 testing upstream.
- C++ engine (GameplayFootball; SDL2, Boost) compiled at `pip install`.
  Linux apt deps: `git cmake build-essential libgl1-mesa-dev libsdl2-dev
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev
  libdirectfb-dev libst-dev mesa-utils xvfb`.
- Throughput ~100–500 steps/s per process without rendering. Academy scenarios
  ≈ ≤400 steps/episode; full 11v11 = 3000 steps (out of scope for a course).
- Target stack we must be compatible with (conda env `cleanrl`):
  Python 3.10.18, numpy 2.2.6, gymnasium 1.2.0, torch 2.9.1.

## Phase 0 — Feasibility spike (scratch env, nothing committed to t-zero)

Purpose: find out *how* gfootball can be installed next to our stack before
designing around it. Work in a throwaway conda env (`gfootball-spike`), never
in `cleanrl`.

- [x] 0.1 Native build on Python 3.10: apt deps present? `pip install
      gfootball --no-deps` compiles? Record wall time.
- [x] 0.2 Dependency shim: install `gym` at a version that merely *imports*
      (`gym.Env`, `gym.spaces.Box/Discrete/MultiDiscrete`) — candidate
      `gym==0.26.2` — and gfootball's other runtime deps (`absl-py`,
      `psutil`, `scipy`?) individually. Confirm
      `from gfootball.env import create_environment` works.
- [x] 0.3 numpy 2.x compatibility: create `academy_empty_goal_close`,
      `representation="simple115_v2"`, `rewards="scoring,checkpoints"`,
      `number_of_left_players_agent_controls=1`; run 2 episodes with random
      actions. Note obs shape/dtype, action space, reward values, `done`
      semantics, `info` keys (`score_reward`?).
- [x] 0.4 Throughput: steps/s over ~2000 steps, no render.
- [x] 0.5 Rendering path: (a) `create_environment(render=True)` + `env.render()`
      returning an RGB array under `xvfb-run`/headless; (b) gfootball's own
      `write_video=True, logdir=...` dumps. Decide which (if any) the
      framework video path can use.
- [x] 0.6 Vectorisation: `gymnasium.vector.AsyncVectorEnv` with 2 thunks
      (gfootball already forks an engine process per env — confirm fork-safety
      and clean shutdown).
- [x] 0.7 Seeding/determinism: what does the env expose? (`create_environment`
      has no seed arg; check `env.seed`, scenario `deterministic` flag,
      `other_config_options`.) Document, don't fight it.
- [x] 0.8 Write findings at the bottom of this file (section "Spike findings")
      and pick the install recipe for Phase 1/3.

Exit criteria: a written recipe (`apt` list + `pip` sequence) that yields a
working `create_environment` on Python 3.10 / numpy 2 / gymnasium 1.2, **or**
a clear statement that a separate pinned image (older Python/torch) is
required, in which case Phase 3 builds that image instead.

## Phase 1 — Framework integration (Case 4 of docs/adding-a-new-environment.md)

Core rule: `envs/factory.py`, `envs/wrappers.py`, `train.py`, algorithms are
**not** touched. Everything gfootball-specific lives in the two extension
points.

- [ ] 1.1 `envs/custom_envs/gfootball.py` — gymnasium shim
  - `class GFootballEnv(gym.Env)` wrapping `gfootball.env.create_environment`.
  - `reset(seed=None, options=None) -> (obs, info)`; seed forwarded if 0.7
    finds a way, otherwise ignored and documented.
  - `step(a) -> (obs, reward, terminated, truncated, info)`; gfootball only
    yields `done`. Mapping: `terminated=done, truncated=False` (scenario
    time-limit treated as terminal; note the bias in the docstring; revisit
    if `info` lets us split them).
  - Squeeze the leading player axis when controlling one player so spaces are
    `Box(115,) float32` and `Discrete(19)`.
  - `metadata["render_modes"]` and `render()` per 0.5 outcome.
  - `close()` shuts the engine subprocess down.
  - Factory `make_gfootball(scenario, representation="simple115_v2",
    rewards="scoring,checkpoints", render_mode=None, **kw)` — all config
    `env_kwargs` arrive here; `kw` forwarded to `create_environment`.
- [ ] 1.2 `envs/custom_envs/__init__.py` — register ids, guarded by
      `try: import gfootball`:
  - `GFootball/academy_empty_goal_close-v0`
  - `GFootball/academy_empty_goal-v0`
  - `GFootball/academy_run_to_score-v0`
  - `GFootball/academy_run_to_score_with_keeper-v0`
  - `GFootball/academy_pass_and_shoot_with_keeper-v0`
  - `GFootball/academy_3_vs_1_with_keeper-v0`
  - (rest of the academy set is cheap to add; keep the list explicit so ids
    are stable and versioned.)
- [ ] 1.3 `envs/adapters/gfootball.py` — `register_adapter("GFootball/",
      EnvAdapter(...))`: `supports_training_video` per 0.5; `skip_episode_stats`
      stays `False` (stack adds `RecordEpisodeStatistics`); no custom
      vectoriser or evaluate unless 0.6 forces it. Import from the bundled
      adapters block in `envs/adapters/__init__.py` (guarded import, Meta-World
      pattern).
- [ ] 1.4 Config `configs/dqn_gfootball_empty_goal.yml` — `algorithm: dqn`,
      `env_id: GFootball/academy_empty_goal_close-v0`, `env_kwargs` spelling
      out representation/rewards, `num_envs: 1`, `track: true`,
      `checkpoint_every: 0`, DQN hyperparameters tuned for a ~300 steps/s env
      (fewer `total_timesteps`, `learning_starts`, buffer sized for it).
      Comment header in the CartPole-config style.
- [ ] 1.5 Tests `tests/test_gfootball.py` — `pytest.importorskip("gfootball")`
      at module level: spaces, 5-tuple `step`, `reset` returns `(obs, info)`,
      `discrete_control_wrappers` applies, `make_env` thunk builds. Keep
      step counts tiny (engine start-up is seconds). `tests/test_configs_load.py`
      covers the new YAML automatically.
- [ ] 1.6 Smoke: `python train.py --config dqn_gfootball_empty_goal
      --override total_timesteps=2000 track=false capture_video=false` runs
      end-to-end incl. final eval. Then a real run to confirm learning on
      `empty_goal_close` (should reach ~1.0 scoring within a few 10^5 steps).
- [ ] 1.7 Docs: `docs/adding-a-new-environment.md` gets a short "GFootball"
      note (guarded registration + shim as a worked example of Case 4);
      `docs/quickstart-training.md` one-liner. Update
      `/memories/repo/conventions.md`.

## Phase 2 — Student quick-start image

The point of the whole exercise for the class: nobody compiles anything.

- [ ] 2.1 `docker/Dockerfile.gfootball` (separate from the MuJoCo image to keep
      the base image lean): apt deps from Phase 0, `pip install` recipe from
      0.8, then t-zero `requirements.txt`. CPU-only torch is acceptable here
      (DQN on a 115-d vector; students' laptops rarely have CUDA under Docker).
      Provide `docker-compose` service `gfootball` mirroring the existing one.
- [ ] 2.2 `cluster/t-zero-gfootball.def` (Apptainer) for the cluster path.
- [ ] 2.3 Verify the image on a clean machine: `docker build` time, image
      size, `python train.py --config dqn_gfootball_empty_goal` starts
      training within the first minute; wandb login via `.env` works.
- [ ] 2.4 Windows/macOS notes (Docker Desktop, volume mount path quirks,
      `xvfb` only needed for video). One page in the handout, not more.
- [ ] 2.5 Decide whether to publish a prebuilt image (GHCR / Docker Hub) so
      students `docker pull` instead of `docker build`. Strongly preferred if
      build time > ~10 min.

## Phase 3 — Project handout (outline only; content is Bruno's)

- Baseline: the provided config + a reference wandb run.
- Axes students can explore: scenario progression (`empty_goal_close` →
  `run_to_score_with_keeper` → `3_vs_1_with_keeper`), reward shaping
  (`scoring` vs `scoring,checkpoints`), DQN hyperparameters carried over from
  the assignment, optional SMM/pixels + CNN as stretch (needs its own stack).
- Same protocol as the DQN assignment: predict → run → chart → explain.
- Final greedy-eval video as the demo deliverable (if 0.5 rendering works).

## Out of scope (tracked elsewhere)

- `algorithms/ppo_discrete_action.py` (CleanRL `ppo.py` port) — separate
  branch after this lands.
- Multi-agent control (`number_of_left_players_agent_controls > 1`), full
  11v11, self-play, SMM/pixel representation + CNN stack.
- DQN checkpoint/resume (unchanged limitation).

## Risks / open decisions

| Risk | Mitigation |
|---|---|
| gfootball won't compile on Py 3.10 / numpy 2 | Phase 0 exit criteria → separate pinned image; framework code unaffected |
| `gym` 0.21 pin | `--no-deps` + import-only `gym` version; shim never calls gym API |
| Rendering headless is flaky | `supports_training_video=False`; eval video via gfootball's own dump, or drop video |
| Engine subprocess + `AsyncVectorEnv` | 0.6 test; fallback `num_envs=1` (DQN's constraint anyway) |
| Slow env → students' runs too long | pick `empty_goal_close`, cap `total_timesteps`, tuned config in 1.4 |
| Students on Windows/macOS | Docker image (Phase 2) is the only supported path |

## Spike findings (2026-09-21, Ubuntu 22.04 image, Python 3.10.12)

Scratch artefacts: `/tmp/gfootball-spike/{Dockerfile,spike.py,spike2.py,spike3.py}`.
Host had no cmake/SDL/Boost, so everything ran in Docker (which is the
student path anyway).

**Install recipe — works on Py 3.10 / numpy 2.2.6 / gymnasium 1.2.0:**

```dockerfile
apt-get install -y git cmake build-essential libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
    libdirectfb-dev libst-dev mesa-utils xvfb python3-pip python3-dev
pip install --upgrade pip setuptools wheel psutil
git clone --depth 1 https://github.com/google-research/football.git /opt/football
pip install --no-deps --no-build-isolation /opt/football     # ~100 s on 12 cores
pip install "gym==0.23.1" absl-py six pygame "opencv-python-headless>=4.10"
```

Why each flag:
- `--no-deps`: skips upstream's `gym<=0.21.0` pin (0.21 no longer installs on
  modern pip — `extras_require` metadata error).
- `--no-build-isolation`: `build_game_engine.sh` imports `psutil`; in pip's
  isolated build env it's absent, the engine silently isn't built (setup.py
  marks the extension `optional=True`) and `pip install` *still reports
  success*. Always assert `gfootball_engine/_gameplayfootball*.so` exists.
- `gym==0.23.1`: gfootball subclasses `gym.Env`/`gym.Wrapper`. 0.26 changed
  `Wrapper.reset` to `(obs, info)` → breaks gfootball's wrappers. 0.23.1
  installs, keeps the old wrapper API, no warnings, imports under numpy 2
  (prints a one-time "Gym unmaintained" banner). 0.24.1/0.25.2 also work but
  warn.
- `six`: imported by `football_action_set.py`, missing from upstream deps.
  `tensorflow/joblib/grpc/baselines` imports are only in optional modules
  (`players/ppo2_cnn.py`, `remote_football_env.py`) — not needed.
- Installed version reports `gfootball-2.10.3` (master, unreleased).

**API facts (representation `simple115v2` — no underscore; `simple115_v2`
raises):**
- `create_environment(env_name, representation="simple115v2",
  rewards="scoring,checkpoints", number_of_left_players_agent_controls=1,
  render=False, other_config_options={...})`.
- Spaces: `Box(-inf, inf, (115,), float32)`, `Discrete(19)`. `reset()` →
  `ndarray(115,) float32` (already squeezed by `SingleAgentObservationWrapper`).
- `step(a)` → `(obs, np.float32 reward, done: bool, info)`; `info` has only
  `score_reward` (±1 on goal, else 0). Checkpoint reward gives +0.1 per zone.
- Wrapper chain: `GetStateWrapper → SingleAgentRewardWrapper →
  SingleAgentObservationWrapper → Simple115StateWrapper →
  CheckpointRewardWrapper → FootballEnv`.
- `env.unwrapped.observation()[0]` returns the raw dict (has `steps_left`,
  `game_mode`, `score`, …) — usable to split terminated/truncated:
  `truncated = done and steps_left <= 0`.
- Episode end for academy scenarios is scenario-defined (`end_episode_on_score`,
  `end_episode_on_out_of_play`, `end_episode_on_possession_change`,
  `game_duration=400`). Idle policy on `empty_goal_close` ended at step 152
  via possession change, not time-out. Random policy scores in ~10–75 steps
  (returns 0.9–2.0 with checkpoints).

**Seeding:** `other_config_options={"game_engine_random_seed": s}` — same seed +
same actions → identical trajectory; different seed → different trajectory.
(First observation is identical regardless — fixed scenario start positions.)
Legacy `env.seed(s)` exists but is a no-op for the engine. Note upstream
derives `reverse_team_processing` from seed parity unless set explicitly.

**Throughput (12-core host, shared with another 7-core job — conservative):**
- single env, no render: **~280 steps/s**.
- `gymnasium.vector.AsyncVectorEnv` × 4 with a minimal shim: **~720 steps/s
  aggregate**, 0.2 s start-up, clean `close()`. Fork-safe.
- **Rendering is software GL (llvmpipe) and slow**: 1280×720 → ~5 steps/s;
  640×360 → ~9; 320×180 → ~16 (`other_config_options={"render_resolution_x":
  X, "render_resolution_y": Y}` — set both; y is *not* recomputed). No
  display/xvfb needed ("No DISPLAY defined, doing off-screen rendering").
  Once rendering is enabled the engine renders every step, so a
  rendering env is slow for its whole life. The rendering engine is cached
  per process — resolution can't be changed after the first render in a
  process.

**Decisions taken for Phase 1:**
- Shim creates the engine with `render=True` **only** when
  `render_mode="rgb_array"` (eval video), default 640×360. Adapter sets
  `supports_training_video=False`. A 400-step eval episode ≈ 45 s of video
  rendering — acceptable for the final eval, and the demo payoff for students.
- `terminated/truncated` split via `steps_left` from the raw observation.
- Seed: `reset(seed=s)` recreates nothing; seeding is applied at construction
  via `game_engine_random_seed`. The shim accepts `seed` in `env_kwargs`-style
  construction and documents that `reset(seed=)` is honoured only on the first
  reset (engine seed is fixed at construction).
- **Development loop for Phase 1 runs inside the Docker image** (host lacks
  the C++ toolchain). That image = spike image + t-zero `requirements.txt`
  with CPU torch — i.e. Phase 2.1 gets built now and doubles as the dev env.

Phase 0 checklist: all items `[x]`.
