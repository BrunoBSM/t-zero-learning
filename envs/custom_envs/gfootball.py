"""Google Research Football (``gfootball``) as a Gymnasium environment.

gfootball (github.com/google-research/football, archived 2026) targets the
legacy ``gym`` API: ``reset() -> obs``, ``step() -> (obs, r, done, info)``,
``render(mode)``, no ``seed`` in ``reset``.  :class:`GFootballEnv` adapts one
scenario to the Gymnasium 1.x contract so the rest of the framework can treat
it like any other env.  Ids are registered in ``envs/custom_envs/__init__.py``
(only when gfootball is importable) as ``GFootball/<scenario>-v0``.

Engine facts this shim relies on (verified against gfootball 2.10.3, see
docs/plans/gfootball-integration.md):

- ``representation="simple115v2"`` gives a flat ``(115,) float32`` vector and
  the default action set is ``Discrete(19)`` — the input contract of the
  ``discrete_control`` wrapper stack / DQN.
- Engine randomness is read from ``config["game_engine_random_seed"]`` at
  every ``reset()`` (a fresh random value if the key is absent).  The shim
  seeds ``self.np_random`` from ``reset(seed=...)`` and writes one engine seed
  per episode, so a run seed gives a reproducible *sequence* of episodes and
  a deterministic policy still sees different episodes.  Note the seed only
  matters where the engine makes random decisions (built-in AI players);
  ``academy_empty_goal_close`` is fully deterministic given the actions.
- The engine only reports ``done``.  The raw observation's ``steps_left``
  tells us whether the scenario clock ran out (``truncated``) or the scenario
  ended on its own terms — goal, ball out, possession change (``terminated``).
- Rendering is software GL and slow (~10 steps/s); it is enabled lazily on the
  first ``render()`` call and switched off again on the next ``step()`` that
  nobody rendered, so a ``render_mode="rgb_array"`` env that is not being
  recorded runs at full speed (the framework builds worker 0 with
  ``render_mode="rgb_array"`` whenever ``capture_video`` is on).
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np

# Football Academy scenarios (gfootball/scenarios/academy_*.py), roughly by difficulty.
ACADEMY_SCENARIOS = (
    "academy_empty_goal_close",
    "academy_empty_goal",
    "academy_run_to_score",
    "academy_run_to_score_with_keeper",
    "academy_pass_and_shoot_with_keeper",
    "academy_run_pass_and_shoot_with_keeper",
    "academy_3_vs_1_with_keeper",
    "academy_corner",
    "academy_counterattack_easy",
    "academy_counterattack_hard",
    "academy_single_goal_versus_lazy",
)


class GFootballEnv(gym.Env):
    """One gfootball scenario, single controlled player, Gymnasium API.

    Parameters map onto ``gfootball.env.create_environment``:

    - ``scenario``: gfootball level name (``env_name``), e.g.
      ``"academy_empty_goal_close"``.
    - ``representation``: observation encoding; ``"simple115v2"`` (flat
      vector) is the one the flat-MLP agents expect.
    - ``rewards``: ``"scoring"`` (±1 per goal) or ``"scoring,checkpoints"``
      (adds +0.1 shaping for each of 10 zones approached with the ball).
    - ``render_resolution``: ``(width, height)`` of rendered frames.
    - ``**create_kwargs``: forwarded verbatim (e.g. ``stacked``,
      ``other_config_options``).
    """

    # One agent action = 100 ms of game time.
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scenario: str,
        representation: str = "simple115v2",
        rewards: str = "scoring,checkpoints",
        render_mode: str | None = None,
        render_resolution: tuple[int, int] = (640, 360),
        **create_kwargs,
    ):
        if render_mode not in (None, "rgb_array"):
            raise ValueError(
                f"GFootball: unsupported render_mode {render_mode!r}; "
                "use None or 'rgb_array'."
            )
        self.render_mode = render_mode
        self._scenario = scenario
        self._representation = representation
        self._rewards = rewards
        self._render_resolution = tuple(int(v) for v in render_resolution)
        self._create_kwargs = dict(create_kwargs)
        self._user_pins_team_order = "reverse_team_processing" in (
            self._create_kwargs.get("other_config_options") or {}
        )

        self._env = None
        self._render_requested = False
        self._build_engine()

        obs_space = self._env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.asarray(obs_space.low, dtype=np.float32),
            high=np.asarray(obs_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(int(self._env.action_space.n))

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def _build_engine(self) -> None:
        # Deferred import: keeps `import envs` cheap and banner-free when
        # gfootball is installed but unused.
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        from gfootball.env import create_environment

        other = dict(self._create_kwargs.get("other_config_options") or {})
        # Resolution is read when the (rendering) engine is created, so it
        # must be passed up front even though rendering starts off.
        width, height = self._render_resolution
        other.setdefault("render_resolution_x", width)
        other.setdefault("render_resolution_y", height)

        kwargs = {k: v for k, v in self._create_kwargs.items() if k != "other_config_options"}
        self._env = create_environment(
            env_name=self._scenario,
            representation=self._representation,
            rewards=self._rewards,
            number_of_left_players_agent_controls=1,
            render=False,
            other_config_options=other,
            **kwargs,
        )
        self._rendering_on = False

    def _raw_observation(self) -> dict:
        return self._env.unwrapped.observation()[0]

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Both keys are read by the scenario builder inside the engine's reset.
        # reverse_team_processing mirrors the processing order; upstream derives
        # it from the seed's parity once per env, we do it once per episode.
        engine_seed = int(self.np_random.integers(0, 2**31 - 1))
        config = self._env.unwrapped._config
        config["game_engine_random_seed"] = engine_seed
        if not self._user_pins_team_order:
            config["reverse_team_processing"] = bool(engine_seed % 2)
        obs = self._env.reset()
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        self._maybe_disable_render()
        obs, reward, done, info = self._env.step(int(action))
        info = dict(info)
        if done:
            raw = self._raw_observation()
            steps_left = int(raw["steps_left"])
            info["steps_left"] = steps_left
            scored = int(info.get("score_reward", 0)) != 0
            truncated = steps_left <= 0 and not scored
            terminated = not truncated
        else:
            terminated = truncated = False
        return np.asarray(obs, dtype=np.float32), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        self._render_requested = True
        self._rendering_on = True
        frame = self._env.render(mode="rgb_array")
        return np.asarray(frame, dtype=np.uint8)

    def _maybe_disable_render(self) -> None:
        # Rendering was on but nobody asked for a frame since the last step:
        # stop paying for it (RecordVideo has finished its episode).
        if self._rendering_on and not self._render_requested:
            self._env.unwrapped.disable_render()
            self._rendering_on = False
        self._render_requested = False

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


def make_gfootball_env(scenario: str, render_mode: str | None = None, **kwargs) -> gym.Env:
    """Registry entry point (``gym.make`` forwards ``env_kwargs`` here)."""
    return GFootballEnv(scenario, render_mode=render_mode, **kwargs)
