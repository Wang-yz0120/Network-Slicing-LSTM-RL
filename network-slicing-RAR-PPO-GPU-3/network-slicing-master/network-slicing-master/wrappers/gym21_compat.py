# wrappers/gym21_compat.py
import numpy as np
import gymnasium as gym

class Gym21Compat(gym.Wrapper):
    """
    Adapt a legacy Gym env (reset()->obs, step()->(obs,reward,done,info))
    to Gymnasium API:
      - reset(seed=None, options=None) -> (obs, info)
      - step(action) -> (obs, reward, terminated, truncated, info)
    Also forwards render/close, ignores unused kwargs safely.
    """
    def __init__(self, env):
        super().__init__(env)
        # keep original spaces as-is
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)

    def reset(self, *, seed=None, options=None):
        # Try to seed if legacy env exposes a seeding API
        if seed is not None:
            # common legacy seeding patterns
            if hasattr(self.env, "seed"):
                try:
                    self.env.seed(seed)
                except TypeError:
                    # some old envs take a list/np.random.RandomState; ignore
                    pass
            elif hasattr(self.env, "np_random"):
                try:
                    self.env.np_random = np.random.RandomState(seed)
                except Exception:
                    pass
        # Call legacy reset (obs only)
        obs = self.env.reset()
        info = {}
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
            return obs, float(reward), terminated, truncated, info if isinstance(info, dict) else {}
        if isinstance(out, tuple) and len(out) == 5:
            # already gymnasium-style
            obs, reward, terminated, truncated, info = out
            return obs, float(reward), bool(terminated), bool(truncated), info if isinstance(info, dict) else {}
        # Fallback
        obs, reward = out[0], out[1]
        return obs, float(reward), False, False, {}

    def render(self, *args, **kwargs):
        if hasattr(self.env, "render"):
            return self.env.render(*args, **kwargs)

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
