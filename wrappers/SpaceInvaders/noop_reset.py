import gym
import random

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP", "First action must be NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = random.randint(1, self.noop_max)
        for _ in range(noops):
            obs, reward, done, truncated, info = self.env.step(0)
            if done or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info
