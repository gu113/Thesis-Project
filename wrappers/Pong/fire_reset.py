import gymnasium as gym

# Fire Reset Wrapper
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        self.fire_action = 1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs, _, terminated, truncated, _ = self.env.step(self.fire_action) # Take FIRE action
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
            
        return obs, info
    