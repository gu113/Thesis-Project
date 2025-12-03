import gymnasium as gym

# Fire Reset Wrapper
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        self.fire_action = 1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
    
        obs, _, terminated, truncated, _ = self.env.step(self.fire_action) # Take first FIRE action
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        obs, _, terminated, truncated, _ = self.env.step(self.fire_action) # Take second FIRE action
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        return obs, info

# Fire on Life Loss Wrapper
class FireOnLifeLoss(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_lives = None
        self.fire_action = 1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_lives = info.get('lives', None)
        
        # FIRE twice on reset
        for _ in range(2):
            obs, _, terminated, truncated, info = self.env.step(self.fire_action) 
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.last_lives = info.get('lives', None)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_lives = info.get('lives', None)

        # Check for life loss
        if self.last_lives is not None and current_lives is not None and current_lives < self.last_lives:
            # Life lost, press FIRE
            obs, _, terminated, truncated, info = self.env.step(self.fire_action)

        self.last_lives = current_lives
        return obs, reward, terminated, truncated, info
