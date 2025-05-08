import gymnasium as gym

# Custom Reward Modifier Wrapper
class RewardModifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        # Custom reward shaping
        if reward > 0:
            reward += 5  # Encourage hitting enemies
        if terminated:
            reward -= 10  # Penalize game over

        return state, reward, terminated, truncated, info

# Custom Reward Modifier Wrapper
class ComplexRewardModifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_lives = None
        self.frame_count = 0
        self.noop_streak = 0
        self.prev_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_lives = info.get('ale.lives', 3)
        self.frame_count = 0
        self.noop_streak = 0
        self.prev_action = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.frame_count += 1
        current_lives = info.get('ale.lives', self.last_lives)

        shaped_reward = float(reward)

        # Bonus for hitting an enemy
        if reward > 0:
            shaped_reward += 5.0

        # Penalty for losing a life
        if current_lives < self.last_lives:
            shaped_reward -= 20.0
        self.last_lives = current_lives

        # Survival reward every 50 frames
        if self.frame_count % 50 == 0:
            shaped_reward += 0.5

        # Penalty for idling (too many NOOPs)
        if action == 0:
            self.noop_streak += 1
        else:
            self.noop_streak = 0

        if self.noop_streak >= 10:
            shaped_reward -= 0.1

        # Penalty at episode end
        if done:
            shaped_reward -= 10.0

        self.prev_action = action
        return obs, shaped_reward, terminated, truncated, info