import gymnasium as gym

# Pong Score Tracking Wrapper
class PongScoreWrapper(gym.Wrapper):
    def __init__(self, env, win_score=21):
        super().__init__(env)
        self.agent_score = 0
        self.opponent_score = 0
        self.win_score = win_score

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.agent_score = 0
        self.opponent_score = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward == 1.0:
            self.agent_score += 1  # Increment agent's score
        elif reward == -1.0:
            self.opponent_score += 1  # Increment opponent's score

        info['agent_score'] = self.agent_score
        info['opponent_score'] = self.opponent_score

        if self.agent_score == self.win_score or self.opponent_score == self.win_score:
            terminated = True  # End episode if either player reaches the win score

        return obs, reward, terminated, truncated, info

    def get_scores(self):
        return self.agent_score, self.opponent_score