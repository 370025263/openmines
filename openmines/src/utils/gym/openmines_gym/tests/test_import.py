import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from openmines_gym import GymMineEnv

# 创建和测试环境
env = GymMineEnv("../../../../conf/north_pit_mine.json")
env = FlattenObservation(env)

# 测试环境
obs, info = env.reset()
print("Observation shape:", obs.shape)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()
        break

env.close()