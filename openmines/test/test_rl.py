"""
在不使用gym包的情况下训练一个rl-agent
跟矿山的交互主要通过rl_dispatch调度模块
"""
from openmines.src.utils.rl_env import MineEnv


# 创建CartPole环境
env = MineEnv.make("../../openmines/src/conf/north_pit_mine.json")

# 设置随机种子，保证可复现性
env.reset(seed=42)
env.action_space.seed(42)

# 从env中获取dispatcher，然后覆盖为
class Agent:
    def __init__(self):
        pass

    def act(self, obs):
        return 0

agent = Agent()

# 开始训练
for episode in range(10):  # 训练10个回合
    observation, info = env.reset()
    done = False
    total_reward = 0

    action = agent.act(observation)
    while not done:
        # 随机选择一个动作
        #action = env.action_space.sample()
        action = agent(observation)
        # 应用动作到环境中
        observation, reward, done, truncated, info = env.step(action)

        # 累加奖励
        total_reward += reward

        # 如果游戏结束，打印本回合的总奖励
        if done:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()
