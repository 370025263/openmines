import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocessing as mp
from openmines.src.utils.rl_env import MineEnv

# 禁用警告以加速运算
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 特征预处理函数
def preprocess_features(ob):
    # 提取相关特征
    truck_features = [
        ob['the_truck_status']['truck_load'],
        ob['the_truck_status']['truck_capacity'],
        ob['the_truck_status']['truck_cycle_time'],
        ob['the_truck_status']['truck_speed']
    ]

    # 处理目标状态
    target_features = []
    for i in range(5):  # 假设有5个目标位置
        target_features.extend([
            ob['target_status']['queue_lengths'][i],
            ob['target_status']['capacities'][i],
            ob['target_status']['est_wait'][i],
            ob['target_status']['produced_tons'][i],
            ob['target_status']['service_counts'][i]
        ])

    # 处理道路状态
    road_features = []
    for road_type in ['charging2load', 'load2dump', 'dump2load']:
        for i in range(5):  # 假设每种道路类型有5个
            road_features.extend([
                ob['cur_road_status'][road_type]['truck_count'].get(i, 0),
                ob['cur_road_status'][road_type]['distances'].get(i, 0),
                ob['cur_road_status'][road_type]['truck_jam_count'].get(i, 0),
                ob['cur_road_status'][road_type]['repair_count'].get(i, 0)
            ])

    # 合并所有特征
    all_features = truck_features + target_features + road_features

    # 归一化特征
    normalized_features = (np.array(all_features) - np.mean(all_features)) / (np.std(all_features) + 1e-8)

    return torch.FloatTensor(normalized_features).unsqueeze(0).to(device)


# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


# 定义值网络
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# PPO Agent
class PPOAgent:
    def __init__(self, input_dim, output_dim, learning_rate=3e-4, gamma=0.99, epsilon=0.2, value_coef=0.5,
                 entropy_coef=0.01):
        self.policy = PolicyNetwork(input_dim, output_dim).to(device)
        self.value = ValueNetwork(input_dim).to(device)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def act(self, state):
        with torch.no_grad():
            probs = self.policy(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, states, actions, old_log_probs, rewards, next_states, dones):
        states = torch.cat(states)
        actions = torch.tensor(actions).long().to(device)
        old_log_probs = torch.cat(old_log_probs)
        rewards = torch.tensor(rewards).float().to(device)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones).float().to(device)

        # 计算优势
        values = self.value(states).squeeze()
        next_values = self.value(next_states).squeeze()
        td_target = rewards + self.gamma * next_values * (1 - dones)
        td_error = td_target - values
        advantages = td_error.detach()

        # PPO更新
        for _ in range(10):  # 进行多次更新
            new_probs = self.policy(states)
            new_log_probs = torch.log(new_probs.gather(1, actions.unsqueeze(1)).squeeze())
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(self.value(states).squeeze(), td_target.detach())

            entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=1).mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# 训练函数
def train(env, agent, num_episodes, max_steps):
    writer = SummaryWriter()

    for episode in tqdm(range(num_episodes)):
        observation, info = env.reset()
        state = preprocess_features(observation)
        episode_reward = 0

        for step in range(max_steps):
            action, log_prob = agent.act(state)
            next_observation, reward, done, truncated, info = env.step(action)
            next_state = preprocess_features(next_observation)
            episode_reward += reward

            agent.update([state], [action], [log_prob], [reward], [next_state], [done or truncated])

            if done or truncated:
                break

            state = next_state

        writer.add_scalar('Reward', episode_reward, episode)

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

    writer.close()


# 多进程训练函数
def train_worker(rank, num_episodes, max_steps):
    env = MineEnv.make("../../openmines/src/conf/north_pit_mine.json", log=False, ticks=False)
    env.reset(seed=42 + rank)  # 每个进程使用不同的种子
    agent = PPOAgent(input_dim=85, output_dim=5)  # 根据你的环境调整输入和输出维度
    train(env, agent, num_episodes // mp.cpu_count(), max_steps)


if __name__ == "__main__":
    num_episodes = 1000
    max_steps = 1000
    num_processes = mp.cpu_count()

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train_worker, args=(rank, num_episodes, max_steps))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()