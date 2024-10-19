import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import argparse
from collections import deque
from datetime import datetime
from tqdm import tqdm

# 禁用警告以加快执行速度
import warnings
warnings.filterwarnings('ignore')

# 导入 TensorBoard 的 SummaryWriter
from torch.utils.tensorboard import SummaryWriter

# 导入您的环境
from openmines.src.utils.rl_env import MineEnv

# 超参数定义
GAMMA = 0.99
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_STEPS = 1000
NUM_EPISODES = 500
MEMORY_SIZE = 10000
TARGET_UPDATE = 10  # 每隔多少个回合更新目标网络
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500  # 衰减率

# 定义 Q 网络
class DQN(nn.Module):
    def __init__(self, state_dim, action_dims):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_dims = action_dims
        # 输出层：我们将为每个事件类型创建一个独立的输出层
        self.output_layers = nn.ModuleList([
            nn.Linear(128, dim) for dim in action_dims
        ])

    def forward(self, state, event_type):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # 由于 event_type 可能是批量的，需要特殊处理
        if state.shape[0] == 1:
            # 单个样本
            q_values = self.output_layers[event_type.item()](x)
        else:
            # 批量样本
            q_values = torch.stack([
                self.output_layers[et.item()](h) for et, h in zip(event_type, x)
            ])
        return q_values

# 经验回放缓冲区
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(tuple(args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 特征预处理函数（与 PPO 代码一致）
def preprocess_features(observation):
    event_name = observation['event_name']
    if event_name == "init":
        event_type = 0
        action_space_n = observation['info']['load_num']
    elif event_name == "haul":
        event_type = 1
        action_space_n = observation['info']['unload_num']
    else:
        event_type = 2
        action_space_n = observation['info']['load_num']

    truck_location = observation['the_truck_status']['truck_location_index']
    if truck_location is None:
        truck_location = 0

    order_and_position = np.array([event_type, truck_location, action_space_n])

    truck_num = observation['mine_status']['truck_count']

    truck_features = np.array([
        observation['the_truck_status']['truck_load'] / (observation['the_truck_status']['truck_capacity'] + 1e-8),
        observation['the_truck_status']['truck_cycle_time'] / 1000
    ])

    target_features = np.concatenate([
        np.array(observation['target_status']['queue_lengths']) / (truck_num + 1e-8),
        np.log(np.array(observation['target_status']['capacities']) + 1),
        np.log(np.array(observation['target_status']['est_wait']) + 1),
        np.log(np.array(observation['target_status']['produced_tons']) + 1),
        np.log(np.array(observation['target_status']['service_counts']) + 1),
    ])

    state = np.concatenate([order_and_position, truck_features, target_features])

    assert not np.isnan(state).any(), f"NaN detected in state: {state}"

    return state, event_type

# 训练 DQN 模型的函数
def train_dqn(args):
    # 创建环境并获取状态和动作空间维度
    env = MineEnv.make(args.env_config, log=False, ticks=False)
    observation, _ = env.reset(seed=42)
    state, event_type = preprocess_features(observation)
    state_dim = len(state)
    action_dims = [
        observation['info']['load_num'],  # 对于 init
        observation['info']['unload_num'],  # 对于 haul
        observation['info']['load_num']  # 对于 unhaul
    ]
    env.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(state_dim, action_dims).to(device)
    target_net = DQN(state_dim, action_dims).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0

    # 创建 TensorBoard SummaryWriter
    writer = SummaryWriter()

    def select_action(state, event_type):
        nonlocal steps_done
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if random.random() < eps_threshold:
            action = random.randrange(action_dims[event_type.item()])
            return torch.tensor([[action]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                q_values = policy_net(state, event_type)
                action = q_values.max(1)[1].view(1, 1)
                return action

    episode_rewards = []
    total_progress = tqdm(range(NUM_EPISODES), desc='Training DQN')

    for episode in total_progress:
        env = MineEnv.make(args.env_config, log=False, ticks=False)
        observation, _ = env.reset(seed=episode)
        state_np, event_type = preprocess_features(observation)
        state = torch.tensor([state_np], device=device, dtype=torch.float)
        event_type = torch.tensor([event_type], device=device, dtype=torch.long)
        total_reward = 0

        # 用于记录每个动作的选择次数
        action_counts = {0: {}, 1: {}, 2: {}}

        for t in range(MAX_STEPS):
            action = select_action(state, event_type)
            action_item = action.item()
            observation, reward, done, truncated, _ = env.step(action_item)
            total_reward += reward
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float)

            next_state_np, next_event_type = preprocess_features(observation)
            next_state = torch.tensor([next_state_np], device=device, dtype=torch.float)
            next_event_type = torch.tensor([next_event_type], device=device, dtype=torch.long)

            # 将经验存入回放缓冲区
            memory.push(state, event_type, action, reward_tensor, next_state, next_event_type, done)

            # 记录动作选择次数
            et = event_type.item()
            if action_item in action_counts[et]:
                action_counts[et][action_item] += 1
            else:
                action_counts[et][action_item] = 1

            state = next_state
            event_type = next_event_type

            # 经验回放并训练
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch_state, batch_event_type, batch_action, batch_reward, batch_next_state, batch_next_event_type, batch_done = zip(*transitions)

                batch_state = torch.cat(batch_state)
                batch_action = torch.cat(batch_action)
                batch_reward = torch.cat(batch_reward)
                batch_next_state = torch.cat(batch_next_state)
                batch_done = torch.tensor(batch_done, device=device, dtype=torch.bool)

                batch_event_type = torch.cat(batch_event_type)
                batch_next_event_type = torch.cat(batch_next_event_type)

                # 计算 Q(s_t, a)
                state_action_values = policy_net(batch_state, batch_event_type).gather(1, batch_action)

                # 计算 V(s_{t+1})，对于终止状态，V(s)为0
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                with torch.no_grad():
                    non_final_mask = ~batch_done
                    non_final_next_states = batch_next_state[non_final_mask]
                    non_final_next_event_types = batch_next_event_type[non_final_mask]
                    if len(non_final_next_states) > 0:
                        next_q_values = target_net(non_final_next_states, non_final_next_event_types)
                        next_state_values[non_final_mask] = next_q_values.max(1)[0]
                    else:
                        next_state_values[non_final_mask] = 0.0

                # 计算期望的 Q 值
                expected_state_action_values = (next_state_values * GAMMA) + batch_reward.squeeze()

                # 计算损失
                loss = nn.MSELoss()(state_action_values.squeeze(), expected_state_action_values)

                # 优化模型
                optimizer.zero_grad()
                loss.backward()
                # 防止梯度爆炸
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
                optimizer.step()

                # 记录参数和梯度的平均大小
                total_norm = 0
                for p in policy_net.parameters():
                    param_norm = p.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar('Parameters/Parameter Norm', total_norm, steps_done)

                total_grad_norm = 0
                for p in policy_net.parameters():
                    if p.grad is not None:
                        grad_norm = p.grad.data.norm(2)
                        total_grad_norm += grad_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                writer.add_scalar('Parameters/Gradient Norm', total_grad_norm, steps_done)

            if done or truncated:
                break

        episode_rewards.append(total_reward)
        env.close()

        # 每隔一定的回合数更新目标网络
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 记录每个回合的平均奖励
        writer.add_scalar('Episode Reward', total_reward, episode)

        # 记录每个事件类型的动作选择频率
        for et in action_counts:
            total_actions = sum(action_counts[et].values())
            if total_actions > 0:
                action_freq = {f'Action_{a}': count / total_actions for a, count in action_counts[et].items()}
                writer.add_scalars(f'Action Frequencies/EventType_{et}', action_freq, episode)

        # 更新进度条
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            total_progress.set_postfix({'Average Reward': f'{avg_reward:.2f}'})

    print('Training complete')
    writer.close()
    torch.save(policy_net.state_dict(), 'dqn_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DQN Mining Environment Tester")
    parser.add_argument("--env_config", type=str, default="../../openmines/src/conf/north_pit_mine.json",
                        help="Path to environment configuration file")
    args = parser.parse_args()

    train_dqn(args)
