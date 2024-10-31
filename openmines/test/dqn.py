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
import string
from collections import deque, defaultdict
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# 禁用警告以加快执行速度
import warnings
warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter
from openmines.src.utils.rl_env import MineEnv

# ===== 超参数定义 =====
GAMMA = 0.999  # 折扣因子
TIME_SCALE = 1  # 时间衰减系数 X
LEARNING_RATE = 1e-3  # 学习率
BATCH_SIZE = 256  # 批次大小
MAX_STEPS = 1000  # 每个回合最大步数
NUM_EPISODES = 20000  # 总回合数
MEMORY_SIZE = 512*10  # 经验回放缓冲区大小
TARGET_UPDATE = 5  # 目标网络更新频率
EPS_START = 0.19  # 初始探索率
EPS_END = 0.01  # 最终探索率
EPS_DECAY = 1000*100  # 探索率衰减速度
TIME_ATTENTION = False  # 是否使用时间注意力
MODEL_NAME = "DuelingDQN" # "DQN" or "DuelingDQN"


def generate_run_id():
    """生成唯一的运行ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{timestamp}_{random_string}"

def generate_run_color():
    """生成随机颜色"""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


class DQN(nn.Module):
    def __init__(self, state_dim, action_dims):
        super(DQN, self).__init__()
        self.input_dim = state_dim + 1  # 状态维度 + 时间特征

        # 全连接层
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.ln1 = nn.LayerNorm(256)  # 添加LayerNorm

        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)  # 添加LayerNorm

        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)  # 添加LayerNorm

        # 输出层
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, dim),
                nn.LayerNorm(dim)  # 为每个输出层添加LayerNorm
            ) for dim in action_dims
        ])

        # 时间注意力
        self.time_attention = nn.Sequential(
            nn.Linear(1, 128),
            nn.LayerNorm(128)  # 为时间注意力添加LayerNorm
        )

    def forward(self, state, event_type, time_delta):
        # 确保time_delta是2维的 [B,1]
        if time_delta.dim() == 1:
            time_delta = time_delta.unsqueeze(-1)
        elif time_delta.dim() == 3:
            time_delta = time_delta.squeeze(1)  # 从[B,1,1]变为[B,1]

        # 时间特征处理，确保维度一致
        time_feature = time_delta  # 已经是[B,1]了


        combined_state = torch.cat([state, time_feature], dim=1)  # [B,state_dim+1]
        time_attention = torch.sigmoid(self.time_attention(time_delta))  # [B,128]

        # 前向传播
        x = self.ln1(torch.relu(self.fc1(combined_state)))
        x = self.ln2(torch.relu(self.fc2(x)))
        x = self.ln3(torch.relu(self.fc3(x)))

        if TIME_ATTENTION:
            x = x * time_attention

        # 根据batch size选择不同的处理方式
        if state.shape[0] == 1:
            q_values = self.output_layers[event_type.item()](x)
        else:
            q_values = torch.stack([
                self.output_layers[et.item()](h)
                for et, h in zip(event_type, x)
            ])
        return q_values


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dims):
        super(DuelingDQN, self).__init__()
        self.input_dim = state_dim + 1  # 状态维度 + 时间特征

        # 特征提取层
        self.features = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # 时间注意力层
        self.time_attention = nn.Sequential(
            nn.Linear(1, 128),
            nn.LayerNorm(128)
        )

        # Value Stream - 评估状态的价值
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出单个状态值
        )

        # Advantage Stream - 评估每个动作的优势
        self.advantage_streams = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, dim)  # 每个事件类型对应的动作维度
            ) for dim in action_dims
        ])

    def forward(self, state, event_type, time_delta):
        # 确保time_delta是2维的 [B,1]
        if time_delta.dim() == 1:
            time_delta = time_delta.unsqueeze(-1)
        elif time_delta.dim() == 3:
            time_delta = time_delta.squeeze(1)

        # 时间特征处理
        time_feature = time_delta
        combined_state = torch.cat([state, time_feature], dim=1)

        # 特征提取
        features = self.features(combined_state)

        if TIME_ATTENTION:
            time_attention = torch.sigmoid(self.time_attention(time_delta))
            features = features * time_attention

        # 计算状态值
        values = self.value_stream(features)

        # 根据batch size选择不同的处理方式
        if state.shape[0] == 1:
            # 单个样本处理
            advantages = self.advantage_streams[event_type.item()](features)
            # Q = V + (A - mean(A))
            q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            # batch处理
            q_values = []
            for i, et in enumerate(event_type):
                advantages = self.advantage_streams[et.item()](features[i:i + 1])
                q = values[i:i + 1] + (advantages - advantages.mean(dim=1, keepdim=True))
                q_values.append(q)
            q_values = torch.cat(q_values, dim=0)

        return q_values


class ReplayMemory:
    def __init__(self, capacity, device):
        self.memory = deque(maxlen=capacity)
        self.device = device

    def push(self, state, event_type, action, reward, next_state, next_event_type, done, time_delta, next_time_delta):
        # 确保所有输入都是tensor并且在正确的设备上
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        event_type = torch.as_tensor(event_type, dtype=torch.long, device=self.device)
        action = torch.as_tensor(action, dtype=torch.long, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        next_event_type = torch.as_tensor(next_event_type, dtype=torch.long, device=self.device)
        done = torch.as_tensor(done, dtype=torch.bool, device=self.device)
        time_delta = torch.as_tensor(time_delta, dtype=torch.float32, device=self.device)
        next_time_delta = torch.as_tensor(next_time_delta, dtype=torch.float32, device=self.device)

        self.memory.append((state, event_type, action, reward, next_state,
                            next_event_type, done, time_delta, next_time_delta))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def preprocess_features(observation):
    """特征预处理"""
    time_delta = float(observation['info']['delta_time'])
    time_now = float(observation['info']['time'])

    event_name = observation['event_name']
    if event_name == "init":
        event_type = [1,0,0]
        action_space_n = observation['info']['load_num']
    elif event_name == "haul":
        event_type = [0,1,0]
        action_space_n = observation['info']['unload_num']
    else:
        event_type = [0,0,1]
        action_space_n = observation['info']['load_num']

    truck_location:list = observation['the_truck_status']['truck_location_onehot']  # [1,M+N+1]
    #print("event_type", event_type)
    #print("truck_location", truck_location)

    order_and_position = np.array([event_type + truck_location + [action_space_n]])  # action_space_n maybe meanless
    truck_num = observation['mine_status']['truck_count']

    truck_features = np.array([
        np.log(observation['the_truck_status']['truck_load'] + 1),
        np.log(observation['the_truck_status']['truck_cycle_time'] + 1),
    ])

    # range should be 0-M+N as well.
    target_features = np.concatenate([
        np.array(observation['target_status']['queue_lengths']) / (truck_num + 1e-8),
        np.log(np.array(observation['target_status']['capacities']) + 1),
        np.log(np.array(observation['target_status']['est_wait']) + 1),
    ])
    # road distances, traffic truck count
    road_dist = np.array(observation['cur_road_status']['oh_distances'])
    road_traffic = np.array(observation['cur_road_status']['oh_truck_jam_count'])
    road_jam = np.array(observation['cur_road_status']['oh_truck_jam_count'])
    #print("road_dist", road_dist)
    #print("road_traffic", road_traffic)

    state = np.concatenate([order_and_position.squeeze(), truck_features, target_features, road_dist, road_traffic, road_jam]) # ])  # 3+M+N+1,2,3(M+N),(M+(M+N)*2)*3
    #state = np.concatenate([order_and_position.squeeze()])
    assert not np.isnan(state).any(), f"NaN detected in state: {state}"
    assert not np.isnan(time_delta), f"NaN detected in time_delta: {time_delta}"
    assert not np.isnan(time_now), f"NaN detected in time_now: {time_now}"

    event_type_index = event_type.index(1)
    return state, event_type_index, time_delta, time_now


def train_dqn(args):
    """训练DQN智能体"""
    # 初始化运行标识
    run_id = generate_run_id()
    run_color = generate_run_color()
    log_dir = os.path.join("runs", run_id)
    os.makedirs(log_dir, exist_ok=True)

    # 配置参数
    config = {
        "run_id": run_id,
        "run_color": run_color,
        "env_config": args.env_config,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "time_scale": TIME_SCALE,
        "batch_size": BATCH_SIZE,
        "memory_size": MEMORY_SIZE,
        "eps_start": EPS_START,
        "eps_end": EPS_END,
        "eps_decay": EPS_DECAY,
    }

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 初始化环境和网络
    env = MineEnv.make(args.env_config, log=False, ticks=False)
    observation, _ = env.reset(seed=42)
    state, event_type, time_delta, _ = preprocess_features(observation)
    state_dim = len(state)
    action_dims = [
        observation['info']['load_num'],
        observation['info']['unload_num'],
        observation['info']['load_num']
    ]
    env.close()



    # 设置设备和网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODEL_NAME == "DQN":
        policy_net = DQN(state_dim, action_dims).to(device)
        target_net = DQN(state_dim, action_dims).to(device)
    elif MODEL_NAME == "DuelingDQN":
        policy_net = DuelingDQN(state_dim, action_dims).to(device)
        target_net = DuelingDQN(state_dim, action_dims).to(device)
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")

    # 加载预训练模型
    if args.pretrained_path:
        print(f"Loading pretrained model from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        target_net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model (epoch {checkpoint['epoch']}, accuracy {checkpoint['accuracy']:.4f})")
    else:
        target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    # 使用SGD with momentum
    optimizer = optim.SGD(
        policy_net.parameters(),
        lr=LEARNING_RATE,  # SGD通常需要比Adam大的学习率
        momentum=0.9,  # 动量项
        weight_decay=1e-4  # L2正则化
    )

    memory = ReplayMemory(MEMORY_SIZE, device)
    writer = SummaryWriter(log_dir)

    # hyperparameters
    writer.add_text(f'Hyperparameters/gamma', str(GAMMA))
    writer.add_text(f'Hyperparameters/learning_rate', str(LEARNING_RATE))
    writer.add_text(f'Hyperparameters/time_scale', str(TIME_SCALE))
    writer.add_text(f'Hyperparameters/batch_size', str(BATCH_SIZE))
    writer.add_text(f'Hyperparameters/memory_size', str(MEMORY_SIZE))
    writer.add_text(f'Hyperparameters/eps_start', str(EPS_START))
    writer.add_text(f'Hyperparameters/eps_end', str(EPS_END))
    writer.add_text(f'Hyperparameters/eps_decay', str(EPS_DECAY))
    writer.add_text(f'Hyperparameters/TIME_ATTENTION', str(TIME_ATTENTION))

    steps_done = 0

    def select_action(state, event_type, time_delta):
        """选择动作"""
        nonlocal steps_done
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        np.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        writer.add_scalar('Training/Epsilon', eps_threshold, steps_done)
        if random.random() < eps_threshold:
            return torch.tensor([[random.randrange(action_dims[event_type.item()])]],
                                device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                q_values = policy_net(state, event_type, time_delta)
                return q_values.max(1)[1].view(1, 1)

    episode_rewards = []
    episode_lengths = []
    total_progress = tqdm(range(NUM_EPISODES), desc=f'Training {MODEL_NAME} (Run ID: {run_id})')

    total_production = 0
    event_types = ['Init', 'Haul', 'Unhaul']

    # 预分配内存给batch训练
    batch_state = torch.zeros(BATCH_SIZE, state_dim, device=device)
    batch_event_type = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
    batch_action = torch.zeros(BATCH_SIZE, 1, dtype=torch.long, device=device)
    batch_reward = torch.zeros(BATCH_SIZE, 1, device=device)
    batch_next_state = torch.zeros(BATCH_SIZE, state_dim, device=device)
    batch_next_event_type = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
    batch_done = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=device)
    batch_time_delta = torch.zeros(BATCH_SIZE, 1, device=device)
    batch_next_time_delta = torch.zeros(BATCH_SIZE, 1, device=device)

    # 预分配状态和动作的内存
    current_state = torch.zeros(1, state_dim, device=device)
    current_event_type = torch.zeros(1, dtype=torch.long, device=device)
    current_time_delta = torch.zeros(1, dtype=torch.float, device=device)

    for episode in total_progress:
        env = MineEnv.make(args.env_config, log=False, ticks=False)
        observation, _ = env.reset(seed=episode)
        state_np, event_type, time_delta, time_now = preprocess_features(observation)

        # 使用预分配的内存
        current_state[0] = torch.tensor(state_np, device=device)
        current_event_type[0] = event_type
        current_time_delta[0] = time_delta

        total_reward = 0
        step_count = 0

        order_dict = [0,0,0]
        load_order_dist_dict = [0 for site_index in range(observation['info']['load_num'])]
        dump_order_dist_dict = [0 for site_index in range(observation['info']['unload_num'])]

        reward_time_dict = defaultdict(int)

        for t in range(MAX_STEPS):
            # 选择动作
            # 使用预分配的内存选择动作
            action = select_action(current_state, current_event_type, current_time_delta)
            action_item = action.item()

            # 收集ORDER数据（每5个回合）
            if episode % 5 == 0:
                time_key = int(time_now)
                order_dict[event_type] += 1
                if event_type == 0:  # Init事件
                    load_order_dist_dict[action_item] += 1
                elif event_type == 1:  # Haul事件
                    dump_order_dist_dict[action_item] += 1
                elif event_type == 2:  # Unhaul事件
                    load_order_dist_dict[action_item] += 1
                else:
                    raise ValueError(f"Unknown event type: {event_type}")

            # 执行动作
            observation, reward, done, truncated, _ = env.step(action_item)
            total_reward += reward
            reward_time_dict[int(time_now)] = reward

            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)  # adjusted_reward

            # 处理下一个状态
            # 处理下一个状态，使用预分配的内存
            next_state_np, next_event_type, next_time_delta, next_time_now = preprocess_features(observation)
            with torch.no_grad():
                next_state = torch.tensor([next_state_np], device=device)
                next_event_type_tensor = torch.tensor([next_event_type], device=device)
                next_time_delta_tensor = torch.tensor([next_time_delta], device=device)

            memory.push(state, event_type, action, reward_tensor, next_state,
                        next_event_type, done, current_time_delta, next_time_delta_tensor)  # next_time_delta is used for gamma

            state = next_state
            event_type = next_event_type
            current_time_delta = next_time_delta_tensor
            time_now = next_time_now
            step_count += 1

            # 经验回放训练
            if len(memory) >= BATCH_SIZE and t%500 == 0:
                transitions = memory.sample(BATCH_SIZE)
                # batch = list(zip(*transitions))

                # 使用预分配的内存填充batch数据
                for i, (s, et, a, r, ns, net, d, td, ntd) in enumerate(transitions):
                    batch_state[i] = s
                    batch_event_type[i] = et
                    batch_action[i] = a
                    batch_reward[i] = r
                    batch_next_state[i] = ns
                    batch_next_event_type[i] = net
                    batch_done[i] = d
                    batch_time_delta[i] = td
                    batch_next_time_delta[i] = ntd

                # 计算current Q values
                current_q_values = policy_net(batch_state, batch_event_type,
                                              batch_time_delta).gather(1, batch_action)
                # 计算next state values
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                with torch.no_grad():
                    non_final_mask = ~batch_done
                    if non_final_mask.any():
                        next_q_values = target_net(
                            batch_next_state[non_final_mask],
                            batch_next_event_type[non_final_mask],
                            batch_next_time_delta[non_final_mask]
                        )
                        next_state_values[non_final_mask] = next_q_values.max(1)[0]

                # 计算expected Q values
                gamma_t = GAMMA ** batch_next_time_delta.squeeze().float()  # gamma^delta_t. here is S_(t+1) - S_t time delta
                expected_q_values = (next_state_values * gamma_t) + batch_reward.squeeze()

                # 计算loss并更新
                loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1)  # 梯度裁剪
                optimizer.step()
                writer.add_scalar('Training/Loss', loss.item(), global_step=episode)
                # Q value stuff
                writer.add_scalar('Training/QMax', current_q_values.max().item(), global_step=episode)
                writer.add_scalar('Training/QMin', current_q_values.min().item(), global_step=episode)
                writer.add_scalar('Training/QMean', current_q_values.mean().item(), global_step=episode)

            # 更新当前状态
            current_state[0] = torch.tensor(next_state_np, device=device)
            current_event_type[0] = next_event_type
            current_time_delta[0] = next_time_delta
            time_now = next_time_now

            if done or truncated:
                break

        # 0. total length
        writer.add_scalar('Episode/Length', sum(order_dict), global_step=episode)
        # 每5个回合创建比较图表
        if episode % 5 == 0:
            # 1. 记录TOTAL ORDER分布 ON TYPE
            # 如果你想看到总的订单比例分布
            writer.add_scalars(
                'Orders/Distribution',
                {
                    'Init': order_dict[0],
                    'Haul': order_dict[1],
                    'Unhaul': order_dict[2]
                },
                global_step=episode
            )

            # 2. 记录奖励分布
            # 对每个时间点分别记录奖励
            for t, reward in reward_time_dict.items():
                writer.add_scalar(
                    f'Rewards/Episode_{episode}',  # 每个episode一条曲线
                    reward,  # 该时间点的奖励值
                    global_step=t  # x轴是时间步
                )

            # 3. 记录站点分布
            # 装载点分布
            writer.add_histogram(
                f'ORDER/LoadSite',
                np.array(load_order_dist_dict),
                episode
            )
            # 卸载点分布
            writer.add_histogram(
                f'ORDER/DumpSite',
                np.array(dump_order_dist_dict),
                episode
            )
            # 对于装载点
            for site_idx in range(len(load_order_dist_dict)):
                writer.add_scalar(
                    f'LoadSite/Site_{site_idx}',  # 每个装载点一条曲线
                    load_order_dist_dict[site_idx],  # 该装载点在当前episode的订单数
                    global_step=episode
                )

            # 也可以用add_scalars将所有装载点放在同一个图中
            writer.add_scalars(
                'LoadSites/All',
                {f'Site_{i}': count for i, count in enumerate(load_order_dist_dict)},
                global_step=episode
            )

            # 对于卸载点
            for site_idx in range(len(dump_order_dist_dict)):
                writer.add_scalar(
                    f'DumpSite/Site_{site_idx}',  # 每个卸载点一条曲线
                    dump_order_dist_dict[site_idx],  # 该卸载点在当前episode的订单数
                    global_step=episode
                )

            # 同样可以用add_scalars将所有卸载点放在同一个图中
            writer.add_scalars(
                'DumpSites/All',
                {f'Site_{i}': count for i, count in enumerate(dump_order_dist_dict)},
                global_step=episode
            )

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

        # Fixed Q-targets: V. Mnih et al., "Human-level control through deep reinforcement learning." Nature, 518 (7540):529–533, 2015.
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        update_production = observation['info']['produce_tons']
        production_increase = update_production - total_production
        total_production = update_production

        writer.add_scalar('Production/Total', total_production, episode)
        writer.add_scalar('Production/Increase', production_increase, episode)

        if episode % 10 == 0:
            avg_reward_last_10 = np.mean(episode_rewards[-10:])
            avg_length = step_count

        total_progress.set_postfix({
            'Reward': f'{total_reward:.2f}',
            'L10 Avg Reward': f'{avg_reward_last_10:.2f}',
            'Avg Length': f'{avg_length:.2f}',
            'Total Production': f'{total_production:.2f}'
        })

    total_progress.close()
    writer.close()

    print(f"\nTraining completed. Run ID: {run_id}, Color: {run_color}")
    print(f"Final Total Production: {total_production}")

    model_save_path = os.path.join(log_dir, "dqn_model.pth")
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'config': config,
    }, model_save_path)


def evaluate_model(model, env, num_episodes=10, device="cuda"):
    """评估训练好的模型"""
    model.eval()
    rewards = []
    productions = []

    writer = SummaryWriter('runs/evaluation')

    with torch.no_grad():
        for i in range(num_episodes):
            observation, _ = env.reset(seed=i)
            episode_reward = 0
            done = False

            eval_data = {
                'times': [],
                'events': [],
                'rewards': [],
            }

            while not done:
                state_np, event_type, time_delta, time_now = preprocess_features(observation)
                state = torch.tensor([state_np], device=device, dtype=torch.float)
                event_type = torch.tensor([event_type], device=device, dtype=torch.long)
                time_delta = torch.tensor([time_delta], device=device, dtype=torch.float)

                eval_data['times'].append(time_now)
                eval_data['events'].append(event_type.item())

                q_values = model(state, event_type, time_delta)
                action = q_values.max(1)[1].view(1, 1)

                observation, reward, done, truncated, _ = env.step(action.item())
                eval_data['rewards'].append(reward)
                episode_reward += reward

                writer.add_scalar(f'Evaluation/Timeline/Event_{i}',
                                  event_type.item(),
                                  global_step=int(time_now))
                writer.add_scalar(f'Evaluation/Timeline/Reward_{i}',
                                  reward,
                                  global_step=int(time_now))

                if truncated:
                    break

            rewards.append(episode_reward)
            productions.append(observation['info']['produce_tons'])

            print(f"\nEvaluation Episode {i} stats:")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Production: {observation['info']['produce_tons']:.2f}")

    writer.close()

    return {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_production': np.mean(productions),
        'std_production': np.std(productions)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time-Aware DQN Mining Environment")
    parser.add_argument("--env_config", type=str,
                        default="../../openmines/src/conf/north_pit_mine.json",
                        help="环境配置文件路径")
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES,
                        help="训练回合数")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS,
                        help="每个回合的最大步数")
    parser.add_argument("--eval", action="store_true",
                        help="运行评估模式")
    parser.add_argument("--model_path", type=str,
                        help="用于评估的模型路径")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="评估回合数")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="预训练模型路径")

    args = parser.parse_args()

    if args.eval and args.model_path:
        # 评估模式
        env = MineEnv.make(args.env_config, log=False, ticks=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        checkpoint = torch.load(args.model_path)
        state_dim = checkpoint['config']['state_dim']
        action_dims = checkpoint['config']['action_dims']

        model = DQN(state_dim, action_dims).to(device)
        model.load_state_dict(checkpoint['policy_net_state_dict'])

        # 运行评估
        results = evaluate_model(model, env, num_episodes=args.eval_episodes)

        print("\nEvaluation Results:")
        print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Average Production: {results['avg_production']:.2f} ± {results['std_production']:.2f}")

        env.close()
    else:
        # 训练模式
        train_dqn(args)
