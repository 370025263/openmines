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
from collections import deque
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# 禁用警告以加快执行速度
import warnings

warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter
from openmines.src.utils.rl_env import MineEnv

# ===== 超参数定义 =====
GAMMA = 0.99
TIME_SCALE = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_STEPS = 1000
NUM_EPISODES = 500
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500


def generate_run_id():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{timestamp}_{random_string}"


def generate_run_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


class DQN(nn.Module):
    def __init__(self, state_dim, action_dims):
        super(DQN, self).__init__()
        self.input_dim = state_dim + 1

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)

        self.output_layers = nn.ModuleList([
            nn.Linear(128, dim) for dim in action_dims
        ])

        self.time_attention = nn.Linear(1, 128)

    def forward(self, state, event_type, time_delta):
        time_feature = time_delta.unsqueeze(-1)
        combined_state = torch.cat([state, time_feature], dim=1)
        time_attention = torch.sigmoid(self.time_attention(time_delta.unsqueeze(-1)))

        x = torch.relu(self.fc1(combined_state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        x = x * time_attention

        if state.shape[0] == 1:
            q_values = self.output_layers[event_type.item()](x)
        else:
            q_values = torch.stack([
                self.output_layers[et.item()](h)
                for et, h in zip(event_type, x)
            ])
        return q_values


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, event_type, action, reward, next_state, next_event_type, done, time_delta):
        self.memory.append((state, event_type, action, reward, next_state,
                            next_event_type, done, time_delta))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def preprocess_features(observation):
    time_delta = float(observation['info']['delta_time'])
    time_now = float(observation['info']['time'])

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
    assert not np.isnan(time_delta), f"NaN detected in time_delta: {time_delta}"
    assert not np.isnan(time_now), f"NaN detected in time_now: {time_now}"

    return state, event_type, time_delta, time_now


def create_episode_figures(episode_data, episode, event_types):
    # 创建事件时间线图
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    for evt_type in range(3):
        times = [t for e, t in zip(episode_data['events'], episode_data['times']) if e == evt_type]
        if times:
            ax1.scatter(times, [evt_type] * len(times), label=event_types[evt_type], alpha=0.6)
    ax1.set_yticks(range(3))
    ax1.set_yticklabels(event_types)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Event Type')
    ax1.set_title(f'Episode {episode} Event Timeline')
    ax1.legend()

    # 创建奖励时间线图
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(episode_data['times'], episode_data['rewards'], 'b-', alpha=0.6)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Reward')
    ax2.set_title(f'Episode {episode} Reward Timeline')

    return fig1, fig2


def create_episode_figures(episode_data, episode, event_types):
    try:
        # 直接记录原始数据而不是创建matplotlib图表
        data = {
            'event_counts': {},
            'time_intervals': [],
            'reward_stats': {}
        }

        # 统计事件计数
        for evt_type in range(3):
            count = sum(1 for e in episode_data['events'] if e == evt_type)
            data['event_counts'][event_types[evt_type]] = count

        # 计算时间间隔
        if len(episode_data['times']) > 1:
            data['time_intervals'] = np.diff(episode_data['times']).tolist()

        # 计算奖励统计
        rewards = np.array(episode_data['rewards'])
        data['reward_stats'] = {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards))
        }

        return data
    except Exception as e:
        print(f"Error in create_episode_figures: {e}")
        return None

def train_dqn(args):
    run_id = generate_run_id()
    run_color = generate_run_color()

    log_dir = os.path.join("runs", run_id)
    os.makedirs(log_dir, exist_ok=True)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(state_dim, action_dims).to(device)
    target_net = DQN(state_dim, action_dims).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)
    writer = SummaryWriter(log_dir)

    steps_done = 0

    def select_action(state, event_type, time_delta):
        nonlocal steps_done
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        np.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        if random.random() < eps_threshold:
            return torch.tensor([[random.randrange(action_dims[event_type.item()])]],
                                device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                q_values = policy_net(state, event_type, time_delta)
                return q_values.max(1)[1].view(1, 1)

    episode_rewards = []
    episode_lengths = []
    event_types = ['Init', 'Haul', 'Unhaul']
    total_progress = tqdm(range(NUM_EPISODES), desc=f'Training DQN (Run ID: {run_id})')

    total_production = 0

    for episode in total_progress:
        env = MineEnv.make(args.env_config, log=False, ticks=False)
        observation, _ = env.reset(seed=episode)
        state_np, event_type, time_delta, time_now = preprocess_features(observation)

        state = torch.tensor([state_np], device=device, dtype=torch.float)
        event_type = torch.tensor([event_type], device=device, dtype=torch.long)
        time_delta_tensor = torch.tensor([time_delta], device=device, dtype=torch.float)

        total_reward = 0
        step_count = 0
        action_counts = {0: {}, 1: {}, 2: {}}

        # 收集当前回合数据
        episode_data = {
            'times': [],
            'events': [],
            'rewards': [],
            'steps': []
        }

        for t in range(MAX_STEPS):
            # 记录当前状态
            episode_data['times'].append(time_now)
            episode_data['events'].append(event_type.item())
            episode_data['steps'].append(t)

            # 选择并执行动作
            action = select_action(state, event_type, time_delta_tensor)
            action_item = action.item()
            observation, reward, done, truncated, _ = env.step(action_item)

            # 计算调整后的奖励
            adjusted_reward = reward * np.exp(-TIME_SCALE * time_delta)
            episode_data['rewards'].append(adjusted_reward)
            total_reward += adjusted_reward

            reward_tensor = torch.tensor([adjusted_reward], device=device, dtype=torch.float32)

            # 处理下一个状态
            next_state_np, next_event_type, next_time_delta, next_time_now = preprocess_features(observation)
            next_state = torch.tensor([next_state_np], device=device, dtype=torch.float)
            next_event_type = torch.tensor([next_event_type], device=device, dtype=torch.long)
            next_time_delta_tensor = torch.tensor([next_time_delta], device=device, dtype=torch.float)

            memory.push(state, event_type, action, reward_tensor, next_state,
                        next_event_type, done, time_delta_tensor)

            # 更新动作统计
            et = event_type.item()
            if action_item in action_counts[et]:
                action_counts[et][action_item] += 1
            else:
                action_counts[et][action_item] = 1

            # 状态转移
            state = next_state
            event_type = next_event_type
            time_delta_tensor = next_time_delta_tensor
            time_now = next_time_now
            step_count += 1

            # 经验回放训练
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = list(zip(*transitions))

                batch_state = torch.cat(batch[0])
                batch_event_type = torch.cat(batch[1])
                batch_action = torch.cat(batch[2])
                batch_reward = torch.cat(batch[3])
                batch_next_state = torch.cat(batch[4])
                batch_next_event_type = torch.cat(batch[5])
                batch_done = torch.tensor(batch[6], device=device)
                batch_time_delta = torch.cat(batch[7])

                # 计算当前Q值
                current_q_values = policy_net(batch_state, batch_event_type,
                                              batch_time_delta).gather(1, batch_action)

                # 计算目标Q值
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                with torch.no_grad():
                    non_final_mask = ~batch_done
                    non_final_next_states = batch_next_state[non_final_mask]
                    non_final_next_event_types = batch_next_event_type[non_final_mask]
                    non_final_next_time_deltas = batch_time_delta[non_final_mask]

                    if len(non_final_next_states) > 0:
                        next_q_values = target_net(non_final_next_states,
                                                   non_final_next_event_types,
                                                   non_final_next_time_deltas)
                        next_state_values[non_final_mask] = next_q_values.max(1)[0]

                # 计算目标值
                gamma_t = GAMMA ** batch_time_delta.squeeze().float()
                expected_q_values = (next_state_values * gamma_t) + batch_reward.squeeze()

                # 计算损失和优化
                loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
                optimizer.step()

                writer.add_scalar('Training/Loss', loss.item(), steps_done)
                writer.add_scalar('Training/Time_Delta', batch_time_delta.mean().item(), steps_done)

            if done or truncated:
                break
        # 每个回合结束后的可视化处理
        # 1. 将时序数据写入TensorBoard
        for step, (time_val, event_val, reward_val) in enumerate(zip(
                episode_data['times'], episode_data['events'], episode_data['rewards'])):
            writer.add_scalar(f'Episode_{episode}/Timeline/Event',
                              event_val,
                              global_step=step)
            writer.add_scalar(f'Episode_{episode}/Timeline/Reward',
                              reward_val,
                              global_step=step)
            writer.add_scalar(f'Episode_{episode}/Timeline/Time',
                              time_val,
                              global_step=step)

        # 2. 创建并保存图表
        stats = create_episode_figures(episode_data, episode, event_types)
        if stats:
            # 记录事件计数
            writer.add_scalars('Statistics/Event_Counts',
                               stats['event_counts'],
                               episode)

            # 记录时间序列数据
            for step, (time_val, event_val, reward_val) in enumerate(zip(
                    episode_data['times'],
                    episode_data['events'],
                    episode_data['rewards'])):
                writer.add_scalar(f'Timeline/Event_Type',
                                  event_val,
                                  global_step=step)
                writer.add_scalar(f'Timeline/Reward',
                                  reward_val,
                                  global_step=step)
                writer.add_scalar(f'Timeline/Time',
                                  time_val,
                                  global_step=step)

            # 记录时间间隔
            if stats['time_intervals']:
                writer.add_histogram('Statistics/Time_Intervals',
                                     np.array(stats['time_intervals']),
                                     episode)
                writer.add_scalar('Statistics/Avg_Time_Interval',
                                  np.mean(stats['time_intervals']),
                                  episode)

            # 记录奖励统计
            for key, value in stats['reward_stats'].items():
                writer.add_scalar(f'Rewards/{key}', value, episode)

        # 3. 记录事件统计
        event_counts = {event_types[i]: episode_data['events'].count(i) for i in range(3)}
        writer.add_scalars('Statistics/Event_Counts', event_counts, episode)

        # 4. 记录时间间隔统计
        if len(episode_data['times']) > 1:
            time_intervals = np.diff(episode_data['times'])
            writer.add_histogram('Statistics/Time_Intervals', time_intervals, episode)
            writer.add_scalar('Statistics/Avg_Time_Interval',
                              np.mean(time_intervals), episode)

        # 5. 记录每种事件类型的时间间隔
        for evt_type in range(3):
            evt_times = [t for e, t in zip(episode_data['events'], episode_data['times'])
                         if e == evt_type]
            if len(evt_times) > 1:
                evt_intervals = np.diff(evt_times)
                writer.add_histogram(f'Event_Intervals/{event_types[evt_type]}',
                                     evt_intervals, episode)
                writer.add_scalar(f'Statistics/Avg_{event_types[evt_type]}_Interval',
                                  np.mean(evt_intervals), episode)

        # 6. 记录奖励统计
        writer.add_scalar('Rewards/Total', total_reward, episode)
        writer.add_scalar('Rewards/Average', np.mean(episode_data['rewards']), episode)
        writer.add_histogram('Rewards/Distribution', np.array(episode_data['rewards']), episode)

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        env.close()

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        update_production = observation['info']['produce_tons']
        production_increase = update_production - total_production
        total_production = update_production

        writer.add_scalar('Production/Total', total_production, episode)
        writer.add_scalar('Production/Increase', production_increase, episode)

        for et in action_counts:
            total_actions = sum(action_counts[et].values())
            if total_actions > 0:
                action_freq = {f'Action {a}': count / total_actions
                               for a, count in action_counts[et].items()}
                writer.add_scalars(f'Actions/{event_types[et]}_Distribution',
                                   action_freq, episode)

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        np.exp(-1. * steps_done / EPS_DECAY)
        writer.add_scalar('Parameters/Epsilon', eps_threshold, episode)

        if episode % 10 == 0:
            avg_reward_last_10 = np.mean(episode_rewards[-10:])
            avg_length = step_count
            total_progress.set_postfix({
                'Avg Reward': f'{avg_reward_last_10:.2f}',
                'Avg Length': f'{avg_length:.2f}',
                'Total Production': f'{total_production:.2f}'
            })

            # 打印调试信息
            print(f"\nEpisode {episode} stats:")
            print(f"Events recorded: {len(episode_data['events'])}")
            print(f"Times recorded: {len(episode_data['times'])}")
            print(f"Rewards recorded: {len(episode_data['rewards'])}")
            print(f"Event counts: {event_counts}")
            print(f"Average reward: {np.mean(episode_data['rewards']):.2f}")

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
                'steps': []
            }

            step = 0
            while not done:
                state_np, event_type, time_delta, time_now = preprocess_features(observation)
                state = torch.tensor([state_np], device=device, dtype=torch.float)
                event_type = torch.tensor([event_type], device=device, dtype=torch.long)
                time_delta = torch.tensor([time_delta], device=device, dtype=torch.float)

                eval_data['times'].append(time_now)
                eval_data['events'].append(event_type.item())
                eval_data['steps'].append(step)

                q_values = model(state, event_type, time_delta)
                action = q_values.max(1)[1].view(1, 1)

                observation, reward, done, truncated, _ = env.step(action.item())
                eval_data['rewards'].append(reward)
                episode_reward += reward

                writer.add_scalar(f'Evaluation/Timeline/Event_{i}',
                                  event_type.item(), step)
                writer.add_scalar(f'Evaluation/Timeline/Reward_{i}',
                                  reward, step)

                step += 1
                if truncated:
                    break

            # 记录评估统计
            stats = create_episode_figures(eval_data, i, ['Init', 'Haul', 'Unhaul'])
            if stats:
                writer.add_scalars(f'Evaluation/Episode_{i}/Event_Counts',
                                   stats['event_counts'], i)
                for key, value in stats['reward_stats'].items():
                    writer.add_scalar(f'Evaluation/Episode_{i}/Rewards/{key}',
                                      value, i)

            rewards.append(episode_reward)
            productions.append(observation['info']['produce_tons'])

            writer.add_scalar('Evaluation/Episode_Reward', episode_reward, i)
            writer.add_scalar('Evaluation/Production',
                              observation['info']['produce_tons'], i)

            print(f"\nEvaluation Episode {i} stats:")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Production: {observation['info']['produce_tons']:.2f}")
            if stats:
                print(f"Event counts: {stats['event_counts']}")

    writer.close()

    return {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_production': np.mean(productions),
        'std_production': np.std(productions),
        'all_rewards': rewards,
        'all_productions': productions
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time-Aware DQN Mining Environment")
    parser.add_argument("--env_config", type=str,
                        default="../../openmines/src/conf/north_pit_mine.json",
                        help="Path to environment configuration file")
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES,
                        help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS,
                        help="Maximum number of steps per episode")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation mode")
    parser.add_argument("--model_path", type=str,
                        help="Path to saved model for evaluation")

    args = parser.parse_args()

    if args.eval and args.model_path:
        env = MineEnv.make(args.env_config, log=False, ticks=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(args.model_path)
        state_dim = checkpoint['config']['state_dim']
        action_dims = checkpoint['config']['action_dims']

        model = DQN(state_dim, action_dims).to(device)
        model.load_state_dict(checkpoint['policy_net_state_dict'])

        results = evaluate_model(model, env)
        print("\nEvaluation Results:")
        print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Average Production: {results['avg_production']:.2f} ± {results['std_production']:.2f}")

        env.close()
    else:
        train_dqn(args)
