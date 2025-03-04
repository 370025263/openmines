# 导入必要的库
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from tqdm import tqdm
import random
import string
import json
import argparse
from datetime import datetime

# 禁用警告以加快执行速度
import warnings
warnings.filterwarnings('ignore')

# 导入您的环境
from openmines.src.utils.rl_env import MineEnv

# 超参数定义
GAMMA = 0.99
# 在超参数定义区域添加
GAE_LAMBDA = 0.95  # GAE lambda参数

CLIP_EPSILON = 0.2
CRITIC_DISCOUNT = 0.5
ENTROPY_BETA = 0.01
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
PPO_EPOCHS = 10
BATCH_SIZE = 64
MAX_STEPS = 1000
NUM_PROCESSES = 7
NUM_UPDATES = 1000
MAX_GRAD_NORM = 0.5

# 生成唯一运行标识符的函数
def generate_run_id():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{timestamp}_{random_string}"

# 生成唯一颜色的函数
def generate_run_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# 定义具有独立Actor和Critic网络的神经网络模型
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dims):
        super(ActorCritic, self).__init__()
        self.actor_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.actor_heads = nn.ModuleList([
            nn.Linear(64, dim) for dim in action_dims
        ])
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, event_type, action=None):
        encoded_state = self.actor_encoder(state)
        action_logits = self.actor_heads[event_type](encoded_state)
        # action_probs = nn.functional.softmax(action_logits, dim=-1)  # 由于Categorical分布会自动进行softmax，因此不需要手动进行
        assert torch.isfinite(action_logits).all(), f"NaN detected in action_logits: {action_logits}"
        dist = Categorical(action_logits)
        if action is None:
            # 如果没有指定动作，则从分布中采样
            action = dist.sample()
        value = self.critic(state)
        return action, dist.log_prob(action), dist.entropy(), value, dist  #  action_probs

    def evaluate(self, state, action, event_type):
        encoded_state = self.actor_encoder(state)
        if event_type.dim() > 0:
            action_logits = torch.stack([self.actor_heads[et.item()](es) for et, es in zip(event_type, encoded_state)])
        else:
            action_logits = self.actor_heads[event_type.item()](encoded_state)
        action_probs = nn.functional.softmax(action_logits, dim=-1)
        assert torch.isfinite(action_probs).all(), f"NaN detected in action_probs: {action_probs}"
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        value = self.critic(state)
        return action_logprobs, value, dist_entropy

    def actor_parameters(self):
        return list(self.actor_encoder.parameters()) + list(self.actor_heads.parameters())

    def critic_parameters(self):
        return self.critic.parameters()


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        self.event_types = []
        self.action_probs = []
        self.total_production = 0
        self.next_value = None  # 添加next_value属性

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]
        del self.event_types[:]
        del self.action_probs[:]
        self.total_production = 0
        self.next_value = None

    def compute_gae(self):
        # 转换为tensor以提高效率
        values_tensor = torch.tensor(self.values + [self.next_value])
        rewards_tensor = torch.tensor(self.rewards)
        masks = torch.tensor([not done for done in self.is_terminals])

        # 预分配advantages数组
        advantages = torch.zeros_like(rewards_tensor)
        lastgaelam = 0

        for t in reversed(range(len(self.rewards))):
            next_mask = masks[t]
            nextvalue = values_tensor[t + 1]

            delta = rewards_tensor[t] + GAMMA * nextvalue * next_mask - values_tensor[t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * next_mask * lastgaelam

        returns = advantages + values_tensor[:-1]
        return returns.numpy()

    def get(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze(-1)

        # 使用GAE计算returns和advantages
        returns = torch.tensor(self.compute_gae())
        advantages = returns - values

        # 标准化优势
        if args.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        event_types = torch.tensor(self.event_types, dtype=torch.long)
        action_probs = torch.stack(self.action_probs)

        return states, actions, log_probs, returns, advantages, event_types, action_probs

# PPO算法，具有独立的Actor和Critic优化器
class PPO:
    def __init__(self, state_dim, action_dims, device):
        self.device = device
        self.policy = ActorCritic(state_dim, action_dims).to(device)
        self.actor_optimizer = optim.Adam(self.policy.actor_parameters(), lr=ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.policy.critic_parameters(), lr=CRITIC_LEARNING_RATE)
        self.policy_old = ActorCritic(state_dim, action_dims).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.action_dims = action_dims

    def update(self, memories):
        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_advantages = []
        all_event_types = []
        all_action_probs = []

        for memory in memories:
            states, actions, log_probs, returns, advantages, event_types, action_probs = memory.get()
            all_states.extend(states)
            all_actions.extend(actions)
            all_log_probs.extend(log_probs)
            all_returns.extend(returns)
            all_advantages.extend(advantages)
            all_event_types.extend(event_types)
            all_action_probs.extend(action_probs)

        all_states = torch.stack(all_states).to(self.device)
        all_actions = torch.stack(all_actions).to(self.device)
        all_log_probs = torch.stack(all_log_probs).to(self.device)
        all_returns = torch.tensor(all_returns).to(self.device)
        all_advantages = torch.tensor(all_advantages).to(self.device)
        all_event_types = torch.stack(all_event_types).to(self.device)
        all_action_probs = torch.stack(all_action_probs).to(self.device)

        # 标准化优势函数，提高训练稳定性
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            for index in range(0, len(all_states), BATCH_SIZE):
                states = all_states[index:index + BATCH_SIZE]
                actions = all_actions[index:index + BATCH_SIZE]
                old_log_probs = all_log_probs[index:index + BATCH_SIZE]
                returns = all_returns[index:index + BATCH_SIZE]
                advantages = all_advantages[index:index + BATCH_SIZE]
                event_types = all_event_types[index:index + BATCH_SIZE]

                new_log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions, event_types)

                ratio = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values.squeeze(-1), returns)
                entropy_loss = dist_entropy.mean()

                # 更新Actor
                self.actor_optimizer.zero_grad()
                (actor_loss - ENTROPY_BETA * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.policy.actor_parameters(), MAX_GRAD_NORM)
                self.actor_optimizer.step()

                # 更新Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.critic_parameters(), MAX_GRAD_NORM)
                self.critic_optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        avg_action_probs = []
        for event_type in range(3):
            event_mask = all_event_types == event_type
            if event_mask.sum() > 0:
                avg_probs = all_action_probs[event_mask].mean(dim=0).cpu().numpy()
                avg_action_probs.append(avg_probs[:self.action_dims[event_type]])
            else:
                avg_action_probs.append(np.zeros(self.action_dims[event_type]))

        return avg_action_probs

# 特征预处理函数
def preprocess_features(observation):
    time_delta = float(observation['info']['delta_time'])
    time_now = float(observation['info']['time'])

    event_name = observation['event_name']
    if event_name == "init":
        event_type = [1, 0, 0]
        action_space_n = observation['info']['load_num']
    elif event_name == "haul":
        event_type = [0, 1, 0]
        action_space_n = observation['info']['unload_num']
    else:
        event_type = [0, 0, 1]
        action_space_n = observation['info']['load_num']

    truck_location: list = observation['the_truck_status']['truck_location_onehot']  # [1,M+N+1]
    # print("event_type", event_type)
    # print("truck_location", truck_location)

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
    # print("road_dist", road_dist)
    # print("road_traffic", road_traffic)

    state = np.concatenate([order_and_position.squeeze(), truck_features, target_features, road_dist, road_traffic,
                            road_jam])  # ])  # 3+M+N+1,2,3(M+N),(M+(M+N)*2)*3
    # state = np.concatenate([order_and_position.squeeze()])
    assert not np.isnan(state).any(), f"NaN detected in state: {state}"
    assert not np.isnan(time_delta), f"NaN detected in time_delta: {time_delta}"
    assert not np.isnan(time_now), f"NaN detected in time_now: {time_now}"

    event_type_index = event_type.index(1)

    return state, event_type_index

# 收集轨迹的工作进程函数（修改了奖励的处理）
def collect_trajectory(env_config, policy, device, process_id, trajectory_queue):
    env = MineEnv.make(env_config, log=False, ticks=False)
    memory = Memory()

    observation, _ = env.reset(seed=int(time.time() + process_id))
    state, event_type = preprocess_features(observation)
    episode_reward = 0
    for step in range(MAX_STEPS):
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action, log_prob, _, value, action_probs = policy.act(state_tensor, event_type)

        observation, reward, done, truncated, _ = env.step(action.item())
        next_state, next_event_type = preprocess_features(observation)

        # 不对奖励进行缩放
        memory.states.append(state_tensor.detach().cpu())
        memory.actions.append(action.detach().cpu())
        memory.log_probs.append(log_prob.detach().cpu())
        memory.rewards.append(reward)  # 保留原始奖励
        memory.is_terminals.append(done)
        memory.values.append(value.detach().cpu())
        memory.event_types.append(event_type)
        memory.action_probs.append(action_probs.detach().cpu())

        if done or truncated:
            break

        state = next_state
        event_type = next_event_type
        episode_reward += reward  # 累积奖励

    memory.total_production = observation['info']['produce_tons']  # 记录总产量

    # 如果未结束，使用最后的价值估计作为下一个价值
    with torch.no_grad():
        next_state_tensor = torch.FloatTensor(next_state).to(device)
        _, _, _, next_value, _ = policy.act(next_state_tensor, next_event_type)
        memory.next_value = next_value.item()

    trajectory_queue.put((memory, episode_reward, step + 1))
    env.close()

# 主函数
def main(args):
    run_id = generate_run_id()
    run_color = generate_run_color()

    log_dir = os.path.join("runs", run_id)
    os.makedirs(log_dir, exist_ok=True)

    # 保存运行配置
    config = {
        "run_id": run_id,
        "run_color": run_color,
        "env_config": args.env_config,
        "num_updates": args.num_updates,
        "num_processes": args.num_processes,
        "max_steps": args.max_steps,
        "actor_learning_rate": ACTOR_LEARNING_RATE,
        "critic_learning_rate": CRITIC_LEARNING_RATE,
        "gamma": GAMMA,
        "clip_epsilon": CLIP_EPSILON,
        "ppo_epochs": PPO_EPOCHS,
        "batch_size": BATCH_SIZE,
    }

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    env = MineEnv.make(args.env_config, log=False, ticks=False)
    observation, _ = env.reset(seed=42)
    state, _ = preprocess_features(observation)
    state_dim = len(state)
    action_dims = [
        observation['info']['load_num'],  # 对于 init
        observation['info']['unload_num'],  # 对于 haul
        observation['info']['load_num']  # 对于 unhaul
    ]
    env.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_agent = PPO(state_dim, action_dims, device)

    writer = SummaryWriter(log_dir)

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    trajectory_queue = manager.Queue()

    total_progress = tqdm(total=args.num_updates, desc=f'Training Progress (Run ID: {run_id})')

    # 创建动作的颜色（可选）
    action_colors = [
        [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(dim)]
        for dim in action_dims
    ]

    event_types = ['Init', 'Haul', 'Unhaul']

    total_production = 0


    for update in range(args.num_updates):
        processes = []

        # learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / args.num_updates
            ppo_agent.actor_optimizer.lr = ACTOR_LEARNING_RATE * frac
            ppo_agent.critic_optimizer.lr = CRITIC_LEARNING_RATE * frac


        # ROLLOUT
        for i in range(args.num_processes):
            p = mp.Process(target=collect_trajectory,
                           args=(args.env_config, ppo_agent.policy_old, device, i, trajectory_queue))
            p.start()
            processes.append(p)

        memories = []
        episode_rewards = []
        episode_lengths = []
        episode_productions = []
        for p in processes:
            p.join()

        for _ in range(args.num_processes):
            memory, episode_reward, episode_length = trajectory_queue.get()
            memories.append(memory)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_productions.append(memory.total_production)

        assert len(memories) > 0, "No trajectories collected"

        avg_action_probs = ppo_agent.update(memories)

        avg_reward = sum(episode_rewards) / args.num_processes
        avg_length = sum(episode_lengths) / args.num_processes

        # 计算此更新的总产量
        update_production = sum(episode_productions) / args.num_processes
        production_increase = update_production - total_production
        total_production = update_production

        writer.add_scalar('Average Reward', avg_reward, update)
        writer.add_scalar('Average Episode Length', avg_length, update)
        writer.add_scalar('Total Production', total_production, update)
        writer.add_scalar('Production Increase', production_increase, update)

        # 记录平均动作概率
        for event_type, probs, colors in zip(event_types, avg_action_probs, action_colors):
            action_prob_dict = {f'Action {i}': prob for i, prob in enumerate(probs)}
            writer.add_scalars(f'Avg Action Probabilities/{event_type}', action_prob_dict, update)

        total_progress.update(1)
        total_progress.set_postfix({'Avg Reward': f'{avg_reward:.2f}', 'Avg Length': f'{avg_length:.2f}', 'Total Production': f'{total_production:.2f}'})

    total_progress.close()
    writer.close()
    print(f"\nTraining completed. Run ID: {run_id}, Color: {run_color}")
    print(f"Final Total Production: {total_production}")

    # 保存最终模型
    torch.save(ppo_agent.policy.state_dict(), os.path.join(log_dir, "final_model.pth"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PPO Mining Environment Trainer")
    parser.add_argument("--env_config", type=str, default="../../openmines/src/conf/north_pit_mine.json",
                        help="Path to environment configuration file")
    parser.add_argument("--num_updates", type=int, default=NUM_UPDATES, help="Number of updates to perform")
    parser.add_argument("--num_processes", type=int, default=NUM_PROCESSES, help="Number of parallel processes to use")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS, help="Maximum number of steps per episode")
    args = parser.parse_args()

    main(args)
