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

# Disable warnings to speed up execution
import warnings

warnings.filterwarnings('ignore')

# Import your environment
from openmines.src.utils.rl_env import MineEnv

# Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
CRITIC_DISCOUNT = 0.5
ENTROPY_BETA = 0.01
LEARNING_RATE = 3e-4
PPO_EPOCHS = 10
BATCH_SIZE = 64
MAX_STEPS = 1000  # Maximum number of steps per episode
NUM_PROCESSES = 7
NUM_UPDATES = 1000
MAX_GRAD_NORM = 0.5


# Function to generate a unique run identifier
def generate_run_id():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{timestamp}_{random_string}"


# Function to generate a unique color for the run
def generate_run_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


# Neural network model
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(state)

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


# Memory class to handle variable-length episodes
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]

    def compute_gae(self, next_value):
        values = self.values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + GAMMA * values[step + 1] * (1 - self.is_terminals[step]) - values[step]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - self.is_terminals[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def get(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze(-1)
        returns = torch.tensor(self.compute_gae(values[-1].item()))
        advantages = returns - values

        return states, actions, log_probs, returns, advantages


# PPO algorithm
class PPO:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def update(self, memories):
        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_advantages = []

        for memory in memories:
            states, actions, log_probs, returns, advantages = memory.get()
            all_states.extend(states)
            all_actions.extend(actions)
            all_log_probs.extend(log_probs)
            all_returns.extend(returns)
            all_advantages.extend(advantages)

        all_states = torch.stack(all_states).to(self.device)
        all_actions = torch.stack(all_actions).to(self.device)
        all_log_probs = torch.stack(all_log_probs).to(self.device)
        all_returns = torch.stack(all_returns).to(self.device)
        all_advantages = torch.stack(all_advantages).to(self.device)

        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            for index in range(0, len(all_states), BATCH_SIZE):
                states = all_states[index:index + BATCH_SIZE]
                actions = all_actions[index:index + BATCH_SIZE]
                old_log_probs = all_log_probs[index:index + BATCH_SIZE]
                returns = all_returns[index:index + BATCH_SIZE]
                advantages = all_advantages[index:index + BATCH_SIZE]

                new_log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)

                ratio = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values, returns.unsqueeze(-1))
                entropy_loss = -dist_entropy.mean()

                loss = actor_loss + CRITIC_DISCOUNT * critic_loss + ENTROPY_BETA * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

# Feature preprocessing function
def preprocess_features(observation):
    truck_features = np.array([
        observation['the_truck_status']['truck_load'] / observation['the_truck_status']['truck_capacity'],
        observation['the_truck_status']['truck_speed'] / 100,
        observation['the_truck_status']['truck_cycle_time'] / 1000
    ])

    road_features = np.concatenate([
        np.array(list(observation['cur_road_status']['charging2load']['truck_count'].values())) / 10,
        np.array(list(observation['cur_road_status']['charging2load']['distances'].values())) / 10,
        np.array(list(observation['cur_road_status']['charging2load']['truck_jam_count'].values())) / 5,
        np.array(list(observation['cur_road_status']['charging2load']['repair_count'].values())) / 5,
    ])

    target_features = np.concatenate([
        np.array(observation['target_status']['queue_lengths']) / 10,
        np.array(observation['target_status']['capacities']) / 1000,
        np.array(observation['target_status']['est_wait']) / 100,
        np.array(observation['target_status']['produced_tons']) / 1000,
        np.array(observation['target_status']['service_counts']) / 100,
    ])

    state = np.concatenate([truck_features, road_features, target_features])

    return state


# Worker function for collecting trajectories
def collect_trajectory(env_config, policy, device, process_id, trajectory_queue):
    env = MineEnv.make(env_config, log=False, ticks=False)
    memory = Memory()

    observation, _ = env.reset(seed=int(time.time() + process_id))  # Use different seeds for each process
    state = preprocess_features(observation)
    episode_reward = 0
    for step in range(MAX_STEPS):
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action, log_prob, _, value = policy.act(state_tensor)

        observation, reward, done, truncated, _ = env.step(action.item())
        next_state = preprocess_features(observation)

        memory.states.append(state_tensor.detach().cpu())
        memory.actions.append(action.detach().cpu())
        memory.log_probs.append(log_prob.detach().cpu())
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        memory.values.append(value.detach().cpu())

        state = next_state
        episode_reward += reward

        if done or truncated:
            break

    trajectory_queue.put((memory, episode_reward, step + 1))  # Return the number of steps taken
    env.close()


# Main function
def main(args):
    run_id = generate_run_id()
    run_color = generate_run_color()

    log_dir = os.path.join("runs", run_id)
    os.makedirs(log_dir, exist_ok=True)

    # Save run configuration
    config = {
        "run_id": run_id,
        "run_color": run_color,
        "env_config": args.env_config,
        "num_updates": args.num_updates,
        "num_processes": args.num_processes,
        "max_steps": args.max_steps,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_epsilon": CLIP_EPSILON,
        "ppo_epochs": PPO_EPOCHS,
        "batch_size": BATCH_SIZE,
    }

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    env = MineEnv.make(args.env_config, log=False, ticks=False)
    observation, _ = env.reset(seed=42)
    state = preprocess_features(observation)
    state_dim = len(state)
    action_dim = len(observation['target_status']['queue_lengths'])
    env.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_agent = PPO(state_dim, action_dim, device)

    writer = SummaryWriter(log_dir)

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    trajectory_queue = manager.Queue()

    total_progress = tqdm(total=args.num_updates, desc=f'Training Progress (Run ID: {run_id})')

    for update in range(args.num_updates):
        processes = []
        for i in range(args.num_processes):
            p = mp.Process(target=collect_trajectory,
                           args=(args.env_config, ppo_agent.policy_old, device, i, trajectory_queue))
            p.start()
            processes.append(p)

        memories = []
        episode_rewards = []
        episode_lengths = []
        for p in processes:
            p.join()

        for _ in range(args.num_processes):
            memory, episode_reward, episode_length = trajectory_queue.get()
            memories.append(memory)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        ppo_agent.update(memories)

        avg_reward = sum(episode_rewards) / args.num_processes
        avg_length = sum(episode_lengths) / args.num_processes
        writer.add_scalar('Average Reward', avg_reward, update)
        writer.add_scalar('Average Episode Length', avg_length, update)

        total_progress.update(1)
        total_progress.set_postfix({'Avg Reward': f'{avg_reward:.2f}', 'Avg Length': f'{avg_length:.2f}'})

    total_progress.close()
    writer.close()
    print(f"\nTraining completed. Run ID: {run_id}, Color: {run_color}")

    # Save the final model
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