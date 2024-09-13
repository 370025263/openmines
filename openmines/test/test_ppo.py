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
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
PPO_EPOCHS = 10
BATCH_SIZE = 64
MAX_STEPS = 1000
NUM_PROCESSES = 7
NUM_UPDATES = 1000
MAX_GRAD_NORM = 0.5

# Function to generate unique run identifier
def generate_run_id():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{timestamp}_{random_string}"

# Function to generate unique color
def generate_run_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Neural network model with separate Actor and Critic networks
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

    def act(self, state, event_type):
        encoded_state = self.actor_encoder(state)
        action_logits = self.actor_heads[event_type](encoded_state)
        action_probs = nn.functional.softmax(action_logits, dim=-1)
        assert torch.isfinite(action_probs).all(), f"NaN detected in action_probs: {action_probs}"
        dist = Categorical(action_probs)
        action = dist.sample()
        value = self.critic(state)
        return action, dist.log_prob(action), dist.entropy(), value, action_probs

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

# Memory class to handle variable-length episodes
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
        self.next_value = 0
        self.total_production = 0

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

    def compute_gae(self, next_value):
        values = self.values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + GAMMA * values[step + 1] - values[step]
            gae = delta + GAMMA * GAE_LAMBDA * gae
            returns.insert(0, gae + values[step])
        return returns

    def get(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze(-1)
        returns = torch.tensor(self.compute_gae(self.next_value))
        advantages = returns - values
        event_types = torch.tensor(self.event_types, dtype=torch.long)
        action_probs = torch.stack(self.action_probs)

        return states, actions, log_probs, returns, advantages, event_types, action_probs

# PPO algorithm with separate Actor and Critic optimizers
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
        all_returns = torch.stack(all_returns).to(self.device)
        all_advantages = torch.stack(all_advantages).to(self.device)
        all_event_types = torch.stack(all_event_types).to(self.device)
        all_action_probs = torch.stack(all_action_probs).to(self.device)

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

                self.actor_optimizer.zero_grad()
                (actor_loss - ENTROPY_BETA * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.policy.actor_parameters(), MAX_GRAD_NORM)
                self.actor_optimizer.step()

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

# Feature preprocessing function
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

# Worker function for collecting trajectories
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

        memory.states.append(state_tensor.detach().cpu())
        memory.actions.append(action.detach().cpu())
        memory.log_probs.append(log_prob.detach().cpu())
        memory.rewards.append(reward / 10000)  # Divide reward by 10000
        memory.is_terminals.append(done)
        memory.values.append(value.detach().cpu())
        memory.event_types.append(event_type)
        memory.action_probs.append(action_probs.detach().cpu())

        if done or truncated:
            break

        state = next_state
        event_type = next_event_type
        episode_reward += reward / 10000  # Accumulate scaled reward

    memory.total_production = observation['info']['produce_tons']  # Record total production
    if done or truncated:
        next_value = 0
    else:
        next_state_tensor = torch.FloatTensor(next_state).to(device)
        with torch.no_grad():
            _, _, _, next_value, _ = policy.act(next_state_tensor, next_event_type)
        next_value = next_value.item()
    memory.next_value = next_value
    trajectory_queue.put((memory, episode_reward, step + 1))
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
        "actor_learning_rate": ACTOR_LEARNING_RATE,
        "critic_learning_rate": CRITIC_LEARNING_RATE,
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
    state, _ = preprocess_features(observation)
    state_dim = len(state)
    action_dims = [
        observation['info']['load_num'],  # for init
        observation['info']['unload_num'],  # for haul
        observation['info']['load_num']  # for unhaul
    ]
    env.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_agent = PPO(state_dim, action_dims, device)

    writer = SummaryWriter(log_dir)

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    trajectory_queue = manager.Queue()

    total_progress = tqdm(total=args.num_updates, desc=f'Training Progress (Run ID: {run_id})')

    # Create colors for actions (optional)
    action_colors = [
        [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(dim)]
        for dim in action_dims
    ]

    event_types = ['Init', 'Haul', 'Unhaul']

    total_production = 0

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
        
        # Calculate total production for this update
        update_production = sum(episode_productions) / args.num_processes
        production_increase = update_production - total_production
        total_production = update_production

        writer.add_scalar('Average Reward', avg_reward, update)
        writer.add_scalar('Average Episode Length', avg_length, update)
        writer.add_scalar('Total Production', total_production, update)
        writer.add_scalar('Production Increase', production_increase, update)

        # Log average action probabilities
        for event_type, probs, colors in zip(event_types, avg_action_probs, action_colors):
            action_prob_dict = {f'Action {i}': prob for i, prob in enumerate(probs)}
            writer.add_scalars(f'Avg Action Probabilities/{event_type}', action_prob_dict, update)

        total_progress.update(1)
        total_progress.set_postfix({'Avg Reward': f'{avg_reward:.2f}', 'Avg Length': f'{avg_length:.2f}', 'Total Production': f'{total_production:.2f}'})

    total_progress.close()
    writer.close()
    print(f"\nTraining completed. Run ID: {run_id}, Color: {run_color}")
    print(f"Final Total Production: {total_production}")

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
