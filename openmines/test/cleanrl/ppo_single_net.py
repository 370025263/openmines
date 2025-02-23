# ppo_single_dt.py
# ppo_with_tuned_params.py
# ------------------------------------------------
# CleanRL-style PPO with custom hyperparameters
# 由你的超参数
# {'learning_rate': 0.0002275497790550207,
#  'ent_coef': 0.014319441675556047,
#  'gamma': 0.9967378999910834,
#  'clip_coef': 0.19502127394785151,
#  'gae_lambda': 0.9930159847502031,
#  'update_epochs': 11,
#  'max_grad_norm': 0.363605168165705,
#  'hidden_size': 256}
# 组成

import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import openmines_gym


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    # Algorithm-specific arguments
    env_id: str = "mine/Mine-v1"
    mine_config: str = "/home/weiyu/stone/openmines_project/openmines/openmines/src/conf/north_pit_mine.json"
    total_timesteps: int = 10000000

    # ------ 以下是目标超参数 ------
    learning_rate: float = 0.002275497790550207
    ent_coef: float = 0.014319441675556047
    gamma: float = 0.9967378999910834
    clip_coef: float = 0.19502127394785151
    gae_lambda: float = 0.9930159847502031
    update_epochs: int = 4
    max_grad_norm: float = 0.363605168165705

    # 其余保持不变
    num_envs: int = 50
    num_steps: int = 1400
    anneal_lr: bool = True
    norm_adv: bool = True
    vf_coef: float = 0.5
    target_kl: float = None # 防止训练不稳定
    clip_vloss: bool = True
    num_minibatches: int = 4

    # checkpoint
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 100
    keep_checkpoint_max: int = 5
    checkpoint_path: Optional[str] = None
    save_best_only: bool = True
    save_best_only_params: bool = True # 只保存best model的参数

    # teacher guide (保留但不使用)
    guide_initial_value: float = 0.1
    guide_final_value: float = 0.0
    guide_start_decay_step: int = 3_000_000
    guide_decay_steps: int = 500_000
    teacher_acceptable_tons: float = 12000
    teacher_alpha: float = 0 #0.5

    # runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    # 添加网络宽度参数
    hidden_size: int = 256
    """neural network hidden size"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, config_file=args.mine_config, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, config_file=args.mine_config)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.load_sites_num = 5
        self.dump_sites_num =  5
        self.max_action_dim = max(self.load_sites_num, self.dump_sites_num)

        self.obs_shape = 194
        hidden_size = args.hidden_size  # 使用配置的隐藏层大小

        self.shared_net = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )

        self.actor = layer_init(nn.Linear(hidden_size, self.max_action_dim), std=0.01)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)), 
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, sug_action=None):
        features = self.shared_net(x)
        logits = self.actor(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        value = self.critic(x)

        sug_logprob = None
        if sug_action is not None:
            valid_mask = (sug_action >= 0)
            if valid_mask.any():
                sug_action_clamped = torch.clamp(sug_action, min=0)
                sug_logprob = probs.log_prob(sug_action_clamped)
            else:
                sug_logprob = None

        return action, probs.log_prob(action), probs.entropy(), value, sug_logprob

    @property
    def device(self):
        return next(self.parameters()).device

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))


class GuidanceDecay:
    def __init__(self, initial_value=0.1, final_value=0.0, start_decay_step=3_000_000, decay_steps=500_000):
        self.initial_value = initial_value
        self.final_value = final_value
        self.start_decay_step = start_decay_step
        self.decay_steps = decay_steps

    def get_value(self, global_step: int) -> float:
        if global_step < self.start_decay_step:
            return self.initial_value
        if global_step >= self.start_decay_step + self.decay_steps:
            return self.final_value
        decay_progress = (global_step - self.start_decay_step) / self.decay_steps
        return self.initial_value + (self.final_value - self.initial_value) * decay_progress


class CheckpointManager:
    def __init__(self, args: Args, exp_name: str):
        self.args = args
        self.exp_name = exp_name
        self.checkpoint_dir = os.path.join(args.checkpoint_dir, exp_name)
        self.best_reward = float('-inf')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self,
                        agent: nn.Module,
                        optimizer: optim.Optimizer,
                        iteration: int,
                        reward: float,
                        is_best: bool = False,
                        additional_info: Dict = None) -> str:
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'reward': reward,
            'args': self.args,
            'info': additional_info or {}
        }

        if not self.args.save_best_only:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_{iteration:07d}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            if self.args.save_best_only_params: # 只保存参数
                torch.save(agent.state_dict(), best_path)
            else: # 保存完整checkpoint
                torch.save(checkpoint, best_path)
            print(f"New best model saved with reward: {reward:.2f}")

        self._cleanup_old_checkpoints()
        return checkpoint_path

    def load_checkpoint(self,
                        agent: nn.Module,
                        optimizer: Optional[optim.Optimizer] = None,
                        path: Optional[str] = None) -> tuple:
        if path is None:
            checkpoints = self._get_checkpoints()
            if not checkpoints:
                return 0, float('-inf')
            path = checkpoints[-1]

        try:
            print(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location=agent.device)
            agent.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['iteration'], checkpoint['reward']
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 0, float('-inf')

    def _cleanup_old_checkpoints(self):
        if self.args.save_best_only:
            return
        checkpoints = self._get_checkpoints()
        if len(checkpoints) > self.args.keep_checkpoint_max:
            for ckpt in checkpoints[:-self.args.keep_checkpoint_max]:
                os.remove(ckpt)

    def _get_checkpoints(self):
        files = [f for f in os.listdir(self.checkpoint_dir)
                 if f.startswith('checkpoint_') and f.endswith('.pt')]
        files = [os.path.join(self.checkpoint_dir, f) for f in files]
        return sorted(files)


if __name__ == "__main__":
    args = tyro.cli(Args)
    # compute batch sizes
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.env_id}__{args.exp_name}__"\
               f"s{args.seed}__"\
               f"lr{args.learning_rate:.2e}__"\
               f"e{args.ent_coef:.2e}__"\
               f"g{args.gamma:.3f}__"\
               f"c{args.clip_coef:.2f}__"\
               f"l{args.gae_lambda:.3f}__"\
               f"ep{args.update_epochs}__"\
               f"gr{args.max_grad_norm:.2f}__"\
               f"hs{args.hidden_size}__"\
               f"ns{args.num_steps}__"\
               f"ne{args.num_envs}__"\
               f"mb{args.num_minibatches}__"\
               f"t{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
        "only discrete action space is supported"

    agent = Agent(envs).to(device)

    # 使用指定learning rate
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    checkpoint_manager = CheckpointManager(args, run_name)
    guide_decay = GuidanceDecay(
        initial_value=args.guide_initial_value,
        final_value=args.guide_final_value,
        start_decay_step=args.guide_start_decay_step,
        decay_steps=args.guide_decay_steps
    )

    start_iteration = 0
    best_reward = float('-inf')
    if args.checkpoint_path:
        start_iteration, best_reward = checkpoint_manager.load_checkpoint(
            agent, optimizer, args.checkpoint_path
        )

    obs_shape = agent.obs_shape
    obs = torch.zeros((args.num_steps, args.num_envs, obs_shape), device=device)
    time_deltas = torch.zeros((args.num_steps, args.num_envs), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    sug_actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    latest_produce_tons = 0.0

    env_seeds = [random.randint(0, 2 ** 31 - 1) for _ in range(args.num_envs)]
    next_obs, infos = envs.reset(seed=env_seeds)
    next_obs = torch.FloatTensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs, device=device)

    enable_guide = True

    # 初始化episode rewards追踪
    if not hasattr(envs, 'episode_rewards'):
        envs.episode_rewards = [0.0] * args.num_envs  # 每个环境的累计奖励
        envs.episode_counts = [0] * args.num_envs     # 每个环境的步数计数
    if not hasattr(envs, 'episode_produce_tons'):
        envs.episode_produce_tons = []

    for iteration in range(start_iteration + 1, args.num_iterations + 1):
        # 1) Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # 2) rollout data collection
        # if iteration % args.save_interval == 0: # 移除save_interval条件限制
        is_best = False
        current_reward = 0
        if "episode" in infos and "r" in infos["episode"]:
            current_reward = infos["episode"]["r"]
        if current_reward > best_reward:
            best_reward = current_reward
            is_best = True

        if iteration % args.save_interval == 0 or is_best: # 周期性checkpoint和best model checkpoint都保存
            checkpoint_manager.save_checkpoint(
                agent,
                optimizer,
                iteration,
                current_reward,
                is_best,
                additional_info={
                    'global_step': global_step,
                    'time_elapsed': time.time() - start_time
                }
            )

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                current_sug_action = torch.tensor(infos.get("sug_action", [-1] * args.num_envs), device=device)
                action, logprob, _, value, _ = agent.get_action_and_value(
                    next_obs, sug_action=current_sug_action
                )
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob
            sug_actions[step] = current_sug_action

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_obs = torch.FloatTensor(next_obs_np).to(device)
            next_done = torch.FloatTensor(np.logical_or(terminations, truncations)).to(device)
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            time_deltas[step] = torch.FloatTensor(next_obs_np[:, 4]).to(device)

            # 更新每个环境的累计奖励
            for idx in range(args.num_envs):
                envs.episode_rewards[idx] += reward[idx]
                envs.episode_counts[idx] += 1

            # 记录每个env的终止状态产出
            for idx, (term, trunc) in enumerate(zip(terminations, truncations)):
                if term or trunc:  # 当环境终止时
                    # 记录产出吨数
                    produce_tons = sum(np.exp(infos["final_observation"][idx][-5:]) - 1)
                    envs.episode_produce_tons.append(produce_tons)
                    episode_reward = envs.episode_rewards[idx]
                    episode_length = envs.episode_counts[idx]
                    print(f"global_step={global_step}, env_id={idx}, produce_tons={produce_tons}, reward={episode_reward}")
                    writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)
                        
                    # 重置计数器
                    envs.episode_rewards[idx] = 0.0
                    envs.episode_counts[idx] = 0

            # 在rollout结束时记录平均产出和奖励
            if step == args.num_steps - 1:
                # 计算平均产出
                if len(envs.episode_produce_tons) > 0:
                    avg_tons = sum(envs.episode_produce_tons) / len(envs.episode_produce_tons)
                    writer.add_scalar("charts/avg_produce_tons_per_rollout", avg_tons, global_step)
                    envs.episode_produce_tons = []
                


        # 3) GAE + returns
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                gamma_dt = (args.gamma ** time_deltas[t]).to(device)
                lambda_dt = (args.gae_lambda ** time_deltas[t]).to(device)

                delta = rewards[t] + gamma_dt * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma_dt * lambda_dt * nextnonterminal * lastgaelam

        returns = advantages + values

        # 4) Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_sug_actions = sug_actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 5) Policy & value update
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                (
                    _,
                    newlogprob,
                    entropy,
                    newvalue,
                    sug_logprob,
                ) = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions.long()[mb_inds],
                    b_sug_actions.long()[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                # teacher guidance (保留可用)
                valid_sug_mask = (b_sug_actions[mb_inds] >= 0).float()
                with torch.no_grad():
                    if sug_logprob is not None:
                        probs = torch.exp(sug_logprob) * valid_sug_mask
                        sum_probs = probs.sum()
                        valid_count = valid_sug_mask.sum()
                        if valid_count > 0:
                            c_teacher = (sum_probs / valid_count).item()
                        else:
                            c_teacher = 0.0
                    else:
                        c_teacher = 0.0

                guide_coef = 0.0
                if enable_guide:
                    if latest_produce_tons >= args.teacher_acceptable_tons:
                        guide_coef = 0.0
                        enable_guide = False
                    else:
                        guide_coef = args.teacher_alpha * (1.0 - c_teacher)

                if sug_logprob is not None:
                    guide_loss = -(sug_logprob * valid_sug_mask).mean()
                else:
                    guide_loss = 0.0

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + guide_coef * guide_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # 6) logging
        explained_var = 0.0
        with torch.no_grad():
            y_pred = b_values.cpu().numpy()
            y_true = b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if isinstance(guide_loss, torch.Tensor):
            writer.add_scalar("losses/guide_loss", guide_loss.item(), global_step)
        else:
            writer.add_scalar("losses/guide_loss", 0.0, global_step)
        writer.add_scalar("charts/guide_coef", guide_coef, global_step)

        sps = int(global_step / (time.time() - start_time))
        print(f"SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    # 最后保存
    checkpoint_manager.save_checkpoint(
        agent,
        optimizer,
        args.num_iterations,
        best_reward,
        additional_info={'final': True}
    )

    envs.close()
    writer.close()
