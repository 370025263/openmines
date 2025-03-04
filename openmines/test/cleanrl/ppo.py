# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "mine/Mine-v1"   # mine/Mine-v0(multiprocessing, SyncVectorEnv) mine/Mine-v1(threading, AsyncVectorEnv)
    """the id of the environment"""
    mine_config: str = "../../src/conf/north_pit_mine.json"
    """the config file of the mine environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-3
    """the learning rate of the optimizer"""
    num_envs: int = 50  # 4
    """the number of parallel game environments"""
    num_steps: int = 700  # 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 50
    """the number of mini-batches"""
    update_epochs: int = 12
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.1
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # 添加检查点相关参数
    checkpoint_dir: str = "checkpoints"
    """检查点保存目录"""
    save_interval: int = 100
    """每N个iteration保存一次模型"""
    keep_checkpoint_max: int = 5
    """保留最近的N个检查点"""
    checkpoint_path: Optional[str] = None
    """加载特定检查点的路径"""
    save_best_only: bool = False
    """是否只保存最佳模型"""

    # 添加指导损失相关参数（原有的线性衰减参数仍保留，但不再使用）
    guide_initial_value: float = 0.1
    guide_final_value: float = 0.0
    guide_start_decay_step: int = 3_000_000
    guide_decay_steps: int = 500_000

    # ====== 新增的产量阈值 + 教师权重因子 ======
    teacher_acceptable_tons: float = 12000
    """当 produce_tons >= 这个数值，视为模型已足够好，可将教师loss系数置 0"""
    teacher_alpha: float = 0.5
    """当未达标时，教师系数使用 alpha * (1 - c_teacher)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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
        # 提取环境信息
        self.load_sites_num = envs.env_fns[0]().config['load_sites'].__len__()
        self.dump_sites_num = envs.env_fns[0]().config['dump_sites'].__len__()
        self.max_action_dim = max(self.load_sites_num, self.dump_sites_num)

        # 获取观察空间的维度
        self.obs_shape = 194  #212 #@self._get_obs_shape(212)

        # 共享特征提取层
        self.shared_net = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh()
        )

        # 为不同事件类型创建独立的actor heads
        self.actor_heads = nn.ModuleDict({
            '0': layer_init(nn.Linear(64, self.load_sites_num), std=0.01),
            '1': layer_init(nn.Linear(64, self.dump_sites_num), std=0.01),
            '2': layer_init(nn.Linear(64, self.load_sites_num), std=0.01)
        })
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, sug_action=None):
        features = self.shared_net(x)
        batch_size = x.shape[0]
        event_types = torch.argmax(x[:, :3], dim=1)

        logits = torch.zeros((batch_size, self.max_action_dim), device=x.device)

        for i in range(batch_size):
            event_type = str(event_types[i].item())
            head_output = self.actor_heads[event_type](features[i:i + 1])
            logits[i] = head_output

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), (
            probs.log_prob(sug_action) if sug_action is not None else None
        )

    @property
    def device(self):
        """获取模型所在设备"""
        return next(self.parameters()).device

    def save(self, path):
        """保存模型"""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """加载模型"""
        self.load_state_dict(torch.load(path, map_location=self.device))


# 保留原有的 GuidanceDecay 类，但不再在训练中实际调用
class GuidanceDecay:
    """指导系数衰减器
        coef
    0.1 |XXXXXX
        |      X
        |       X
        |        X
        |         X
    0.0 |          XXXX
        +---------------
        0    3M   4M  steps
    """

    def __init__(self,
                 initial_value: float = 0.1,
                 final_value: float = 0.0,
                 start_decay_step: int = 3_000_000,
                 decay_steps: int = 500_000):
        self.initial_value = initial_value
        self.final_value = final_value
        self.start_decay_step = start_decay_step
        self.decay_steps = decay_steps

    def get_value(self, global_step: int) -> float:
        """获取当前步骤的指导系数 (线性衰减示例)"""
        if global_step < self.start_decay_step:
            return self.initial_value

        if global_step >= self.start_decay_step + self.decay_steps:
            return self.final_value

        decay_progress = (global_step - self.start_decay_step) / self.decay_steps
        return self.initial_value + (self.final_value - self.initial_value) * decay_progress


class CheckpointManager:
    """检查点管理器"""

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
        """保存检查点"""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'reward': reward,
            'args': self.args,
            'info': additional_info or {}
        }

        # 保存当前检查点
        if not self.args.save_best_only:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_{iteration:07d}.pt'
            )
            torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with reward: {reward:.2f}")

        self._cleanup_old_checkpoints()
        return checkpoint_path

    def load_checkpoint(self,
                        agent: nn.Module,
                        optimizer: Optional[optim.Optimizer] = None,
                        path: Optional[str] = None) -> tuple:
        """加载检查点"""
        if path is None:
            # 尝试加载最新的检查点
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
        """清理旧的检查点，只保留最近的N个"""
        if self.args.save_best_only:
            return

        checkpoints = self._get_checkpoints()
        if len(checkpoints) > self.args.keep_checkpoint_max:
            for checkpoint in checkpoints[:-self.args.keep_checkpoint_max]:
                os.remove(checkpoint)

    def _get_checkpoints(self):
        """获取所有检查点，按迭代次数排序"""
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('checkpoint_') and f.endswith('.pt')
        ]
        checkpoints = [os.path.join(self.checkpoint_dir, f) for f in checkpoints]
        return sorted(checkpoints)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # 创建检查点管理器
    checkpoint_manager = CheckpointManager(args, run_name)

    # 原先的指南衰减器类依然保留，但不做实际调用
    guide_decay = GuidanceDecay(
        initial_value=args.guide_initial_value,
        final_value=args.guide_final_value,
        start_decay_step=args.guide_start_decay_step,
        decay_steps=args.guide_decay_steps
    )

    # 加载检查点(如果指定)
    start_iteration = 0
    best_reward = float('-inf')
    if args.checkpoint_path:
        start_iteration, best_reward = checkpoint_manager.load_checkpoint(
            agent, optimizer, args.checkpoint_path
        )

    # ALGO Logic: Storage setup
    obs_shape = agent.obs_shape
    obs = torch.zeros((args.num_steps, args.num_envs, obs_shape)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    sug_actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # 记录全局步数 & 产量
    global_step = 0
    start_time = time.time()
    latest_produce_tons = 0.0  # 新增：用于存储最近一次episode结束时的产量

    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.FloatTensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    enable_guide = True  # 新增：是否启用教师指导
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # 保存检查点
        if iteration % args.save_interval == 0:
            is_best = False
            current_reward = info["episode"]["r"] if "episode" in info else 0
            if current_reward > best_reward:
                best_reward = current_reward
                is_best = True

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

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                current_sug_action = torch.tensor(infos.get("sug_action", [-1] * args.num_envs)).to(device)
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs, sug_action=current_sug_action)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob
            sug_actions[step] = current_sug_action

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.FloatTensor(next_obs).to(device)
            next_done = torch.FloatTensor(next_done).to(device)

            # 如果episode结束，记录产量
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        produce_tons = info["produce_tons"]
                        if isinstance(produce_tons, (float, int)):
                            avg_tons = produce_tons
                        else:
                            avg_tons = sum(produce_tons) / len(produce_tons)

                        latest_produce_tons = avg_tons  # 记录产量
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}",
                              f"episodic_length={info['episode']['l']}, produce_tons={avg_tons}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        writer.add_scalar("charts/produce_tons", avg_tons, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_sug_actions = sug_actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, sug_logprob = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions.long()[mb_inds],
                    b_sug_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # KL metrics
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
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

                # ========== 自适应教师指导系数的核心修改部分 ==========
                valid_sug_mask = (b_sug_actions[mb_inds] >= 0).float()
                with torch.no_grad():
                    if sug_logprob is not None:
                        # 计算 c_teacher: 在本 mini-batch 上对教师动作的平均概率
                        probs = torch.exp(sug_logprob) * valid_sug_mask
                        sum_probs = probs.sum()
                        valid_count = valid_sug_mask.sum()
                        if valid_count > 0:
                            c_teacher = (sum_probs / valid_count).item()
                        else:
                            c_teacher = 0.0
                    else:
                        c_teacher = 0.0

                # 根据产量阈值 & c_teacher 确定最终 guide_coef
                if enable_guide:
                    if latest_produce_tons >= args.teacher_acceptable_tons:
                        guide_coef = 0.0
                        enable_guide = False  # 达标后不再启用教师指导
                    else:
                        guide_coef = args.teacher_alpha * (1.0 - c_teacher)
                else:
                    guide_coef = 0.0


                # 计算指导损失
                if sug_logprob is not None:
                    guide_loss = -(sug_logprob * valid_sug_mask).mean()
                else:
                    guide_loss = 0.0

                # 合并到总loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + guide_coef * guide_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        # 记录 guide_loss 和 guide_coef 方便观察
        if isinstance(guide_loss, torch.Tensor):
            writer.add_scalar("losses/guide_loss", guide_loss.item(), global_step)
        else:
            writer.add_scalar("losses/guide_loss", 0.0, global_step)
        writer.add_scalar("charts/guide_coef", guide_coef, global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # 保存最终模型
    checkpoint_manager.save_checkpoint(
        agent,
        optimizer,
        args.num_iterations,
        best_reward,
        additional_info={'final': True}
    )

    envs.close()
    writer.close()
