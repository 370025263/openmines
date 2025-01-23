import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import openmines_gym  # 你的自定义环境

@dataclass
class DQNArgs:
    exp_name: str = "dueling_dqn"
    """实验名称"""
    seed: int = 1
    """随机种子"""
    torch_deterministic: bool = True
    """是否让 CUDA 在有核算时保持确定性"""
    cuda: bool = True
    """是否使用 GPU"""
    track: bool = False
    """是否使用 Weights & Biases 进行跟踪"""
    wandb_project_name: str = "cleanRL"
    """W&B project 名称"""
    wandb_entity: str = None
    """W&B entity (team)"""
    capture_video: bool = False
    """是否录制环境视频 (保存在 `videos/` 下)"""

    # 环境相关
    env_id: str = "mine/Mine-v1"
    """环境 ID (Gym注册名)"""
    mine_config: str = "../../src/conf/north_pit_mine.json"
    """OpenMines 自定义环境配置"""
    num_envs: int = 4
    """并行环境数量 (vec env)"""

    # 训练超参
    total_timesteps: int = 500000
    """训练总步数(与环境交互次数)"""
    buffer_size: int = 100000
    """Replay Buffer 最大容量"""
    start_e: float = 1.0
    """初始 epsilon (探索率)"""
    end_e: float = 0.1
    """最终 epsilon (探索率)"""
    exploration_fraction: float = 0.1
    """探索率从 start_e 衰减至 end_e 所占 total_timesteps 的比例"""
    learning_rate: float = 2.5e-4
    """学习率"""
    gamma: float = 0.99
    """折扣因子"""
    batch_size: int = 64
    """训练时每次采样的小批量大小"""
    learning_starts: int = 10000
    """在此步数之前，仅与环境交互不训练"""
    train_frequency: int = 4
    """每隔多少环境步，就进行一次训练"""
    target_network_frequency: int = 1000
    """目标网络(用于稳定训练)的更新间隔"""
    max_grad_norm: float = 10.0
    """梯度裁剪最大范数"""

    # 检查点
    checkpoint_dir: str = "checkpoints"
    """检查点保存目录"""
    save_interval: int = 10000
    """每隔多少步保存一次模型"""
    keep_checkpoint_max: int = 5
    """最多保留多少个旧的检查点文件"""
    save_best_only: bool = False
    """是否只保存回报最好的检查点"""
    checkpoint_path: Optional[str] = None
    """从指定路径加载检查点"""

    # 其他
    wandb_tags: Optional[str] = None
    """W&B 的自定义标签, 以逗号分隔"""


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str, mine_config: str):
    """
    生成可选视频录制的环境 thunk，用于后续创建 VectorEnv。
    在这里加入 RecordEpisodeStatistics，
    使得环境在 done=True 时将 'episode' 信息注入 info 里。
    """
    def thunk():
        # 你可以将render_mode修改为需要的视频输出，否则默认无需录制
        env = gym.make(env_id, config_file=mine_config)
        # 包装器
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            # 若需要视频录制，可以加这一行
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: True)
        return env
    return thunk


class ReplayBuffer:
    """循环队列式的 Replay Buffer，用于存储交互转移 (s, a, r, s', done)。"""
    def __init__(self, obs_dim, buffer_size, num_envs, device):
        self.device = device
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.num_envs = num_envs

        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.next_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, actions, rewards, next_obs, dones):
        """
        添加一批 (env_step) 转移，obs/next_obs 的 shape = (num_envs, obs_dim)。
        actions, rewards, dones 的 shape = (num_envs,)。
        """
        for i in range(self.num_envs):
            self.obs[self.ptr] = obs[i]
            self.actions[self.ptr] = actions[i]
            self.rewards[self.ptr] = rewards[i]
            self.next_obs[self.ptr] = next_obs[i]
            self.dones[self.ptr] = dones[i]
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        """
        随机采样 batch_size 条转移，并返回 (obs, actions, rewards, next_obs, dones) 的 Tensor。
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.tensor(self.obs[idxs], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions[idxs], dtype=torch.long, device=self.device)
        rewards = torch.tensor(self.rewards[idxs], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(self.next_obs[idxs], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[idxs], dtype=torch.float32, device=self.device)
        return obs, actions, rewards, next_obs, dones


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN：公共特征层 + value 分支 + advantage 分支。
    假设输入维度为 (obs_dim)，输出维度为离散动作数量。
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_dim)

    def forward(self, x):
        hidden = self.feature(x)
        value = self.value_stream(hidden)            # shape: [batch_size, 1]
        advantage = self.advantage_stream(hidden)    # shape: [batch_size, action_dim]
        # dueling 的组合方式：Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class CheckpointManager:
    """简单的检查点管理器，保存/加载权重与优化器状态。"""
    def __init__(self, args: DQNArgs, exp_name: str):
        self.args = args
        self.exp_name = exp_name
        self.checkpoint_dir = os.path.join(args.checkpoint_dir, exp_name)
        self.best_reward = float('-inf')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self,
                        q_net: nn.Module,
                        optimizer: optim.Optimizer,
                        global_step: int,
                        reward: float,
                        is_best: bool = False,
                        additional_info: Dict[str, Any] = None) -> str:
        """保存检查点"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{global_step:09d}.pt")
        checkpoint = {
            'global_step': global_step,
            'model_state_dict': q_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'reward': reward,
            'args': self.args,
            'info': additional_info or {}
        }

        # 保存当前检查点(如果非只保留 best)
        if not self.args.save_best_only:
            torch.save(checkpoint, checkpoint_path)

        # 如果是最佳模型，额外保存 best_model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] New best model saved with reward: {reward:.2f}")

        self._cleanup_old_checkpoints()
        return checkpoint_path

    def load_checkpoint(self,
                        q_net: nn.Module,
                        optimizer: Optional[optim.Optimizer] = None,
                        path: Optional[str] = None) -> int:
        """加载检查点，返回该 checkpoint 存储的 global_step (如果解析失败则返回 0)。"""
        if path is None:
            checkpoints = self._get_checkpoints()
            if not checkpoints:
                return 0
            path = checkpoints[-1]

        try:
            print(f"[Checkpoint] Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location=q_net.parameters().__next__().device)
            q_net.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_reward = checkpoint.get('reward', float('-inf'))
            return checkpoint['global_step']
        except Exception as e:
            print(f"[Checkpoint] Error loading checkpoint: {e}")
            return 0

    def _cleanup_old_checkpoints(self):
        """清理旧的检查点，只保留最近的 N 个"""
        if self.args.save_best_only:
            return
        checkpoints = self._get_checkpoints()
        if len(checkpoints) > self.args.keep_checkpoint_max:
            for ckpt in checkpoints[:-self.args.keep_checkpoint_max]:
                os.remove(ckpt)

    def _get_checkpoints(self):
        """获取所有检查点文件（不含 best_model），按 global_step 升序排序"""
        files = [f for f in os.listdir(self.checkpoint_dir)
                 if f.startswith('checkpoint_') and f.endswith('.pt')]
        full_paths = [os.path.join(self.checkpoint_dir, f) for f in files]
        full_paths.sort(key=lambda p: int(p.split('_')[-1].split('.')[0]))
        return full_paths


def main(args: DQNArgs):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # 初始化日志
    if args.track:
        import wandb
        wandb_tags = args.wandb_tags.split(",") if args.wandb_tags else None
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            tags=wandb_tags,
            save_code=True
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" % (
                        "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
                    ))

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 创建并行环境（VectorEnv），内部使用 RecordEpisodeStatistics
    envs = gym.vector.AsyncVectorEnv([
        make_env(args.env_id, i, args.capture_video, run_name, args.mine_config)
        for i in range(args.num_envs)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "只支持离散动作空间"

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n

    q_network = DuelingQNetwork(obs_dim, act_dim).to(device)
    target_network = DuelingQNetwork(obs_dim, act_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    rb = ReplayBuffer(obs_dim, args.buffer_size, args.num_envs, device)

    checkpoint_manager = CheckpointManager(args, run_name)
    global_step = 0
    if args.checkpoint_path:
        loaded_step = checkpoint_manager.load_checkpoint(q_network, optimizer, args.checkpoint_path)
        global_step = loaded_step

    max_exploration_steps = int(args.total_timesteps * args.exploration_fraction)
    def get_epsilon(step: int):
        if step >= max_exploration_steps:
            return args.end_e
        slope = (args.end_e - args.start_e) / max_exploration_steps
        return max(args.end_e, args.start_e + slope * step)

    obs, infos = envs.reset(seed=args.seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    start_time = time.time()

    # 用于统计回报
    episode_rewards = []
    best_reward = checkpoint_manager.best_reward

    while global_step < args.total_timesteps:
        epsilon = get_epsilon(global_step)
        if np.random.rand() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            with torch.no_grad():
                q_values = q_network(obs)
                actions = q_values.argmax(dim=1).cpu().numpy()

        next_obs, rewards, dones, truncs, infos = envs.step(actions)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

        rb.add(
            obs.cpu().numpy(),
            actions,
            rewards,
            next_obs,
            dones | truncs
        )

        obs = next_obs_t
        global_step += args.num_envs

        # 如果某个并行环境结束episode，RecordEpisodeStatistics就会往对应的info里注入 "episode"
        # 在VectorEnv里，infos是个元组或列表，可以逐一遍历
        for i, info_item in enumerate(infos):
            if "episode" in info_item:  # 说明这个子环境完成了一个episode
                ep_r = info_item["episode"]["r"]
                ep_l = info_item["episode"]["l"]
                # 有些版本是一个np数组(当并行环境同时结束)，也可能是float，这里做一下处理
                if isinstance(ep_r, np.ndarray):
                    ep_r = ep_r.mean()
                if isinstance(ep_l, np.ndarray):
                    ep_l = ep_l.mean()
                episode_rewards.append(ep_r)
                writer.add_scalar("charts/episodic_return", ep_r, global_step)
                writer.add_scalar("charts/episodic_length", ep_l, global_step)
                print(f"[Env={i}] step={global_step}, episodic_return={ep_r:.2f}, length={ep_l}")

        # DQN训练
        if global_step > args.learning_starts and (global_step % args.train_frequency == 0):
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = rb.sample(args.batch_size)
            with torch.no_grad():
                # 基础 DQN
                target_q = target_network(next_obs_batch).max(dim=1)[0]
                td_target = rew_batch + args.gamma * target_q * (1.0 - done_batch)

            current_q = q_network(obs_batch).gather(1, act_batch.unsqueeze(-1)).squeeze(-1)
            loss = nn.MSELoss()(current_q, td_target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
            optimizer.step()

            writer.add_scalar("losses/td_loss", loss.item(), global_step)

        # 更新 target 网络
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

        # 定期保存模型
        if (args.save_interval > 0) and (global_step % args.save_interval == 0):
            mean_recent_reward = np.mean(episode_rewards[-10:]) if episode_rewards else -9999
            is_best = mean_recent_reward > best_reward
            if is_best:
                best_reward = mean_recent_reward

            checkpoint_manager.save_checkpoint(
                q_net=q_network,
                optimizer=optimizer,
                global_step=global_step,
                reward=mean_recent_reward,
                is_best=is_best,
                additional_info={
                    "time_elapsed": time.time() - start_time
                }
            )

        # 写SPS
        if global_step % 1000 == 0:
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            if args.track:
                import wandb
                wandb.log({"charts/SPS": sps}, step=global_step)

    # 训练结束，保存最终模型
    mean_recent_reward = np.mean(episode_rewards[-10:]) if episode_rewards else -9999
    is_best = mean_recent_reward > best_reward
    checkpoint_manager.save_checkpoint(
        q_net=q_network,
        optimizer=optimizer,
        global_step=global_step,
        reward=mean_recent_reward,
        is_best=is_best,
        additional_info={'final': True}
    )

    envs.close()
    writer.close()
    print("Training finished!")


if __name__ == "__main__":
    args = tyro.cli(DQNArgs)
    main(args)
