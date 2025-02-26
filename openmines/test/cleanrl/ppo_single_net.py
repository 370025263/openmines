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
import json
import datetime
from pathlib import Path
import subprocess
import tempfile
import shutil
import sys

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
    mine_config: str = "../../src/conf/north_pit_mine.json"
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
    num_envs: int = 5
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

    r_mode: str = "reward_norm"  # 可选: "reward_norm", "return_norm", "none"
    """正则化模式选择: reward_norm - 使用预计算的统计值正则化reward, return_norm - 对每个batch的returns进行在线正则化"""


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
    def __init__(self, envs, args, norm_path=None):
        super().__init__()
        self.load_sites_num = 5
        self.dump_sites_num =  5
        self.max_action_dim = max(self.load_sites_num, self.dump_sites_num)

        self.obs_shape = 194
        # 加载正则化参数
        if norm_path is None:
            norm_path = manage_normalization_params(args=args)
        print(f"正在读取正则化参数文件: {norm_path}")
        with open(norm_path, "r") as f:
            normalization_params = json.load(f)
        # 添加维度检查
        assert len(normalization_params["state_mean"]) == self.obs_shape, \
            f"Normalization mean dimension {len(normalization_params['state_mean'])} " \
            f"does not match obs_shape {self.obs_shape}"
        assert len(normalization_params["state_std"]) == self.obs_shape, \
            f"Normalization std dimension {len(normalization_params['state_std'])} " \
            f"does not match obs_shape {self.obs_shape}"

        hidden_size = args.hidden_size

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

        # 注册状态和奖励的正则化参数
        self.register_buffer(
            "obs_mean",
            torch.tensor(normalization_params["state_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "obs_std", 
            torch.tensor(normalization_params["state_std"], dtype=torch.float32)
        )
        self.register_buffer(
            "reward_mean",
            torch.tensor(normalization_params["reward_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "reward_std",
            torch.tensor(normalization_params["reward_std"], dtype=torch.float32)
        )
        
        # 为了数值稳定性,将过小的std设置为1
        self.obs_std[self.obs_std < 1e-5] = 1.0
        if self.reward_std < 1e-5:
            self.reward_std = torch.tensor(1.0, device=self.reward_std.device)

    def normalize_obs(self, obs):
        """对观察值进行正则化"""
        return (obs - self.obs_mean) / self.obs_std
    
    def normalize_reward(self, reward):
        """对奖励进行正则化"""
        if args.r_mode == "reward_norm":  # 只在reward_norm模式下进行正则化
            return (reward - self.reward_mean) / self.reward_std
        return reward  # 其他模式返回原始reward
    
    def denormalize_reward(self, normalized_reward):
        """将正则化的奖励转换回原始尺度"""
        return normalized_reward * self.reward_std + self.reward_mean

    def get_value(self, x):
        x = self.normalize_obs(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None, sug_action=None):
        x = self.normalize_obs(x)
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

        # 获取产出吨数和步数信息
        produce_tons = additional_info.get('produce_tons', 0.0) if additional_info else 0.0
        global_step = additional_info.get('global_step', 0) if additional_info else 0
        
        # 构建包含关键指标的文件名
        filename = f'model_step{global_step:07d}_tons{produce_tons:.1f}_reward{reward:.2f}.pt'

        if not self.args.save_best_only:
            checkpoint_path = os.path.join(self.checkpoint_dir, filename)
            torch.save(checkpoint, checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        if is_best:
            # 最佳模型也使用相同的命名方式
            best_path = os.path.join(self.checkpoint_dir, f'best_{filename}')
            if self.args.save_best_only_params:
                torch.save(agent.state_dict(), best_path)
            else:
                torch.save(checkpoint, best_path)
            print(f"新的最佳模型已保存 - 步数: {global_step}, 产量: {produce_tons:.1f}, 奖励: {reward:.2f}")

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
        # 修改文件筛选方式以匹配新的命名格式
        files = [f for f in os.listdir(self.checkpoint_dir)
                if f.startswith('model_') and f.endswith('.pt')]
        files = [os.path.join(self.checkpoint_dir, f) for f in files]
        return sorted(files)


def manage_normalization_params(specified_path: Optional[str] = None, args: Optional[Args] = None) -> str:
    """
    管理正则化参数文件。
    - 首先检查指定路径或当前工作目录是否存在参数文件
    - 如果存在，询问用户是否使用（带倒计时）
    - 如果不存在或用户选择不使用，则从dqn_collector收集新数据
    
    Args:
        specified_path: 用户指定的参数文件路径
        args: 程序参数，用于获取env_id和配置文件等信息
        
    Returns:
        str: 参数文件的绝对路径
    """
    param_file_name = "normalization_params.json"
    
    # 1. 检查指定路径或当前目录是否存在参数文件
    param_file_path = None
    if specified_path:
        if os.path.isfile(specified_path):
            param_file_path = specified_path
        else:
            print(f"指定的参数文件不存在: {specified_path}")
    
    # 检查当前目录
    if param_file_path is None:
        cwd_param_file = os.path.join(os.getcwd(), param_file_name)
        if os.path.isfile(cwd_param_file):
            param_file_path = cwd_param_file
    
    # 2. 如果找到文件，询问用户是否使用（带10秒倒计时）
    use_existing_file = False
    if param_file_path:
        file_stats = os.stat(param_file_path)
        create_time = datetime.datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"发现正则化参数文件:")
        print(f"  路径: {os.path.abspath(param_file_path)}")
        print(f"  创建时间: {create_time}")
        
        # 带倒计时的输入提示
        import threading
        if os.name == 'nt':
            import msvcrt
        
        timeout = 10  # 10秒倒计时
        answer = None
        
        def input_thread():
            nonlocal answer
            answer = input(f"是否使用此文件? [Y/n] ({timeout}秒后自动选Y): ").strip().lower()
        
        def countdown_thread():
            nonlocal timeout
            while timeout > 0 and answer is None:
                sys.stdout.write(f"\r等待输入... {timeout}秒 ")
                sys.stdout.flush()
                time.sleep(1)
                timeout -= 1
            sys.stdout.write("\r" + " " * 30 + "\r")  # 清除倒计时显示
            sys.stdout.flush()
        
        input_t = threading.Thread(target=input_thread)
        input_t.daemon = True
        input_t.start()
        
        countdown_t = threading.Thread(target=countdown_thread)
        countdown_t.daemon = True
        countdown_t.start()
        
        input_t.join(timeout + 0.5)  # 给一点额外时间
        
        if answer is None:  # 超时，默认使用现有文件
            print("超时，默认使用现有参数文件")
            use_existing_file = True
        else:
            use_existing_file = answer == "" or answer == "y"
    
    # 3. 如果用户选择不使用已有文件或没有找到文件，从dqn_collector收集新数据
    if not use_existing_file:
        print("将使用dqn_collector收集新数据并生成参数文件...")
        
        # 确定系统临时目录路径
        temp_dir = tempfile.mkdtemp(prefix="normalization_params_")
        try:
            # 从args获取环境配置路径和env_id
            env_config = args.mine_config if args and hasattr(args, 'mine_config') else None
            env_id = args.env_id if args and hasattr(args, 'env_id') else "mine/Mine-v1"
            
            # 如果没有提供配置文件，尝试查找默认位置
            if not env_config or not os.path.isfile(env_config):
                # 自动查找项目根目录下的配置文件
                base_dir = Path(__file__).resolve().parents[3]  # 获取项目根目录
                default_configs = [
                    str(base_dir / "src" / "conf" / "north_pit_mine.json"),
                    str(Path(os.getcwd()).parent.parent / "src" / "conf" / "north_pit_mine.json"),
                    str(Path(os.getcwd()) / "north_pit_mine.json")
                ]
                
                for config_path in default_configs:
                    if os.path.isfile(config_path):
                        env_config = config_path
                        print(f"自动找到配置文件: {env_config}")
                        break
            
            if not env_config or not os.path.isfile(env_config):
                raise FileNotFoundError("无法找到有效的环境配置文件，请通过--mine_config指定")
            
            print(f"使用环境配置: {env_config}")
            print(f"环境ID: {env_id}")
            print("正在收集数据，这可能需要一些时间...")
            
            try:
                # 导入数据收集器
                from openmines.src.utils.rl_data_collector.dqn_collector import DataCollector
                
                # 创建数据收集器并收集数据
                collector = DataCollector(
                    env_config=env_config,
                    env_id=env_id,  # 传递env_id参数
                    episodes=3,     # 使用较少的episodes以加快速度
                    max_steps=1000
                )
                collector.collect_data()  # 使用dense奖励模式
                
                # 找到生成的参数文件
                dataset_dir = os.path.join("datasets", collector.run_id)
                new_param_file = os.path.join(dataset_dir, param_file_name)
                
                # 复制到当前目录
                param_file_path = os.path.join(os.getcwd(), param_file_name)
                shutil.copy(new_param_file, param_file_path)
                
                print(f"参数文件已生成并复制到: {param_file_path}")
                
            except Exception as e:
                print(f"收集数据时出错: {str(e)}")
                print("将尝试使用命令行方式运行收集器...")
                
                try:
                    # 备选方案：使用subprocess运行dqn_collector脚本
                    script_path = str(Path(base_dir) / "src" / "utils" / "rl_data_collector" / "dqn_collector.py")
                    if not os.path.isfile(script_path):
                        # 尝试查找相对路径
                        script_path = str(Path(os.getcwd()).parent.parent / "src" / "utils" / "rl_data_collector" / "dqn_collector.py")
                    
                    if not os.path.isfile(script_path):
                        raise FileNotFoundError(f"找不到数据收集器脚本: {script_path}")
                    
                    cmd = [
                        sys.executable, 
                        script_path, 
                        "--episodes", "3", 
                        "--max_steps", "500", 
                        "--env_config", env_config,
                        "--env_id", env_id
                    ]
                    
                    subprocess.run(cmd, check=True)
                    
                    # 查找最新生成的目录
                    datasets_dir = Path("datasets")
                    if datasets_dir.exists():
                        dirs = sorted(datasets_dir.glob("dispatch_data_*"), 
                                     key=lambda x: x.stat().st_mtime, reverse=True)
                        if dirs:
                            new_param_file = dirs[0] / param_file_name
                            param_file_path = os.path.join(os.getcwd(), param_file_name)
                            shutil.copy(str(new_param_file), param_file_path)
                            print(f"参数文件已生成并复制到: {param_file_path}")
                        else:
                            raise Exception("未找到生成的数据集目录")
                    else:
                        raise Exception("未找到datasets目录")
                    
                except Exception as sub_e:
                    print(f"使用命令行运行收集器时出错: {str(sub_e)}")
                    raise Exception("无法生成正则化参数文件") from sub_e
                    
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # 验证参数文件
    if param_file_path and os.path.isfile(param_file_path):
        try:
            with open(param_file_path, 'r') as f:
                params = json.load(f)
                print(f"成功加载正则化参数，特征维度: {params.get('feature_dims', 'unknown')}")
                return param_file_path
        except Exception as e:
            print(f"验证参数文件时出错: {str(e)}")
            raise Exception("参数文件无效") from e
    else:
        raise FileNotFoundError(f"找不到正则化参数文件: {param_file_path}")


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
               f"rm{args.r_mode}__"\
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

    # 使用正则化参数管理函数，传递args以获取env_id和配置信息
    norm_path = manage_normalization_params(args=args)
    agent = Agent(args=args, envs=envs, norm_path=norm_path).to(device)

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

    max_avg_tons = 0
    avg_tons = 0
    for iteration in range(start_iteration + 1, args.num_iterations + 1):
        # 1) Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # 2) rollout data collection
        is_best = False
        current_raw_reward = 0
        current_normalized_reward = 0  # 初始化变量
        
        if "episode" in infos and "r" in infos["episode"]:
            current_raw_reward = infos["episode"]["r"]
            current_normalized_reward = agent.normalize_reward(
                torch.tensor(current_raw_reward, device=device)
            ).item()
            
            # 记录两种奖励
            writer.add_scalar("charts/checkpoint_reward_raw", current_raw_reward, global_step)
            writer.add_scalar("charts/checkpoint_reward_normalized", current_normalized_reward, global_step)

        if avg_tons > max_avg_tons:
            max_avg_tons = max(avg_tons,max_avg_tons)
            is_best = True

        if iteration % args.save_interval == 0 or is_best:
            checkpoint_manager.save_checkpoint(
                agent,
                optimizer,
                iteration,
                current_raw_reward,  # 保存原始奖励
                is_best,
                additional_info={
                    'global_step': global_step,
                    'time_elapsed': time.time() - start_time,
                    'normalized_reward': current_normalized_reward,  # 同时记录正则化奖励
                    'produce_tons': latest_produce_tons,
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
            
            # 对奖励进行正则化
            reward_tensor = torch.tensor(reward, device=device).view(-1)
            rewards[step] = agent.normalize_reward(reward_tensor)  # 根据模式决定是否正则化
            
            time_deltas[step] = torch.FloatTensor(next_obs_np[:, 4]).to(device)

            # 更新每个环境的累计奖励 (使用原始奖励)
            for idx in range(args.num_envs):
                envs.episode_rewards[idx] += reward[idx]
                envs.episode_counts[idx] += 1

            # 记录每个env的终止状态产出
            for idx, (term, trunc) in enumerate(zip(terminations, truncations)):
                if term or trunc:  # 当环境终止时
                    # 记录产出吨数
                    produce_tons = sum(np.exp(infos["final_observation"][idx][-5:]) - 1)
                    envs.episode_produce_tons.append(produce_tons)
                    
                    # 获取原始奖励和正则化奖励
                    raw_episode_reward = envs.episode_rewards[idx]
                    normalized_episode_reward = agent.normalize_reward(
                        torch.tensor(raw_episode_reward, device=device)
                    ).item()
                    
                    episode_length = envs.episode_counts[idx]
                    
                    # 打印原始和正则化的奖励
                    print(f"global_step={global_step}, env_id={idx}")
                    print(f"  produce_tons={produce_tons:.2f}")
                    print(f"  raw_reward={raw_episode_reward:.2f}")
                    print(f"  normalized_reward={normalized_episode_reward:.2f}")
                    
                    # 分别记录原始和正则化的奖励
                    writer.add_scalar("charts/episodic_return", raw_episode_reward, global_step)
                    writer.add_scalar("charts/episodic_return_normalized", normalized_episode_reward, global_step)
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
                
                latest_produce_tons = avg_tons

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

        # 如果使用return normalization,对整个batch的returns进行正则化
        if args.r_mode == "return_norm":
            returns_mean = returns.mean()
            returns_std = returns.std(unbiased=False)
            returns = (returns - returns_mean) / (returns_std + 1e-8)

        # 记录日志时区分两种模式
        writer.add_scalar(f"charts/episodic_return_{args.r_mode}", raw_episode_reward, global_step)
        if args.r_mode == "reward_norm":
            normalized_reward = agent.normalize_reward(torch.tensor(raw_episode_reward, device=device))
            writer.add_scalar("charts/episodic_return_normalized_reward", normalized_reward, global_step)
        elif args.r_mode == "return_norm":
            # 只记录returns的统计信息,不记录normalized reward
            writer.add_scalar("charts/returns_mean", returns_mean.item(), global_step)
            writer.add_scalar("charts/returns_std", returns_std.item(), global_step)

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

    # 在训练结束时打印最终的统计信息
    print("\n训练完成!")
    print(f"最佳原始奖励: {best_reward:.2f}")
    print(f"最佳正则化奖励: {agent.normalize_reward(torch.tensor(best_reward, device=device)).item():.2f}")
