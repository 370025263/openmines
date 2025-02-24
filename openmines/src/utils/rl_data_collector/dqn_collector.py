import os
import json
import time
import numpy as np
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from multiprocessing import Queue

from openmines.src.utils.rl_env import MineEnv
from openmines.src.utils.feature_processing import preprocess_observation


class NumpyEncoder(json.JSONEncoder):
    """处理numpy数据类型的JSON编码器"""

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class DataCollector:
    def __init__(self, env_config, episodes=100, max_steps=1000):
        """初始化数据收集器"""
        self.env_config = env_config
        self.episodes = episodes
        self.max_steps = max_steps
        self.dataset = []
        self.all_states = []
        self.all_rewards = []  # 添加rewards列表用于存储所有奖励值
        
        # 读取配置文件以获取所有调度器
        with open(env_config, 'r') as f:
            config = json.load(f)
        self.dispatchers = config['dispatcher']['type']
        self.sim_time = config['sim_time']  # 获取模拟时间
        
        # 生成唯一的数据集ID
        self.run_id = self._generate_run_id()
        
        # 创建输出目录
        self.output_dir = os.path.join("datasets", self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_run_id(self):
        """生成唯一的运行ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"dispatch_data_{timestamp}"

    def collect_data(self):
        """为每个调度器收集调度决策数据"""
        for dispatcher in tqdm(self.dispatchers, desc="Processing dispatchers"):
            print(f"\n收集调度器 {dispatcher} 的数据...")
            
            # 更新环境配置中的调度器
            with open(self.env_config, 'r') as f:
                config = json.load(f)
            config['dispatcher']['type'] = [dispatcher]
            
            # 创建临时配置文件
            temp_config_path = os.path.join(self.output_dir, f"temp_{dispatcher}_config.json")
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            total_samples = 0
            metrics = defaultdict(list)
            
            # 创建环境
            env = MineEnv.make(temp_config_path, log=False, ticks=False)
            
            for episode in tqdm(range(self.episodes), desc=f"Collecting episodes for {dispatcher}"):
                observation, info = env.reset(seed=episode)
                
                episode_samples = 0
                episode_reward = 0
                last_production = 0
                
                for step in range(self.max_steps):
                    expert_action = observation['info']['sug_action']
                    state = self.preprocess_features(observation)
                    
                    next_observation, reward, done, truncated, info = env.step(expert_action)
                    self.all_rewards.append(reward)  # 收集reward
                    
                    current_production = float(info['produce_tons'])
                    production_increase = current_production - last_production
                    last_production = current_production
                    
                    data_sample = {
                        'dispatcher': dispatcher,
                        'episode': int(episode),
                        'step': int(step),
                        'state': [float(x) for x in state],
                        'action': int(expert_action),
                        'reward': float(reward),  # 添加reward到数据样本中
                        'event_type': observation['event_name'],
                        'delta_time': float(observation['info']['delta_time']),
                        'location': observation['the_truck_status']['truck_location'],
                        'truck_name': observation['truck_name']
                    }
                    
                    self.dataset.append(data_sample)
                    self.all_states.append(state)
                    episode_samples += 1
                    
                    episode_reward += reward
                    
                    if done or truncated:
                        break
                        
                    observation = next_observation
                
                metrics['episode_samples'].append(int(episode_samples))
                metrics['episode_rewards'].append(float(episode_reward))
                metrics['total_production'].append(float(last_production))
                total_samples += episode_samples
            
            env.close()
            
            # 为每个调度器保存单独的指标
            dispatcher_metrics_path = os.path.join(self.output_dir, f"metrics_{dispatcher}.json")
            metrics_summary = {
                'dispatcher': dispatcher,
                'total_samples': int(total_samples),
                'total_episodes': int(self.episodes),
                'avg_samples_per_episode': float(np.mean(metrics['episode_samples'])),
                'avg_reward_per_episode': float(np.mean(metrics['episode_rewards'])),
                'avg_production_per_episode': float(np.mean(metrics['total_production'])),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(dispatcher_metrics_path, 'w') as f:
                json.dump(metrics_summary, f, indent=4, cls=NumpyEncoder)
            
            # 删除临时配置文件
            os.remove(temp_config_path)
        
        # 保存完整数据集
        self._save_dataset(len(self.dataset), metrics)

    def preprocess_features(self, observation):
        """使用导入的preprocess_observation函数"""
        return preprocess_observation(observation, self.sim_time)

    def _save_dataset(self, total_samples, metrics):
        """保存数据集和计算标准化参数"""
        # 保存数据集
        dataset_path = os.path.join(self.output_dir, "dispatch_dataset.jsonl")
        with open(dataset_path, 'w') as f:
            for sample in self.dataset:
                json.dump(sample, f, cls=NumpyEncoder)
                f.write('\n')
        
        # 计算并保存标准化参数
        states_array = np.array(self.all_states)
        rewards_array = np.array(self.all_rewards)
        
        state_mean = np.mean(states_array, axis=0)
        state_std = np.std(states_array, axis=0)
        reward_mean = np.mean(rewards_array)
        reward_std = np.std(rewards_array)
        
        normalization_params = {
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "reward_mean": float(reward_mean),
            "reward_std": float(reward_std),
            "feature_dims": len(state_mean),
            "total_samples": total_samples,
            "dispatchers": self.dispatchers
        }
        
        params_path = os.path.join(self.output_dir, "normalization_params.json")
        with open(params_path, 'w') as f:
            json.dump(normalization_params, f, indent=4, cls=NumpyEncoder)
        
        print(f"\n数据收集完成!")
        print(f"总样本数: {total_samples}")
        print(f"数据集保存至: {dataset_path}")
        print(f"标准化参数保存至: {params_path}")
        print(f"\n特征统计信息:")
        print(f"特征维度: {len(state_mean)}")
        print(f"状态均值范围: [{min(state_mean):.3f}, {max(state_mean):.3f}]")
        print(f"状态标准差范围: [{min(state_std):.3f}, {max(state_std):.3f}]")
        print(f"奖励均值: {reward_mean:.3f}")
        print(f"奖励标准差: {reward_std:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect dispatch decision data")
    parser.add_argument("--env_config", type=str,
                        default="/Users/mac/PycharmProjects/truck_shovel_mix/sisymines_project/openmines/src/conf/north_pit_mine.json",
                        help="环境配置文件路径")
    parser.add_argument("--episodes", type=int, default=10,
                        help="收集数据的回合数")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="每个回合的最大步数")

    args = parser.parse_args()

    collector = DataCollector(
        env_config=args.env_config,
        episodes=args.episodes,
        max_steps=args.max_steps
    )
    collector.collect_data()