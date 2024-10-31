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
        """收集调度决策数据"""
        total_samples = 0
        metrics = defaultdict(list)

        # 创建环境
        env = MineEnv.make(self.env_config, log=False, ticks=False)

        for episode in tqdm(range(self.episodes), desc="Collecting episodes"):
            observation, info = env.reset(seed=episode)

            episode_samples = 0
            episode_reward = 0
            last_production = 0

            for step in range(self.max_steps):
                # 获取专家动作
                expert_action = observation['info']['sug_action']

                # 预处理特征
                state = self.preprocess_features(observation)

                # 记录数据样本
                data_sample = {
                    'episode': int(episode),  # 确保是Python原生int类型
                    'step': int(step),
                    'state': [float(x) for x in state],  # 转换为Python原生float列表
                    'action': int(expert_action),  # 确保是Python原生int类型
                    'event_type': observation['event_name'],
                    'delta_time': float(observation['info']['delta_time']),  # 确保是Python原生float类型
                    'location': observation['the_truck_status']['truck_location'],
                    'truck_name': observation['truck_name']
                }
                self.dataset.append(data_sample)
                episode_samples += 1

                # 执行专家动作
                next_observation, reward, done, truncated, info = env.step(expert_action)

                # 记录产量变化
                current_production = float(info['produce_tons'])
                production_increase = current_production - last_production
                last_production = current_production

                episode_reward += reward

                if done or truncated:
                    break

                observation = next_observation

            # 记录episode指标
            metrics['episode_samples'].append(int(episode_samples))
            metrics['episode_rewards'].append(float(episode_reward))
            metrics['total_production'].append(float(last_production))
            total_samples += episode_samples

        env.close()

        # 保存数据集和指标
        self._save_dataset(total_samples, metrics)

    def preprocess_features(self, observation):
        """特征预处理逻辑"""
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

        truck_location = observation['the_truck_status']['truck_location_onehot']

        order_and_position = np.array([event_type + truck_location + [action_space_n]])
        truck_num = observation['mine_status']['truck_count']

        truck_features = np.array([
            np.log(observation['the_truck_status']['truck_load'] + 1),
            np.log(observation['the_truck_status']['truck_cycle_time'] + 1),
        ])

        target_features = np.concatenate([
            np.array(observation['target_status']['queue_lengths']) / (truck_num + 1e-8),
            np.log(np.array(observation['target_status']['capacities']) + 1),
            np.log(np.array(observation['target_status']['est_wait']) + 1),
        ])

        road_dist = np.array(observation['cur_road_status']['oh_distances'])
        road_traffic = np.array(observation['cur_road_status']['oh_truck_jam_count'])
        road_jam = np.array(observation['cur_road_status']['oh_truck_jam_count'])

        state = np.concatenate([order_and_position.squeeze(), truck_features, target_features,
                                road_dist, road_traffic, road_jam])

        # 转换所有数值为Python原生类型
        return [float(x) for x in state]

    def _save_dataset(self, total_samples, metrics):
        """保存数据集和训练指标"""
        # 保存数据集
        dataset_path = os.path.join(self.output_dir, "dispatch_dataset.jsonl")
        with open(dataset_path, 'w') as f:
            for sample in self.dataset:
                json.dump(sample, f, cls=NumpyEncoder)
                f.write('\n')

        # 保存汇总指标
        metrics_summary = {
            'total_samples': int(total_samples),
            'total_episodes': int(self.episodes),
            'avg_samples_per_episode': float(np.mean(metrics['episode_samples'])),
            'avg_reward_per_episode': float(np.mean(metrics['episode_rewards'])),
            'avg_production_per_episode': float(np.mean(metrics['total_production'])),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=4, cls=NumpyEncoder)

        # 保存详细指标
        detailed_metrics = pd.DataFrame({
            'episode': range(self.episodes),
            'samples': metrics['episode_samples'],
            'rewards': metrics['episode_rewards'],
            'production': metrics['total_production']
        })
        metrics_csv_path = os.path.join(self.output_dir, "detailed_metrics.csv")
        detailed_metrics.to_csv(metrics_csv_path, index=False)

        print(f"\nData collection completed!")
        print(f"Total samples collected: {total_samples}")
        print(f"Dataset saved to: {dataset_path}")
        print(f"Metrics saved to: {metrics_path}")
        print(f"Detailed metrics saved to: {metrics_csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect dispatch decision data")
    parser.add_argument("--env_config", type=str,
                        default="../../conf/north_pit_mine.json",
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