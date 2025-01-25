import gymnasium as gym
from docutils.utils.math.tex2unichar import space
from gymnasium import spaces
import numpy as np
import json
import multiprocessing
from multiprocessing import Queue
import logging
import pathlib
import threading
from typing import Optional, Dict, Any

from openmines.src.utils.rl_env import prepare_env


def preprocess_observation(observation, max_sim_time):
    """预处理原始观察，使其符合observation_space的格式"""

    """
    0.订单信息
    """
    # 1.订单类型,时间信息
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
    # 2.当前订单时间绝对位置和相对位置
    time_delta = float(observation['info']['delta_time']) / max_sim_time  # 距离上次调度的时间
    time_now = float(observation['info']['time']) / max_sim_time  # 当前时间(正则化）
    time_left = 1 - time_now  # 距离结束时间
    order_state = np.array([event_type[0], event_type[1], event_type[2], time_delta, time_now, time_left])

    """
    1.车辆自身信息
    """
    # 矿山总卡车数目（用于正则化）
    truck_num = observation['mine_status']['truck_count']
    # 4.车辆当前位置One-hot编码
    truck_location_onehot = np.array(observation["the_truck_status"]["truck_location_onehot"])
    # 车辆装载量，车辆循环时间（正则化）
    truck_features = np.array([
        np.log(observation['the_truck_status']['truck_load'] + 1),
        np.log(observation['the_truck_status']['truck_cycle_time'] + 1),
    ])
    truck_self_state = np.concatenate([truck_location_onehot, truck_features])

    """
    2.道路相关信息
    """
    # 车预期行驶时间
    travel_time = np.array(observation['cur_road_status']['distances']) * 60 / 25
    # 道路上卡车数量
    truck_counts = np.array(observation['cur_road_status']['truck_counts']) / (truck_num + 1e-8)
    # 道路距离信息
    road_dist = np.array(observation['cur_road_status']['oh_distances'])
    # 道路拥堵信息
    road_jam = np.array(observation['cur_road_status']['oh_truck_jam_count'])

    road_states = np.concatenate([travel_time, truck_counts, road_dist, road_jam])
    """
    3.目标点相关信息
    """
    # 预期等待时间
    est_wait = np.log(observation['target_status']['single_est_wait'] + 1)  # 包含了路上汽车+队列汽车的目标装载点等待时间
    tar_wait_time = np.log(np.array(observation['target_status']['est_wait']) + 1)  # 不包含路上汽车
    # 队列长度（正则化）
    queue_lens = np.array(observation['target_status']['queue_lengths']) / (truck_num + 1e-8)
    # 装载量
    tar_capa = np.log(np.array(observation['target_status']['capacities']) + 1)
    # 各个目标点当前的产能系数(维护导致的产能下降）
    ability_ratio = np.array(observation['target_status']['service_ratio'])
    # 已经生产的矿石量（正则化）
    produced_tons = np.log(np.array(observation['target_status']['produced_tons']) + 1)
    # processed_obs = {
    #     "the_truck_status": {
    #         "truck_location_index": truck_location_onehot,  # 已经是one-hot向量
    #         "truck_load": np.array([observation["the_truck_status"]["truck_load"]], dtype=np.float32),
    #         "truck_capacity": np.array([observation["the_truck_status"]["truck_capacity"]], dtype=np.float32),
    #         "truck_cycle_time": np.array([observation["the_truck_status"]["truck_cycle_time"]], dtype=np.float32),
    #         "truck_speed": np.array([observation["the_truck_status"]["truck_speed"]], dtype=np.float32),
    #     },
    #     "target_status": {
    #         "queue_lengths": np.array(observation["target_status"]["queue_lengths"], dtype=np.float32),
    #         "capacities": np.array(observation["target_status"]["capacities"], dtype=np.float32),
    #         "est_wait": np.array(observation["target_status"]["est_wait"], dtype=np.float32),
    #         "produced_tons": np.array(observation["target_status"]["produced_tons"], dtype=np.float32),
    #         "service_counts": np.array(observation["target_status"]["service_counts"], dtype=np.float32),
    #     },
    #     "cur_road_status": {
    #         "oh_truck_count": np.array(observation["cur_road_status"]["oh_truck_count"], dtype=np.float32),
    #         "oh_distances": np.array(observation["cur_road_status"]["oh_distances"], dtype=np.float32),
    #         "oh_truck_jam_count": np.array(observation["cur_road_status"]["oh_truck_jam_count"], dtype=np.float32),
    #         "oh_repair_count": np.array(observation["cur_road_status"]["oh_repair_count"], dtype=np.float32),
    #     },
    #     "event_name": {"init": 0, "haul": 1, "unhaul": 2}[observation["event_name"]],
    #     "mine_status": {
    #         "truck_count": np.array([observation["mine_status"]["truck_count"]], dtype=np.float32),
    #         "total_production": np.array([observation["mine_status"].get("total_production", 0)], dtype=np.float32),
    #     }
    # }
    tar_state = np.concatenate([est_wait, tar_wait_time, queue_lens, tar_capa, ability_ratio, produced_tons])

    state = np.concatenate([order_state, truck_self_state, road_states, tar_state])
    assert not np.isnan(state).any(), f"NaN detected in state: {state}"
    assert not np.isnan(time_delta), f"NaN detected in time_delta: {time_delta}"
    assert not np.isnan(time_now), f"NaN detected in time_now: {time_now}"

    return state


class GymMineEnv(gym.Env):
    """将矿山环境包装为标准的gym环境"""
    metadata = {"render_modes": None, "render_fps": None}

    def __init__(self, config_file, seed=42, log=False, ticks=False):
        super().__init__()

        # 加载配置
        self.config_file = config_file
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.load_site_n = len(self.config['load_sites'])
        self.dump_site_n = len(self.config['dump_sites'])

        self.log = log
        self.ticks = ticks
        self.seed_value = seed
        self.max_sim_time = self.config['sim_time']

        # 定义动作空间和观察空间
        # 动作空间：离散空间，最大值为最大的可选目标数量
        max_choices = max(
            self.config.get('load_sites', []).__len__(),  # 最大装载点数量
            self.config.get('dump_sites', []).__len__()  # 最大卸载点数量
        )
        self.action_space = spaces.Discrete(max_choices)
        road_n = self.load_site_n * self.dump_site_n * 2 + self.load_site_n
        site_n = self.load_site_n + self.dump_site_n
        # 观察空间：使用Dict空间来匹配原始环境的字典格式
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(194,), dtype=np.float32)
        #
        #     (
        #     spaces.Dict({
        #     # 车辆状态
        #     "the_truck_status": spaces.Dict({
        #         "truck_location_index": spaces.Box(low=0, high=np.inf, shape=(1 + site_n,), dtype=np.float32),
        #         "truck_load": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        #         "truck_capacity": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        #         "truck_cycle_time": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        #         "truck_speed": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        #     }),
        #     # 目标状态
        #     "target_status": spaces.Dict({
        #         "queue_lengths": spaces.Box(low=0, high=np.inf, shape=(site_n,), dtype=np.float32),  # 假设最多10个目标
        #         "capacities": spaces.Box(low=0, high=np.inf, shape=(site_n,), dtype=np.float32),
        #         "est_wait": spaces.Box(low=0, high=np.inf, shape=(site_n,), dtype=np.float32),
        #         "produced_tons": spaces.Box(low=0, high=np.inf, shape=(site_n,), dtype=np.float32),
        #         "service_counts": spaces.Box(low=0, high=np.inf, shape=(site_n,), dtype=np.float32),
        #     }),
        #     # ROAD
        #     "cur_road_status": spaces.Dict({
        #         "oh_truck_count": spaces.Box(low=0, high=np.inf, shape=(road_n,), dtype=np.float32),
        #         "oh_distances":spaces.Box(low=0, high=np.inf, shape=(road_n,), dtype=np.float32),
        #         "oh_truck_jam_count": spaces.Box(low=0, high=np.inf, shape=(road_n,), dtype=np.float32),
        #         "oh_repair_count":spaces.Box(low=0, high=np.inf, shape=(road_n,), dtype=np.float32),
        #     }),
        #
        #     # 事件信息
        #     "event_name": spaces.Discrete(3),  # init, haul, unhaul
        #     # 矿山状态
        #     "mine_status": spaces.Dict({
        #         "truck_count": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        #         "total_production": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        #     })
        # }))

        # 初始化通信队列和进程
        self.obs_queue = None
        self.act_queue = None
        self.process = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[Any, dict[str, Any]]:
        """重置环境到初始状态"""
        # 设置随机种子
        super().reset(seed=seed if seed is not None else self.seed_value)

        # 如果存在旧进程，终止它
        if self.process and self.process.is_alive():
            self.process.terminate()

        # 创建新的通信队列
        self.obs_queue = Queue()
        self.act_queue = Queue()

        # 启动新的环境进程
        self.process = multiprocessing.Process(
            target=prepare_env,
            args=(self.obs_queue, self.act_queue, self.config,
                  self.config['sim_time'], self.log, self.ticks, self.seed_value)
        )
        self.process.start()

        # 获取初始观察
        out = self.obs_queue.get()
        observation = out["ob"]
        info = out["info"]
        # 预处理观察
        processed_obs = preprocess_observation(observation, self.max_sim_time)
        return processed_obs, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """执行一个动作并返回结果"""
        # 发送动作到环境进程
        self.act_queue.put(action)

        # 接收结果
        out = self.obs_queue.get()
        observation = out["ob"]
        reward = out["reward"]
        terminated = out["done"]
        truncated = out["truncated"]
        info = out["info"]

        # 预处理观察
        processed_obs = preprocess_observation(observation, self.max_sim_time)

        return processed_obs, reward, terminated, truncated, info

    def close(self):
        """清理资源"""
        if self.process and self.process.is_alive():
            self.process.terminate()
        super().close()

    def render(self):
        """实现渲染方法"""
        # 当前环境不支持渲染
        pass



class ThreadMineEnv(gym.Env):
    """将矿山环境包装为标准的gym环境(线程版本)"""
    metadata = {"render_modes": None, "render_fps": None}

    def __init__(self, config_file, seed=42, log=False, ticks=False):
        super().__init__()

        # 加载配置
        self.config_file = config_file
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        self.load_site_n = len(self.config['load_sites'])
        self.dump_site_n = len(self.config['dump_sites'])

        self.log = log
        self.ticks = ticks
        self.seed_value = seed
        self.max_sim_time = self.config['sim_time']

        # 定义动作空间和观察空间
        max_choices = max(
            self.config.get('load_sites', []).__len__(),  # 最大装载点数量
            self.config.get('dump_sites', []).__len__()  # 最大卸载点数量
        )
        self.action_space = spaces.Discrete(max_choices)

        # 观察空间：26维状态空间
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(194,),
            dtype=np.float32
        )

        # 初始化通信队列和线程
        self.obs_queue = None
        self.act_queue = None
        self.thread = None
        self.is_running = False

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[Any, dict[str, Any]]:
        """重置环境到初始状态"""
        # 设置随机种子
        super().reset(seed=seed if seed is not None else self.seed_value)

        # 如果存在旧线程，停止它
        if self.thread and self.is_running:
            self.is_running = False
            self.thread.join()

        # 创建新的通信队列
        self.obs_queue = Queue()
        self.act_queue = Queue()
        self.is_running = True

        # 启动新的环境线程
        self.thread = threading.Thread(
            target=prepare_env,
            args=(self.obs_queue, self.act_queue, self.config,
                  self.config['sim_time'], self.log, self.ticks, self.seed_value)
        )
        self.thread.daemon = True  # 设置为守护线程
        self.thread.start()

        # 获取初始观察
        out = self.obs_queue.get()
        observation = out["ob"]
        info = out["info"]

        # 预处理观察
        processed_obs = preprocess_observation(observation, self.max_sim_time)
        return processed_obs, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """执行一个动作并返回结果"""
        # 发送动作到环境线程
        self.act_queue.put(action)

        # 接收结果
        out = self.obs_queue.get()
        observation = out["ob"]
        reward = out["reward"]
        terminated = out["done"]
        truncated = out["truncated"]
        info = out["info"]

        # 预处理观察
        processed_obs = preprocess_observation(observation, self.max_sim_time)

        return processed_obs, reward, terminated, truncated, info

    def close(self):
        """清理资源"""
        if self.thread and self.is_running:
            self.is_running = False
            self.thread.join(timeout=1.0)  # 等待线程结束，最多等待1秒
        super().close()

    def render(self):
        """实现渲染方法"""
        # 当前环境不支持渲染
        pass


# 使用示例
if __name__ == "__main__":
    """
    THREAD MINE"""
    # 创建环境
    env = ThreadMineEnv("../../../../conf/north_pit_mine.json", log=False, ticks=False)
    # 添加episode统计包装器
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # 测试环境
    observation, info = env.reset(seed=42)
    for i in range(2000):
        action = env.action_space.sample()  # 随机动作
        print(f"A_{i} Action: {action}")
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step: {i}, Reward: {reward}, Info: {info} terminated: {terminated}, truncated: {truncated}")
        if terminated or truncated:
            observation, info = env.reset()
            break
    env.close()

    # """Multiprocessing Mine"""
    # # 创建环境
    # env = GymMineEnv("../../../../conf/north_pit_mine.json", log=False, ticks=False)
    # # 添加一些常用的包装器
    # env = gym.wrappers.RecordEpisodeStatistics(env)  # 记录episode统计
    # # 使用环境
    # observation, info = env.reset(seed=42)
    # for i in range(2000):
    #     action = env.action_space.sample()  # 随机动作
    #     print(f"A_{i} Action: {action}")
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     if i % 70 == 0 and i !=0:
    #         pass
    #     print(f"Step: {i}, Reward: {reward}, Info: {info} terminated: {terminated}, truncated: {truncated}")
    #
    #     if terminated or truncated:
    #         observation, info = env.reset()
    #         break
    #
    # env.close()