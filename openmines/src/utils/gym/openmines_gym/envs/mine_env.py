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
import importlib
import pkgutil
import re
import inflection

from openmines.src.utils.rl_env import prepare_env
from openmines.src.utils.feature_processing import preprocess_observation


def get_dispatcher_class(sug_dispatcher):
    """根据调度器名称动态导入调度器类"""
    dispatchers_package = 'openmines.src.dispatch_algorithms'
    
    # 使用inflection库进行转换
    module_name_guess = inflection.underscore(sug_dispatcher)
    potential_module_path = f"{dispatchers_package}.{module_name_guess}"
    
    try:
        # 尝试导入模块
        module = importlib.import_module(potential_module_path)
        
        # 检查模块中是否存在对应类
        if hasattr(module, sug_dispatcher):
            return getattr(module, sug_dispatcher)
    except ImportError:
        # 如果猜测的模块不存在，尝试从总包中导入
        try:
            module = importlib.import_module(dispatchers_package)
            for _, sub_module_name, _ in pkgutil.iter_modules(module.__path__, dispatchers_package + '.'):
                try:
                    sub_module = importlib.import_module(sub_module_name)
                    if hasattr(sub_module, sug_dispatcher):
                        return getattr(sub_module, sug_dispatcher)
                except ImportError:
                    continue
        except Exception:
            pass
    return None

class GymMineEnv(gym.Env):
    """将矿山环境包装为标准的gym环境"""
    metadata = {"render_modes": None, "render_fps": None}

    def __init__(self, config_file, reward_mode:str = "dense", sug_dispatcher:str="ShortestTripDispatcher", seed=42, log=False, ticks=False):
        super().__init__()

        # 加载配置
        self.config_file = config_file
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
        # 添加调度器配置
        self.config['sug_dispatcher'] = sug_dispatcher
        self.reward_mode = reward_mode
        
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
        
        # 获取建议的调度器,如果需要
        if 'sug_dispatcher' in self.config:
            sug_dispatcher = self.config['sug_dispatcher']
            self.config['dispatcher'] = {'type': [sug_dispatcher]}

        # 启动新的环境进程
        self.process = multiprocessing.Process(
            target=prepare_env,
            args=(self.obs_queue, self.act_queue, self.config,
                  self.reward_mode, self.config['sim_time'], self.log, self.ticks, self.seed_value)
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

    def __init__(self, config_file, reward_mode:str = "dense", sug_dispatcher:str="ShortestTripDispatcher", seed=42, log=False, ticks=False):
        super().__init__()

        # 加载配置
        self.config_file = config_file
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
        # 添加调度器配置    
        self.config['sug_dispatcher'] = sug_dispatcher
        self.reward_mode = reward_mode
        
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, full_random:bool = False) -> tuple[Any, dict[str, Any]]:
        # 如果你有自己的种子逻辑
        # chosen_seed = seed if seed is not None else self.seed_value
        # super().reset(seed=chosen_seed)
        # np.random.seed(chosen_seed)

        if self.thread and self.is_running:
            self.is_running = False
            self.thread.join()
        if full_random:
            # 1) 随机修改充电场卡车型号数量
            charging_site = self.config['charging_site']
            for truck_info in charging_site["trucks"]:
                old_count = truck_info["count"]
                new_count = np.random.randint(1, 2 * old_count + 1)  # 1 ~ 2*old_count 范围
                truck_info["count"] = new_count

            # 2) 随机调整各装载区的小铲车数量；将 ≥20 吨的大铲车统一收集，然后随机分配到任意装载区
            big_shovels_global = []
            for load_site in self.config['load_sites']:
                original_shovels = load_site['shovels']

                # 分离出"小于20吨"和"大于等于20吨"
                small_shovels = [s for s in original_shovels if s["tons"] < 20]
                big_shovels = [s for s in original_shovels if s["tons"] >= 20]

                # 随机修改"小铲车"数量，但保证至少有 1 台
                if len(small_shovels) > 0:
                    old_small_count = len(small_shovels)
                    # 保证每个装载区至少生成 1 台小铲车
                    new_small_count = np.random.randint(1, 2 * old_small_count + 1)
                    chosen_shovel_config = small_shovels[0]
                    new_small_shovels = [dict(chosen_shovel_config) for _ in range(new_small_count)]
                else:
                    # 如果原来就没有小铲车，那就给它造一个默认的小铲车，保证至少有 1 台
                    default_small_shovel = {
                        "name": "Default-Small-Shovel",
                        "tons": 2.25,
                        "cycle_time": 1,
                        "position_offset": [0.1, 0.0],
                    }
                    new_small_shovels = [default_small_shovel]

                # 收集本装载区的大铲车，放进全局列表，后面统一随机分配
                big_shovels_global.extend(big_shovels)

                # 本装载区先只保留"新小铲车"列表
                load_site['shovels'] = new_small_shovels

            # 3) 将所有 ≥20 吨的大铲车随机分配到各装载区
            load_sites_count = len(self.config['load_sites'])
            for shovel in big_shovels_global:
                idx = np.random.randint(0, load_sites_count)
                self.config['load_sites'][idx]['shovels'].append(shovel)

            # 4) 统一重命名装载区内所有铲车：名称 = "装载区名字 + -Shovel- + 序号"
            for load_site in self.config['load_sites']:
                for i, shovel in enumerate(load_site['shovels'], start=1):
                    shovel["name"] = f"{load_site['name']}-Shovel-{i}"

        # 重置线程相关
        self.obs_queue = Queue()
        self.act_queue = Queue()
        self.is_running = True

        self.thread = threading.Thread(
            target=prepare_env,
            args=(
                self.obs_queue,
                self.act_queue,
                self.config,
                self.reward_mode,
                self.config['sim_time'],
                self.log,
                self.ticks,
                None
            )
        )
        self.thread.daemon = True
        self.thread.start()

        out = self.obs_queue.get()
        observation = out["ob"]
        info = out["info"]

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


class ThreadMineDenseEnv(ThreadMineEnv):
    """密集奖励的多线程矿山环境"""
    def __init__(self, config_file, sug_dispatcher:str="ShortestTripDispatcher", seed=42, log=False, ticks=False):
        super().__init__(config_file=config_file, reward_mode="dense", sug_dispatcher=sug_dispatcher, seed=seed, log=log, ticks=ticks)

class ThreadMineSparseEnv(ThreadMineEnv):
    """稀疏奖励的多线程矿山环境"""
    def __init__(self, config_file, sug_dispatcher:str="ShortestTripDispatcher", seed=42, log=False, ticks=False):
        super().__init__(config_file=config_file, reward_mode="sparse", sug_dispatcher=sug_dispatcher, seed=seed, log=log, ticks=ticks)

class GymMineDenseEnv(GymMineEnv):
    """密集奖励的单线程矿山环境"""
    def __init__(self, config_file, sug_dispatcher:str="ShortestTripDispatcher", seed=42, log=False, ticks=False):
        super().__init__(config_file=config_file, reward_mode="dense", sug_dispatcher=sug_dispatcher, seed=seed, log=log, ticks=ticks)

class GymMineSparseEnv(GymMineEnv):
    """稀疏奖励的单线程矿山环境"""
    def __init__(self, config_file, sug_dispatcher:str="ShortestTripDispatcher", seed=42, log=False, ticks=False):
        super().__init__(config_file=config_file, reward_mode="sparse", sug_dispatcher=sug_dispatcher, seed=seed, log=log, ticks=ticks)



# 使用示例
if __name__ == "__main__":
    """
    THREAD MINE: dense(default)
    """
    # 创建环境
    env = ThreadMineEnv("../../../../conf/north_pit_mine.json", 
                        sug_dispatcher="NaiveDispatcher",
                         log=False, ticks=False)
    # 添加episode统计包装器
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # 测试环境
    observation, info = env.reset(seed=42)
    for i in range(2000):
        # action = env.action_space.sample()  # 随机动作
        action = info["sug_action"]  # 使用info中的建议动作
        print(f"A_{i} Action: {action}")
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step: {i}, Reward: {reward}, Info: {info} terminated: {terminated}, truncated: {truncated}")
        if terminated or truncated:
            observation, info = env.reset()
            break
    env.close()
    """
    GYM IMPORT MINES
    """
    import openmines_gym
    import gymnasium as gym
    
    # 测试dense环境
    env = gym.make('mine/Mine-v1-dense', config_file="../../../../conf/north_pit_mine.json",
                   sug_dispatcher="NaiveDispatcher", log=False, ticks=False)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    observation, info = env.reset(seed=42)
    for i in range(1000):
        action = info["sug_action"]  # 使用建议动作
        print(f"Dense Step {i}, Action: {action}")
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Dense Step {i}, Reward: {reward}, Info: {info}")
        if terminated or truncated:
            print("Dense Environment terminated!")
            observation, info = env.reset()
            break
    env.close()
    
    # 测试sparse环境
    env = gym.make('mine/Mine-v1-sparse', config_file="../../../../conf/north_pit_mine.json",
                   sug_dispatcher="NaiveDispatcher", log=False, ticks=False)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    observation, info = env.reset(seed=42)
    for i in range(1000):
        action = info["sug_action"]  # 使用建议动作
        print(f"Sparse Step {i}, Action: {action}")
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Sparse Step {i}, Reward: {reward}, Info: {info}")
        if terminated or truncated:
            print("Sparse Environment terminated!")
            observation, info = env.reset()
            break
    env.close()