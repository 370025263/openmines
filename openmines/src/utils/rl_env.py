"""
将矿山环境包装为gym类型的接口
提供给agent训练
不使用gym
"""
import json
import pathlib

import numpy as np
import multiprocessing
from multiprocessing import Queue

from openmines.src.dump_site import DumpSite, Dumper
from openmines.src.mine import Mine
from openmines.src.truck import Truck
from openmines.src.charging_site import ChargingSite
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.dump_site import DumpSite, Dumper
from openmines.src.road import Road
from openmines.src.dispatch_algorithms.rl_dispatch import RLDispatcher


def load_config(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def prepare_env(obs_queue:Queue, act_queue:Queue, config:dict, total_time:float=60*8):
    """接受mp的输入，然后构建mine和子进程。
    为了兼容windows平台的spawn机制
    因为simpy的env无法序列化，所以必须在子进程中构建mine和simpy的env

    :param obs_queue:
    :param act_queue:
    :param total_time:
    :return:
    """
    # log_path 为cwd下的logs文件夹
    log_path = pathlib.Path.cwd() / 'logs'
    # dispatcher
    dispatcher = RLDispatcher()
    # 初始化矿山
    mine = Mine(config['mine']['name'], log_path=log_path)
    mine.add_dispatcher(dispatcher)
    # 初始化充电站和卡车
    charging_site = ChargingSite(config['charging_site']['name'], position=config['charging_site']['position'])
    for truck_config in config['charging_site']['trucks']:
        for _ in range(truck_config['count']):
            truck = Truck(
                name=f"{truck_config['type']}{_ + 1}",
                truck_capacity=truck_config['capacity'],
                truck_speed=truck_config['speed']
            )
            charging_site.add_truck(truck)

    # 初始化装载点和铲车
    for load_site_config in config['load_sites']:
        load_site = LoadSite(name=load_site_config['name'], position=load_site_config['position'])
        for shovel_config in load_site_config['shovels']:
            shovel = Shovel(
                name=shovel_config['name'],
                shovel_tons=shovel_config['tons'],
                shovel_cycle_time=shovel_config['cycle_time'],
                position_offset=shovel_config['position_offset']
            )
            load_site.add_shovel(shovel)
        load_site.add_parkinglot(position_offset=load_site_config['parkinglot']['position_offset'],
                                 name=load_site_config['parkinglot']['name'])
        mine.add_load_site(load_site)

    # 初始化卸载点和卸载机
    for dump_site_config in config['dump_sites']:
        dump_site = DumpSite(dump_site_config['name'], position=dump_site_config['position'])
        for dumper_config in dump_site_config['dumpers']:
            for _ in range(dumper_config['count']):
                dumper = Dumper(
                    name=f"{dump_site_config['name']}-点位{_}",
                    dumper_cycle_time=dumper_config['cycle_time'],
                    position_offset=dumper_config['position_offset']
                )
                dump_site.add_dumper(dumper)
        dump_site.add_parkinglot(position_offset=dump_site_config['parkinglot']['position_offset'],
                                 name=dump_site_config['parkinglot']['name'])
        mine.add_dump_site(dump_site)

    # 初始化道路
    road_matrix = np.array(config['road']['road_matrix'])
    road_event_params = config['road'].get('road_event_params', {})  # 从配置中加载道路事件参数

    charging_to_load_road_matrix = config['road']['charging_to_load_road_matrix']
    road = Road(road_matrix=road_matrix, charging_to_load_road_matrix=charging_to_load_road_matrix,
                road_event_params=road_event_params)
    # # 添加充电站和装载区卸载区
    mine.add_road(road)
    mine.add_charging_site(charging_site)
    mine.start_rl(obs_queue, act_queue, total_time)  # 子进程的任务


class ActionSpace:
    def __init__(self, seed=42):
        """
        sample需要提供obs
        :param seed:
        """
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)

    def sample(self, ob:dict):
        """
        out = {
            "ob": self.current_observation,
            "info": info,
            "reward": reward,
            "truncated": trucated,
            "done": done
        }
        observation = {
            "truck_name":truck.name,
            "event_name": event_name, # order type
            "info": info, # total tons, time
            "the_truck_status":the_truck_status, # the truck stats
            "target_status":target_status, # loading/unloading site stats
            "cur_road_status":cur_road_status, # road net status
            "mine_status": mine.status["cur"],  # 当前矿山KPI、事件总数等统计信息
        }
        info = {"produce_tons": mine.produce_tons, "time": mine.env.now, "load_num":load_num, "unload_num":unload_num}

        the_truck_status = {
            "truck_location": truck.current_location.name,

            "truck_load": truck.truck_load,
            "truck_capacity": truck.truck_capacity,

            "truck_cycle_time": truck_cycle_time,
            "truck_speed": truck.truck_speed,
        }
        :param obs:
        :return:
        """
        order_type = ob["event_name"]
        truck_name = ob["truck_name"]

        info = ob["info"]
        t = info["time"]
        location = ob["the_truck_status"]["truck_location"]

        # 如果是LoadSite
        if order_type=="unhaul":
            ava_choice_count = info["load_num"]
        # 如果是ChargingSite
        elif order_type=="init":
            ava_choice_count = info["load_num"]
        # 如果是DumpSite
        elif order_type=="haul":
            ava_choice_count = info["unload_num"]
        else:
            raise Exception(f"Truck {truck_name} at {location} order is unkown: {order_type}")
        # 随机选择
        action = self.rng.choice(ava_choice_count)
        return action

    def seed(self, seed):
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)

    def limit_choice(self):
        """检查可用性，比如非可用目标。
        TODO：添加装载区临时失效的事件
        """
        pass


class MineEnv:
    def __init__(self,seed_value=42):
        """按照参数初始化矿山
        """
        self.seed_value = seed_value
        self.config_file = None
        self.cur_ob = None
        self.action_space = ActionSpace(seed=seed_value)

    def _seed(self, seed):
        self.seed_value = seed
        self.action_space.seed(seed)

    def step(self, action):
        # action给进程
        self.act_queue.put(action)
        # 等待消息
        out = self.obs_queue.get()
        """
        out = {
            "ob": self.current_observation,
            "info": info,
            "reward": reward,
            "truncated": trucated,
            "done": done
        }
        """
        observation = out["ob"]
        info = out["info"]
        reward = out["reward"]
        done = out["done"]
        truncated = out["truncated"]
        return observation, reward, done, truncated, info

    @staticmethod
    def make(config_file,seed_value=42):
        """通过读取配置文件，返回一个MineRL环境
        :return:
        """
        # 实例化
        env = MineEnv(seed_value=seed_value)
        env.config_file = config_file
        env.config = load_config(config_file)
        # 开始实验
        # 下面开启一个env的子进程
        env.obs_queue = Queue()
        env.act_queue = Queue()
        env.p = multiprocessing.Process(target=prepare_env, args=(env.obs_queue, env.act_queue, env.config, env.config[
            'sim_time']))  # 这里self.mine.start应该单独启动一个进程。和env主进程间使用队列通讯(ob,act等)。注意兼容多个env，防止信息穿台。
        env.p.start()
        print(f"Running re-simulation at pid:{env.p.pid}, env: {env.config['mine']['name']}")

        # # 等待子进程结束
        # p.join()
        # 获取ob
        """
        out = {
            "ob": self.current_observation,
            "info": info,
            "reward": reward,
            "truncated": trucated,
            "done": done
        }
        """
        return env

    def reset(self, seed=42):
        """重制环境生成观察空间,矿车INFO"""
        # 触发reset的时候，需要停止之前的进程
        if self.p.is_alive():
            print(f"Terminating previous simulation with pid: {self.p.pid}")
            self.p.terminate()
            self.p.join()  # 确保子进程结束

        # 开始实验
        self.seed_value = seed
        # todo: ticks功能在RL环境中也可以使用【暂时不做】
        # 下面开启一个env的子进程
        self.obs_queue = Queue()
        self.act_queue = Queue()
        self.p = multiprocessing.Process(target=prepare_env, args=(self.obs_queue, self.act_queue, self.config, self.config['sim_time']))# 这里self.mine.start应该单独启动一个进程。和env主进程间使用队列通讯(ob,act等)。注意兼容多个env，防止信息穿台。
        self.p.start()
        print(f"Running re-simulation at pid:{self.p.pid}, env: {self.config['mine']['name']}")
        # # 等待子进程结束
        # p.join()
        # 获取ob
        """
        out = {
            "ob": self.current_observation,
            "info": info,
            "reward": reward,
            "truncated": trucated,
            "done": done
        }
        """
        out = self.obs_queue.get()
        observation = out["ob"]
        info = out["info"]
        return observation, info

    def close(self):
        """防止产生孤儿进程"""
        print("waitting for mine subprocess to end")
        self.p.join()
        print("env closed")