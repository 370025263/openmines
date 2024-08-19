"""
将矿山环境包装为gym类型的接口
提供给agent训练
不使用gym
"""
import json
import pathlib

import numpy as np

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


class ActionSpace:
    def __init__(self, mine:"Mine", seed=42):
        self.mine = mine
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)

    def sample(self, truck:"Truck"):
        current_location = truck.current_location
        # 如果是LoadSite
        if isinstance(current_location, LoadSite):
            cur_index = self.mine.load_sites.index(current_location)
            cur_to_dump = self.mine.road.road_matrix[cur_index, :]
            ava_choice_count = len(self.mine.dump_sites)
        # 如果是ChargingSite
        elif isinstance(current_location, ChargingSite):
            ava_choice_count = len(self.mine.load_sites)
        # 如果是DumpSite
        elif isinstance(current_location, DumpSite):
            ava_choice_count = len(self.mine.load_sites)
        else:
            raise Exception(f"Truck {truck} is at UNKOWN LOCATION: {type(current_location)}")
        # 随机选择
        action = self.rng.choice(ava_choice_count)
        return action

    def seed(self, seed):
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)

class MineEnv:
    def __init__(self,seed_value=42):
        """按照参数初始化矿山
        """
        self.seed_value = seed_value
        self.config_file = None

    def step(self, action):
        # action给谁？



        return  observation, reward, done, truncated, info

    @staticmethod
    def make(config_file,seed_value=42):
        """通过读取配置文件，返回一个MineRL环境
        :return:
        """

        config = load_config(config_file)
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

        # 实例化
        env = MineEnv(seed_value=seed_value)
        env.config_file = config_file
        # 配置RL动作空间
        env.action_space = ActionSpace(mine, seed=env.seed_value)
        env.mine = mine
        # 开始实验
        print(f"Running simulation for {dispatcher.__class__.__name__}")
        ticks = env.mine.start(total_time=config['sim_time'])
        return env



    def reset(self):
        """重制环境生成观察空间,矿车INFO"""
        config = load_config(self.config_file)
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

        # 配置RL动作空间
        self.action_space = ActionSpace(mine, seed=self.seed_value)
        self.mine = mine
        self.mine.dispatcher.external_api

        # 开始实验
        print(f"Running simulation for {dispatcher.__class__.__name__}")
        ticks = self.mine.start(total_time=config['sim_time'])


        return observation, info


