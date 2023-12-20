import argparse
import json
import pathlib
import sys
import pkgutil
import importlib

import numpy as np

# add the sisymines package to the path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent.absolute()))
print(sys.path)

from sisymines.src.mine import Mine
from sisymines.src.truck import Truck
from sisymines.src.charging_site import ChargingSite
from sisymines.src.load_site import LoadSite, Shovel
from sisymines.src.dump_site import DumpSite, Dumper
from sisymines.src.road import Road
from sisymines.src.dispatcher import BaseDispatcher
from sisymines.src.dispatch_algorithms.naive_dispatch import NaiveDispatcher
from sisymines.src.dispatch_algorithms import *

def load_config(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def main(config_file):
    config = load_config(config_file)

    # 初始化矿山
    mine = Mine(config['mine']['name'])

    # 初始化调度器
    # 定义dispatch_algorithms所在的路径
    dispatchers_package = 'sisymines.src.dispatch_algorithms'
    # 导入dispatch_algorithms包
    dispatchers_module = importlib.import_module(dispatchers_package)
    # 遍历dispatch_algorithms包中的所有模块
    for _, module_name, _ in pkgutil.iter_modules(dispatchers_module.__path__, dispatchers_package + '.'):
        # 动态导入模块
        module = importlib.import_module(module_name)
        # 假设配置中指定的调度器类型
        dispatcher_type = config['dispatcher']['type']
        # 检查模块中是否有该类
        if hasattr(module, dispatcher_type):
            dispatcher_class = getattr(module, dispatcher_type)
            dispatcher = dispatcher_class()
            mine.add_dispatcher(dispatcher)
            break
    # Raise Error if no dispatcher is found
    assert mine.dispatcher is not None, f"Dispatcher {dispatcher_type} not found"

    # 初始化充电站和卡车
    charging_site = ChargingSite(config['charging_site']['name'])
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
        load_site = LoadSite(load_site_config['name'])
        for shovel_config in load_site_config['shovels']:
            shovel = Shovel(
                name=shovel_config['name'],
                shovel_tons=shovel_config['tons'],
                shovel_cycle_time=shovel_config['cycle_time']
            )
            load_site.add_shovel(shovel)
        mine.add_load_site(load_site)

    # 初始化卸载点和卸载机
    for dump_site_config in config['dump_sites']:
        dump_site = DumpSite(dump_site_config['name'])
        for dumper_config in dump_site_config['dumpers']:
            for _ in range(dumper_config['count']):
                dumper = Dumper(
                    name=f"{dump_site_config['name']}-点位{_}",
                    dumper_cycle_time=dumper_config['cycle_time']
                )
                dump_site.add_dumper(dumper)
        mine.add_dump_site(dump_site)

    # 初始化道路
    road_matrix = np.array(config['road']['road_matrix'])
    charging_to_load_road_matrix = config['road']['charging_to_load_road_matrix']
    road = Road(road_matrix=road_matrix, charging_to_load_road_matrix=charging_to_load_road_matrix)

    # # 添加充电站和装载区卸载区
    mine.add_road(road)
    mine.add_charging_site(charging_site)




    # 开始运行实验
    mine.start(total_time=60 * 8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mining simulation experiment")
    parser.add_argument('config_file', help="Path to the mine configuration JSON file")
    args = parser.parse_args()

    main(args.config_file)

