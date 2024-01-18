import argparse
import json
import pathlib
import sys
import pkgutil
import importlib
from datetime import time

import numpy as np

from sisymines.src.utils.visualization.graphher import VisualGrapher

# add the sisymines package to the path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent.absolute()))

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

def run_simulation(config_file=None):
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

    # 开始运行实验
    mine.start(total_time=config['sim_time'])


def run_visualization(tick_file=None):
    visual_grapher = VisualGrapher(tick_file)
    # 构造路径和文件名字
    gif_file = tick_file.strip('.json') + '.gif'
    visual_grapher.create_animation(output_path=gif_file)

def main():
    parser = argparse.ArgumentParser(description='Run a dispatch simulation of a mine with your DISPATCH algorithm and MINE config file')
    subparsers = parser.add_subparsers(help='commands', dest='command')

    # 直接访问入口 sisymine -f config.json, sisymine -v tick.json
    # 添加运行参数 '-f' 和 '-v'
    parser.add_argument('-f', '--config-file', type=str, help='Path to the config file')
    parser.add_argument('-v', '--tick-file', type=str, help='Path to the simulation tick file')

    # add command 'run'
    run_parser = subparsers.add_parser('run', help='Run a simulation experiment')
    run_parser.add_argument('-f', '--config-file', type=str, required=True, help='Path to the config file')

    # add command visualize
    visualize_parser = subparsers.add_parser('visualize', help='Visualize a simulation experiment')
    visualize_parser.add_argument('-f', '--tick-file', type=str, required=True, help='Path to the simulation tick file')

    args = parser.parse_args()
    # 如command为空，那么检查f/v参数是否存在，如果不存在则print help；如果存在f/v参数则执行run/visualize
    if args.command is None:
        if args.config_file is None and args.tick_file is None:
            parser.print_help()
        elif args.config_file is not None:
            run_simulation(config_file=args.config_file)
        elif args.tick_file is not None:
            run_visualization(tick_file=args.tick_file)
    if args.command == 'run':
        print("args.config_file", args.config_file)
        run_simulation(config_file=args.config_file)
    if args.command == 'visualize':
        tick_file = args.tick_file
        run_visualization(tick_file=tick_file)


if __name__ == "__main__":
    config_path = sys.argv[1]
    run_simulation(config_file=config_path)


