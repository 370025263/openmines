import argparse
import json
import pathlib
import sys
import pkgutil
import importlib
from datetime import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from openmines.src.utils.visualization.charter import Charter
from openmines.src.utils.visualization.graphher import VisualGrapher

# add the openmines package to the path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent.absolute()))

from openmines.src.mine import Mine
from openmines.src.truck import Truck
from openmines.src.charging_site import ChargingSite
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.dump_site import DumpSite, Dumper
from openmines.src.road import Road
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.dispatch_algorithms.naive_dispatch import NaiveDispatcher
from openmines.src.dispatch_algorithms import *

def load_config(filename):
    # if Dict just return
    if isinstance(filename, dict):
        return filename
    with open(filename, 'r') as file:
        return json.load(file)

def run_dispatch_sim(dispatcher: BaseDispatcher, config_file):
    config = load_config(config_file)
    # log_path 为cwd下的logs文件夹
    log_path = pathlib.Path.cwd() / 'logs'
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

    # 开始实验
    print(f"Running simulation for {dispatcher.__class__.__name__}")
    ticks = mine.start(total_time=config['sim_time'])
    return ticks


def run_simulation(config_file=None):
    config = load_config(config_file)
    charter = Charter(config_file)
    states_dict = dict()
    # 初始化调度器
    # 定义dispatch_algorithms所在的路径
    dispatchers_package = 'openmines.src.dispatch_algorithms'
    # 导入dispatch_algorithms包
    dispatchers_module = importlib.import_module(dispatchers_package)
    # 遍历dispatch_algorithms包中的所有模块
    dispatchers_list = []
    for _, module_name, _ in pkgutil.iter_modules(dispatchers_module.__path__, dispatchers_package + '.'):
        # 动态导入模块
        module = importlib.import_module(module_name)
        # 假设配置中指定的调度器类型
        dispatcher_types:list = config['dispatcher']['type']
        for dispatcher_type in dispatcher_types:
            # 检查模块中是否有该类
            if hasattr(module, dispatcher_type):
                dispatcher_class = getattr(module, dispatcher_type)
                dispatcher = dispatcher_class()
                if dispatcher.name not in [dispatcher.name for dispatcher in dispatchers_list]:
                    dispatchers_list.append(dispatcher)

    # 开始运行对比实验
    for dispatcher in dispatchers_list:
        dispatcher_name = dispatcher.name
        # RUN SIMULATION
        ticks = run_dispatch_sim(dispatcher, config_file)
        # 读取运行结果并保存，等待绘图
        ## 读取production
        times = []
        produced_tons_list = []
        service_count_list = []
        waiting_truck_count_list = []
        # ticks 是一个字典 key为时间，value为一个字典，包含了当前时间的所有信息
        for tick in ticks.values():
            if 'mine_states' not in tick:
                continue
            tick = tick['mine_states']
            times.append(tick['time'])
            produced_tons_list.append(tick['produced_tons'])
            service_count_list.append(tick['service_count'])
            waiting_truck_count_list.append(tick['waiting_truck_count'])
        states_dict[dispatcher_name] = {
            'times': times,
            'produced_tons_list': produced_tons_list,
            'service_count_list': service_count_list,
            'waiting_truck_count_list': waiting_truck_count_list,
            'summary': ticks['summary']
        }
    # 绘制图表
    if not states_dict:
        print("No data to plot")
        return
    charter.draw(states_dict)
    charter.save()

def run_visualization(tick_file=None):
    visual_grapher = VisualGrapher(tick_file)
    # 构造路径和文件名字
    gif_file = tick_file.strip('.json') + '.gif'
    visual_grapher.create_animation(output_path=gif_file)

####################  NEWLY ADDED FOR ABLATION  ####################

def run_scene_based_fleet_ablation_experiment(config_file, min_truck, max_truck):
    """
    单场景多算法, 在 [min_truck, max_truck] 的fleet size做消融.
    不产生之前的产量/表格，只画 ablation 对比图.
    """
    import math, copy
    from openmines.src.utils.visualization.charter import Charter

    config = load_config(config_file)
    # 原始卡车数:
    truck_info = config["charging_site"]["trucks"]
    original_counts = [t["count"] for t in truck_info]
    total_orig = sum(original_counts)
    ratios = [c / total_orig for c in original_counts]

    minT = int(min_truck)
    maxT = int(max_truck)
    if maxT < minT:
        minT, maxT = maxT, minT

    # 生成10个点
    if maxT == minT:
        fleet_sizes = [minT]
    else:
        step = (maxT - minT)/9
        fleet_sizes = [int(math.floor(minT + i*step)) for i in range(10)]

    # 获取所有dispatcher
    dispatcher_types = config['dispatcher']['type']
    # 结果: { dispatcher_name: {'fleet_sizes':[], 'productions':[]}, ...}
    results = {}
    for dt in dispatcher_types:
        results[dt] = {'fleet_sizes': [], 'productions': []}

    # 依次修改 truck 数, 运行, 并记录产量
    import pkgutil, importlib
    dispatchers_package = 'openmines.src.dispatch_algorithms'
    dispatchers_module = importlib.import_module(dispatchers_package)

    def get_dispatcher_class(name):
        for _, mname, _ in pkgutil.iter_modules(dispatchers_module.__path__, dispatchers_package + '.'):
            mod = importlib.import_module(mname)
            if hasattr(mod, name):
                return getattr(mod, name)
        return None

    for fs in fleet_sizes:
        new_conf = copy.deepcopy(config)
        assigned = 0
        for i, t_ in enumerate(new_conf["charging_site"]["trucks"]):
            new_count = int(math.floor(fs*ratios[i]))
            assigned += new_count
            t_["count"] = new_count
        leftover = fs - assigned
        if leftover>0:
            new_conf["charging_site"]["trucks"][-1]["count"] += leftover

        # 对每个dispatcher跑
        for dt in dispatcher_types:
            d_class = get_dispatcher_class(dt)
            if not d_class:
                print(f"[scene_ablation] Dispatcher {dt} not found, skip.")
                continue
            d_obj = d_class()
            ticks = run_dispatch_sim(d_obj, new_conf)
            produced = 0
            for td in ticks.values():
                if 'mine_states' in td:
                    ms = td['mine_states']
                    produced = ms['produced_tons']  # 最后一刻
            results[dt]['fleet_sizes'].append(fs)
            results[dt]['productions'].append(produced)

    # 画图
    charter = Charter(config_file)
    charter.draw_scene_based_fleet_ablation_experiment(results, original_fleet_size=total_orig)
    # 不保存普通图表, 只保存 ablation
    charter.save_ablation(tag="scene_ablation")


def run_algo_based_fleet_ablation_experiment(config_dir, baseline, target, min_truck=1, max_truck=160):
    """
    多场景+双算法消融实验 (algo-based ablation).
    Bilingual comment:
    1) 读取 config_dir 中的多个场景 (多个JSON文件).
    2) 对每个场景, 在 [min_truck, max_truck] 之间等分若干个点, 改变卡车总数.
    3) 分别运行 baseline 和 target 算法, 得到产量; 计算 ratio= target_produced / baseline_produced.
    4) 最终 scenes_data 中, 每个场景只有一条线: x= fleet_size, y= ratio(target/baseline).
    5) 让 Charter 的 draw_algo_based_fleet_ablation_experiment() 来画出这(3个)场景的(3条)对比线.

    English summary:
    - For each scenario file in config_dir, vary the total truck count from min_truck to max_truck.
    - Run baseline & target, get produced tons. Then ratio = target / baseline.
    - We'll plot one line per scenario: x= fleet size, y= ratio.
    """

    import os, math, copy
    from openmines.src.utils.visualization.charter import Charter
    import pathlib
    import pkgutil, importlib

    cdir = pathlib.Path(config_dir)
    if not cdir.exists():
        print(f"[algo_ablation] config_dir {config_dir} not found.")
        return

    file_list = list(cdir.glob("*.json"))
    if not file_list:
        print(f"No json in {config_dir}. skip.")
        return

    # Build the set of fleet sizes
    minT = int(min_truck)
    maxT = int(max_truck)
    if maxT < minT:
        minT, maxT = maxT, minT

    if maxT == minT:
        fleet_sizes = [minT]
    else:
        step = (maxT - minT) / 9
        fleet_sizes = [int(math.floor(minT + i * step)) for i in range(10)]

    # Find dispatcher classes
    dispatchers_package = 'openmines.src.dispatch_algorithms'
    dispatchers_module = importlib.import_module(dispatchers_package)

    def get_d_class(name):
        for _, mname, _ in pkgutil.iter_modules(dispatchers_module.__path__, dispatchers_package + '.'):
            mod = importlib.import_module(mname)
            if hasattr(mod, name):
                return getattr(mod, name)
        return None

    baseline_class = get_d_class(baseline)
    target_class = get_d_class(target)
    if not baseline_class or not target_class:
        print(f"[algo_ablation] cannot find baseline={baseline} or target={target}.")
        return

    # We'll collect results in scenes_data, each scenario => { 'fleet_sizes': [...], 'ratios': [...] }
    scenes_data = {}

    # Iterate each scenario file
    for cfg_path in file_list:
        scene_name = cfg_path.stem
        conf = load_config(cfg_path)

        # Original total trucks & ratio
        truck_info = conf["charging_site"]["trucks"]
        orig_counts = [t_["count"] for t_ in truck_info]
        total_orig = sum(orig_counts)
        ratios = [c / total_orig for c in orig_counts]

        # scene_data: each scenario has 'fleet_sizes' and 'ratios'
        sc_data = {
            'fleet_sizes': [],
            'ratios': []
        }

        # For each fleet size, build config and run baseline/target
        for fs in fleet_sizes:
            new_conf = copy.deepcopy(conf)
            assigned = 0
            for i, t_ in enumerate(new_conf["charging_site"]["trucks"]):
                new_ct = int(math.floor(fs * ratios[i]))
                assigned += new_ct
                t_["count"] = new_ct
            leftover = fs - assigned
            if leftover > 0:
                new_conf["charging_site"]["trucks"][-1]["count"] += leftover

            # Run baseline
            b_obj = baseline_class()
            b_ticks = run_dispatch_sim(b_obj, new_conf)
            produced_b = 0.0
            for td in b_ticks.values():
                if 'mine_states' in td:
                    produced_b = td['mine_states']['produced_tons']

            # Run target
            t_obj = target_class()
            t_ticks = run_dispatch_sim(t_obj, new_conf)
            produced_t = 0.0
            for td in t_ticks.values():
                if 'mine_states' in td:
                    produced_t = td['mine_states']['produced_tons']

            # Compute ratio = target / baseline (if baseline>0)
            ratio = 0.0
            if produced_b > 1e-9:
                ratio = produced_t / produced_b

            sc_data['fleet_sizes'].append(fs)
            sc_data['ratios'].append(ratio)

        scenes_data[scene_name] = sc_data

    # Pass to Charter
    c = Charter(str(config_dir))
    # We'll reuse the same function name but implement ratio lines
    c.draw_algo_based_fleet_ablation_experiment(scenes_data, baseline, target)
    # Then we save
    c.save_ablation(tag="algo_ablation")

def main():
    parser = argparse.ArgumentParser(description='Run a dispatch simulation of a mine with your DISPATCH algorithm and MINE config file')
    subparsers = parser.add_subparsers(help='commands', dest='command')

    # 直接访问入口 openmines -f config.json, openmines -v tick.json
    # 添加运行参数 '-f' 和 '-v'
    parser.add_argument('-f', '--config-file', type=str, help='Path to the config file')
    parser.add_argument('-v', '--tick-file', type=str, help='Path to the simulation tick file')

    # add command 'run'
    run_parser = subparsers.add_parser('run', help='Run a simulation experiment')
    run_parser.add_argument('-f', '--config-file', type=str, required=True, help='Path to the config file')

    # add command visualize
    visualize_parser = subparsers.add_parser('visualize', help='Visualize a simulation experiment')
    visualize_parser.add_argument('-f', '--tick-file', type=str, required=True, help='Path to the simulation tick file')

    # 在单个scenario中，不同算法在不同车队大小中的消融实验
    scene_based_fleet_ablation_parser = subparsers.add_parser('scene_based_fleet_ablation', help='Run an ablation experiment of fleet size on a certain scene')
    scene_based_fleet_ablation_parser.add_argument('-f', '--config-file', type=str, required=True, help='Path to the config file')
    scene_based_fleet_ablation_parser.add_argument('-m', '--min', type=str, required=True, help='Minimum value of the truck number')
    scene_based_fleet_ablation_parser.add_argument('-M', '--max', type=str, required=True, help='Maximum value of the truck number')

    # 在不同scenario中，目标算法和基线算法在不同车队大小中的消融实验
    algo_based_fleet_ablation_parser = subparsers.add_parser('algo_based_fleet_ablation', help='Run an ablation experiment of fleet size on a certain scene')
    # a folder containing multiple config files
    algo_based_fleet_ablation_parser.add_argument('-d', '--config-dir', type=str, required=True, help='Path to the config directory')
    # baseline algorithm name
    algo_based_fleet_ablation_parser.add_argument('-b', '--baseline', type=str, required=True, help='Baseline algorithm name')
    # target algorithm name
    algo_based_fleet_ablation_parser.add_argument('-t', '--target', type=str, required=True, help='Target algorithm name')
    # truck min number, 如果没有指定就是1
    algo_based_fleet_ablation_parser.add_argument('-m', '--min', type=str, required=False, help='Minimum value of the truck number',
                                                  default=1)
    # truck max number, 如果没有指定就是160
    algo_based_fleet_ablation_parser.add_argument('-M', '--max', type=str, required=False, help='Maximum value of the truck number',
                                                  default=160)


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
    if args.command == 'scene_based_fleet_ablation':
        run_scene_based_fleet_ablation_experiment(config_file=args.config_file, min_truck=args.min, max_truck=args.max)
    if args.command == 'algo_based_fleet_ablation':
        run_algo_based_fleet_ablation_experiment(config_dir=args.config_dir, baseline=args.baseline, target=args.target,
                                                 min_truck=args.min, max_truck=args.max)



if __name__ == "__main__":
    config_path = sys.argv[1]
    run_simulation(config_file=config_path)


