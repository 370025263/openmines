import argparse
import json
import pathlib
import sys
import pkgutil
import importlib
import time
import re
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rich.console import Console

from openmines.src.utils.visualization.charter import Charter
from openmines.src.utils.visualization.graphher import VisualGrapher
from openmines.src.utils.analyzer import LogAnalyzer

# add the openmines package to the path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent.absolute()))

from openmines.src.mine import Mine
from openmines.src.truck import Truck
from openmines.src.charging_site import ChargingSite
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.dump_site import DumpSite, Dumper
from openmines.src.road import Road
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.dispatch_algorithms import *

# 添加包路径到sys.path
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设当前文件在 openmines/src/cli 目录下
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    l2d_road_matrix = np.array(config['road']['l2d_road_matrix'])
    d2l_road_matrix = np.array(config['road']['d2l_road_matrix'])
    road_event_params = config['road'].get('road_event_params', {})  # 从配置中加载道路事件参数

    charging_to_load_road_matrix = config['road']['charging_to_load_road_matrix']
    road = Road(l2d_road_matrix=l2d_road_matrix, d2l_road_matrix=d2l_road_matrix, 
                charging_to_load_road_matrix=charging_to_load_road_matrix,
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
    dispatchers_package = 'openmines.src.dispatch_algorithms'
    dispatchers_list = []
    
    # 在导入之前添加调试信息
    dispatcher_type = config['dispatcher']['type'][0]  # 假设至少有一个调度器
    print(f"Trying to import dispatcher: {dispatcher_type}")

    # 检查模块是否存在
    try:
        import inflection
        module_name_guess = inflection.underscore(dispatcher_type)
        module_path = f"openmines.src.dispatch_algorithms.{module_name_guess}"
        spec = importlib.util.find_spec(module_path)
        if spec:
            print(f"Module {module_path} path: {spec.origin}")
        else:
            print(f"Module {module_path} does not exist!")
    except Exception as e:
        print(f"Error checking module: {e}")

    # 从配置文件中获取需要导入的调度器类型
    dispatcher_types = config['dispatcher']['type']
    
    # 如果没有找到任何调度器，记录在这个变量中
    missing_dispatchers = []
    required_packages = {}

    # 只导入配置文件中指定的调度器
    for dispatcher_type in dispatcher_types:
        dispatcher_found = False
        
        # 第1种方法：尝试直接从总包中导入类
        try:
            # 直接从总包导入类
            module = importlib.import_module('openmines.src.dispatch_algorithms')
            if hasattr(module, dispatcher_type):
                dispatcher_class = getattr(module, dispatcher_type)
                dispatcher = dispatcher_class()
                dispatchers_list.append(dispatcher)
                dispatcher_found = True
                print(f"Successfully imported dispatcher: {dispatcher_type} (directly from package)")
                continue
        except Exception as e:
            # 记录可能需要的包
            if "No module named" in str(e):
                required_package = str(e).split("'")[1]
                required_packages[dispatcher_type] = required_package
        
        # 第2种方法：尝试根据类名猜测模块名
        if not dispatcher_found:
            try:
                import inflection
                module_name_guess = inflection.underscore(dispatcher_type)
                potential_module_path = f"openmines.src.dispatch_algorithms.{module_name_guess}"
                
                module = importlib.import_module(potential_module_path)
                if hasattr(module, dispatcher_type):
                    dispatcher_class = getattr(module, dispatcher_type)
                    dispatcher = dispatcher_class()
                    dispatchers_list.append(dispatcher)
                    dispatcher_found = True
                    print(f"Successfully imported dispatcher: {dispatcher_type} (from submodule)")
                    continue
            except ImportError as e:
                print(f"Warning: Could not import module {potential_module_path}")
                # 记录可能需要的包
                if "No module named" in str(e):
                    required_package = str(e).split("'")[1]
                    required_packages[dispatcher_type] = required_package
        
        # 第3种方法：遍历所有子模块查找
        if not dispatcher_found:
            print(f"Trying to find dispatcher {dispatcher_type} in all submodules")
            try:
                base_module = importlib.import_module('openmines.src.dispatch_algorithms')
                module_found = False
                
                for _, sub_module_name, _ in pkgutil.iter_modules(base_module.__path__, base_module.__name__ + '.'):
                    try:
                        sub_module = importlib.import_module(sub_module_name)
                        if hasattr(sub_module, dispatcher_type):
                            dispatcher_class = getattr(sub_module, dispatcher_type)
                            dispatcher = dispatcher_class()
                            dispatchers_list.append(dispatcher)
                            print(f"Successfully imported dispatcher: {dispatcher_type} (from submodule {sub_module_name})")
                            module_found = True
                            break
                    except Exception as e:
                        # 记录模块错误，并获取缺少的包
                        print(f"Error importing submodule {sub_module_name}: {e}")
                        if "No module named" in str(e):
                            required_package = str(e).split("'")[1]
                            required_packages[dispatcher_type] = required_package
                
                if not module_found:
                    missing_dispatchers.append(dispatcher_type)
            except Exception as e:
                print(f"Error: Failed to import base module: {e}")
                missing_dispatchers.append(dispatcher_type)

    # 如果有未找到的调度器，终止程序并报错
    if missing_dispatchers:
        error_message = f"Error: Could not import the following dispatchers: {', '.join(missing_dispatchers)}"
        
        # 如果我们知道缺少的包，提供安装建议
        if required_packages:
            error_message += "\nMissing required packages:"
            for disp, package in required_packages.items():
                error_message += f"\n  - For {disp}: {package}"
            error_message += "\n\nPlease install the required packages using pip:"
            error_message += "\npip install " + " ".join(set(required_packages.values()))
        
        print(error_message)
        sys.exit(1)  # 终止程序，返回错误码

    # 如果没有找到任何调度器，终止程序
    if not dispatchers_list:
        print(f"Error: No dispatchers found for types: {', '.join(dispatcher_types)}")
        sys.exit(1)

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
    1) 遍历 config_dir 中场景文件。对每个场景，在[min_truck, max_truck]之间等分若干车队规模点。
    2) 分别跑 baseline、target，得到产量并计算 ratio = target / baseline。
    3) 每个场景 -> 一条线 (x=车队规模, y=ratio)。
    English summary:
    - For each scenario file in config_dir, vary total trucks from min_truck to max_truck.
    - Run baseline & target, get produced tons, ratio = target / baseline.
    - We plot exactly one line per scenario: x=fleet size, y=ratio.
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

    minT = int(min_truck)
    maxT = int(max_truck)
    if maxT < minT:
        minT, maxT = maxT, minT

    if maxT == minT:
        fleet_sizes = [minT]
    else:
        step = (maxT - minT) / 9
        fleet_sizes = [int(math.floor(minT + i * step)) for i in range(10)]

    # find dispatcher classes
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

    # Collect each scenario => { 'fleet_sizes': [...], 'ratios': [...] }
    scenes_data = {}

    for cfg_path in file_list:
        scene_name = cfg_path.stem
        conf = load_config(cfg_path)
        truck_info = conf["charging_site"]["trucks"]
        orig_counts = [t_["count"] for t_ in truck_info]
        total_orig = sum(orig_counts)
        ratios_arr = [c / total_orig for c in orig_counts]

        sc_data = {
            'fleet_sizes': [],
            'ratios': []
        }

        for fs in fleet_sizes:
            new_conf = copy.deepcopy(conf)
            assigned = 0
            for i, t_ in enumerate(new_conf["charging_site"]["trucks"]):
                new_ct = int(math.floor(fs * ratios_arr[i]))
                assigned += new_ct
                t_["count"] = new_ct
            leftover = fs - assigned
            if leftover > 0:
                new_conf["charging_site"]["trucks"][-1]["count"] += leftover

            # baseline
            b_obj = baseline_class()
            b_ticks = run_dispatch_sim(b_obj, new_conf)
            produced_b = 0.0
            for td in b_ticks.values():
                if 'mine_states' in td:
                    produced_b = td['mine_states']['produced_tons']

            # target
            t_obj = target_class()
            t_ticks = run_dispatch_sim(t_obj, new_conf)
            produced_t = 0.0
            for td in t_ticks.values():
                if 'mine_states' in td:
                    produced_t = td['mine_states']['produced_tons']

            ratio_val = 0.0
            if produced_b > 1e-9:
                ratio_val = produced_t / produced_b

            sc_data['fleet_sizes'].append(fs)
            sc_data['ratios'].append(ratio_val)

        scenes_data[scene_name] = sc_data

    c = Charter(str(config_dir))
    c.draw_algo_based_fleet_ablation_experiment(scenes_data, baseline, target)
    c.save_ablation(tag="algo_ablation")

def run_analysis(log_path, dispatcher_name=None, model_name=None, language="English"):
    """Run log analysis"""
    console = Console()
    try:
        # 从环境变量获取API配置
        api_key = os.getenv('OPENAI_API_KEY').strip()
        
        api_base = os.getenv('OPENAI_API_BASE')
        # 如果命令行参数中指定了模型，则使用指定的模型，否则使用环境变量或默认值
        model = model_name or os.getenv('OPENAI_MODEL_NAME', 'deepseek-ai/DeepSeek-V3')  # 默认模型名称
        print(f"Api key: {api_key}, Api base: {api_base}, Model: {model}, Language: {language}")
        if not api_key or not api_base or not model:
            console.print("[bold red]Error: Please set environment variables OPENAI_API_KEY, OPENAI_API_BASE, and OPENAI_MODEL_NAME[/bold red]")
            return
            
        # 检查路径是否存在
        path = pathlib.Path(log_path)
        if not path.exists():
            console.print(f"[bold red]Error: Path {log_path} does not exist[/bold red]")
            return

        # 如果是目录，分析最新的日志文件
        if path.is_dir():
            log_files = list(path.glob("*.log"))
            if not log_files:
                console.print(f"[bold red]Error: No log files found in directory {log_path}[/bold red]")
                return
            # 按修改时间排序，取最新的
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            log_path = str(latest_log)
            console.print(f"[bold blue]Analyzing latest log file: {latest_log.name}[/bold blue]")
        
        # 初始化分析器
        analyzer = LogAnalyzer(
            api_key=api_key,
            api_base=api_base,
            model_name=model,
            language=language
        )
        
        if dispatcher_name:
            analyzer.dispatcher_name = dispatcher_name
            
        # 运行分析
        analysis_result = analyzer.analyze_logs(log_path)
        
        # 如果分析结果为空（可能是因为未找到指定的dispatcher），则直接返回
        if not analysis_result:
            return
            
        # 获取当前时间字符串
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        
        # 从日志文件名中提取仿真时间
        sim_time_match = re.search(r'sim_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', os.path.basename(log_path))
        sim_time = sim_time_match.group(1) if sim_time_match else "unknown_sim_time"
        
        # 构造输出文件名
        if dispatcher_name:
            output_filename = f"analysis_report_{dispatcher_name}_{sim_time}_analyzed_{current_time}.md"
        else:
            output_filename = f"analysis_report_all_{sim_time}_analyzed_{current_time}.md"
        
        # 保存结果到当前工作目录
        output_file = pathlib.Path.cwd() / output_filename
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Mining Truck Dispatch Analysis Report\n\n")
            f.write(f"- Analysis Time: {current_time}\n")
            f.write(f"- Simulation Time: {sim_time}\n")
            if dispatcher_name:
                f.write(f"- Dispatch Algorithm: {dispatcher_name}\n")
            f.write("\n---\n\n")
            f.write(analysis_result)
        
        console.print(f"[bold green]✓ Analysis complete, results saved to {output_file}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")

def main():
    parser = argparse.ArgumentParser(
        description='OpenMines: A Light and Comprehensive Mining Simulation Environment for Truck Dispatching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run a simulation with configuration file
  openmines -f config.json
  
  # Visualize simulation results
  openmines -v result.json
  
  # Analyze logs (will use the latest log file if directory is specified)
  openmines -a logs/
  
  # Analyze logs with specific model
  openmines -a logs/ --model "gpt-4"
  
  # Analyze logs in Chinese
  openmines -a logs/ --language "Chinese"
  
  # Analyze logs for a specific dispatcher
  openmines analyze logs/ -d "MyDispatcher"
  
  # Analyze logs with specific model and language
  openmines analyze logs/ -m "gpt-4" -l "Chinese"
  
  # Run fleet size ablation experiment on a single scene
  openmines scene_based_fleet_ablation -f config.json -m 10 -M 50
  
  # Run fleet size ablation experiment comparing algorithms across multiple scenes
  openmines algo_based_fleet_ablation -d configs/ -b NaiveDispatcher -t MyDispatcher -m 10 -M 100

Analysis Usage:
  The analyzer uses LLM to analyze simulation logs and generate insights.
  
  Required environment variables:
    - OPENAI_API_KEY: Your API key
    - OPENAI_API_BASE: API base URL (e.g., https://api.siliconflow.cn)
    - OPENAI_MODEL_NAME: Default model name (optional, can be overridden with --model)
  
  Optional parameters:
    - --language: Language for the analysis report (default: English)
  
  The analysis report will be saved as a Markdown file in the current directory.
        ''')
    
    subparsers = parser.add_subparsers(help='commands', dest='command')

    # 直接访问入口 openmines -f config.json, openmines -v tick.json
    parser.add_argument('-f', '--config-file', type=str, 
                       help='Run simulation with specified config file. Results will be saved in $CWD/result/')
    parser.add_argument('-v', '--tick-file', type=str, 
                       help='Visualize simulation results from tick file. Will generate a GIF in $CWD/result/')
    parser.add_argument('-a', '--analyze', type=str,
                       help='Analyze log file or directory. If directory is specified, will analyze the latest log file')
    parser.add_argument('--model', type=str,
                       help='Specify the model to use for analysis')
    parser.add_argument('--language', type=str, default='English',
                       help='Specify the language for the analysis report (default: English)')

    # add command 'run'
    run_parser = subparsers.add_parser('run', 
                                      help='Run a simulation experiment',
                                      description='Run a simulation with the specified configuration file')
    run_parser.add_argument('-f', '--config-file', type=str, required=True, 
                           help='Path to the config file')

    # add command visualize
    visualize_parser = subparsers.add_parser('visualize', 
                                            help='Visualize a simulation experiment',
                                            description='Create an animation from simulation results')
    visualize_parser.add_argument('-f', '--tick-file', type=str, required=True, 
                                help='Path to the simulation tick file')

    # 在单个scenario中，不同算法在不同车队大小中的消融实验
    scene_based_fleet_ablation_parser = subparsers.add_parser('scene_based_fleet_ablation', 
        help='Run fleet size ablation experiment on a single scene',
        description='Test algorithm performance by varying truck scales in the same scene')
    scene_based_fleet_ablation_parser.add_argument('-f', '--config-file', type=str, required=True, 
                                                  help='Path to the config file')
    scene_based_fleet_ablation_parser.add_argument('-m', '--min', type=str, required=True, 
                                                  help='Minimum number of trucks')
    scene_based_fleet_ablation_parser.add_argument('-M', '--max', type=str, required=True, 
                                                  help='Maximum number of trucks')

    # 在不同scenario中，目标算法和基线算法在不同车队大小中的消融实验
    algo_based_fleet_ablation_parser = subparsers.add_parser('algo_based_fleet_ablation', 
        help='Compare algorithms across multiple scenes with varying fleet sizes',
        description='Compare target algorithm against baseline across multiple scenes and fleet sizes')
    algo_based_fleet_ablation_parser.add_argument('-d', '--config-dir', type=str, required=True, 
                                                 help='Directory containing multiple config files')
    algo_based_fleet_ablation_parser.add_argument('-b', '--baseline', type=str, required=True, 
                                                 help='Name of the baseline algorithm')
    algo_based_fleet_ablation_parser.add_argument('-t', '--target', type=str, required=True, 
                                                 help='Name of the target algorithm to compare')
    algo_based_fleet_ablation_parser.add_argument('-m', '--min', type=str, required=False, 
                                                 help='Minimum number of trucks (default: 1)',
                                                 default=1)
    algo_based_fleet_ablation_parser.add_argument('-M', '--max', type=str, required=False, 
                                                 help='Maximum number of trucks (default: 160)',
                                                 default=160)

    # 作为子命令的分析功能
    analyze_parser = subparsers.add_parser('analyze', 
                                          help='Analyze simulation logs',
                                          description='Extract and analyze data from simulation logs')
    analyze_parser.add_argument('log_path', type=str, 
                              help='Path to the log file or directory (will use latest log if directory)')
    analyze_parser.add_argument('-d', '--dispatcher', type=str, 
                              help='Specify dispatcher name to analyze')
    analyze_parser.add_argument('-m', '--model', type=str, 
                              help='Specify model name to use for analysis')
    analyze_parser.add_argument('-l', '--language', type=str, default='English',
                              help='Specify language for the analysis report (default: English)')

    args = parser.parse_args()
    
    # 处理分析命令
    if args.analyze is not None:
        # 安全获取参数
        dispatcher_name = getattr(args, 'dispatcher', None)
        model_name = args.model
        language = args.language
        run_analysis(args.analyze, dispatcher_name, model_name, language)
        return
        
    # 检查子命令
    if args.command == 'analyze':
        dispatcher_name = getattr(args, 'dispatcher', None)
        model_name = getattr(args, 'model', None)
        language = getattr(args, 'language', 'English')
        run_analysis(args.log_path, dispatcher_name, model_name, language)
        return
        
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


