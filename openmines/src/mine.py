"""模拟的核心对象Mine

使用的时候需要先初始化Mine，
然后添加
LoadSite(Shovel)、DumpSite(Dumper)、
ChargingSite(Truck)，Road，Dispatcher等对象
最后调用Mine.run()方法开始仿真
"""
import glob
import os
import time
from datetime import datetime

import numpy as np
import simpy,logging,math
from functools import reduce
from multiprocessing import Queue

from openmines.src.charging_site import ChargingSite
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.dump_site import DumpSite
from openmines.src.load_site import LoadSite
from openmines.src.road import Road
from openmines.src.truck import Truck
from openmines.src.utils.logger import MineLogger, LOG_FILE_PATH
from openmines.src.utils.ticker import TickGenerator
from openmines.src.utils.event import EventPool

class Mine:
    def __init__(self, name:str, log_path=LOG_FILE_PATH, log_file_level=logging.DEBUG, log_console_level=logging.INFO, seed=42):
        self.env = simpy.Environment()
        self.name = name
        self.load_sites = []
        self.dump_sites = []
        self.trucks = []
        self.road = None
        self.dispatcher = None
        self.random_event_pool = EventPool()  # 随机事件池
        # rl
        self.done = False
        # summary
        self.produce_tons = 0  # the produced tons of this dump site
        self.service_count = 0  # the number of shovel-vehicle cycle in this dump site
        self.status = dict()  # the status of shovel
        # LOGGER配置
        # print(log_path)
        self.global_logger = MineLogger(log_path=log_path, file_level=log_file_level, console_level=log_console_level)
        self.mine_logger = self.global_logger.get_logger(name)

    def monitor_status(self, env, monitor_interval=1):
        """监控卸载区的产量、服务次数等信息
            监控道路的状态
            Trigger BreakDown for truck
            Trigger RoadEvent:jam
        """
        while True:
            """
            1.监控卸载区的产量、服务次数等信息
            """
            # 获取每个dumper信息并统计
            self.produce_tons = sum([dump_site.produce_tons for dump_site in self.dump_sites])
            self.service_count = sum([dump_site.service_count for dump_site in self.dump_sites])
            # 获取卡车整体的统计信息
            working_truck_count = 0
            waiting_truck_count = 0
            load_unload_truck_count = 0
            moving_truck_count = 0
            repairing_truck_count = 0
            for truck in self.trucks:
                # 计算可工作的truck数量
                if truck.status != "unrepairable":
                    working_truck_count += 1
                # 计算正在排队中的truck数量
                if "waiting" in truck.status:
                    waiting_truck_count += 1
                # 计算正在装载卸载的truck数量
                if truck.status == "loading" or truck.status == "unloading":
                    load_unload_truck_count += 1
                # 计算moving的truck数量
                if truck.status == "moving":
                    moving_truck_count += 1
                # 计算正在维修的truck数量
                if truck.status == "repairing":
                    repairing_truck_count += 1
            # 计算出现的random event的数量
            road_jam_count = 0
            road_repair_count = 0
            truck_repair = 0
            truck_unrepairable = 0
            random_event_count = 0
            for event_key in self.random_event_pool.event_set.keys():
                event_type = self.random_event_pool.event_set[event_key].event_type
                if event_type == "RoadEvent:jam":
                    road_jam_count += 1
                if event_type == "RoadEvent:repair":
                    road_repair_count += 1
                if event_type == "TruckEvent:breakdown":
                    truck_repair += 1
                if event_type == "TruckEvent:unrepairable":
                    truck_unrepairable += 1
            random_event_count = road_jam_count + road_repair_count + truck_repair + truck_unrepairable
            self.status[int(env.now)] = {
                # KPIs
                "produced_tons": self.produce_tons,
                "service_count": self.service_count,
                # stats
                "truck_count": len(self.trucks),
                "working_truck_count": working_truck_count,
                "waiting_truck_count": waiting_truck_count,
                "load_unload_truck_count": load_unload_truck_count,
                "moving_truck_count": moving_truck_count,
                "repairing_truck_count": repairing_truck_count,
                # event stats
                "road_jam_count": road_jam_count,
                "road_repair_count": road_repair_count,
                "truck_repair": truck_repair,
                "truck_unrepairable": truck_unrepairable,
                "random_event_count":random_event_count
            }
            self.status["cur"] = self.status[int(env.now)]
            """
            2.监控道路的状态
            """
            self.update_road_status()
            # 等待下一个监控时间点
            """
            3.更新矿车的breakdown(repair)状态
            """
            for truck in self.trucks:
                truck.sample_breakdown()

            """
            4.在每个单位时间内，根据道路的交通流，触发堵车事件并存入池中.
                每个矿车在出发的时候就会受到道路堵车事件的影响(truck.py)
            """
            # 遍历每一个道路
            ## Init Road
            for i in range(self.road.load_site_num):
                charging_site = self.charging_site
                load_site = self.load_sites[i]
                # 从charging到load的道路
                self.road.road_jam_sampling(start=charging_site, end=load_site)
            ## Haul Road
            for i in range(self.road.load_site_num):
                for j in range(self.road.dump_site_num):
                    load_site = self.load_sites[i]
                    dump_site = self.dump_sites[j]
                    self.road.road_jam_sampling(start=load_site, end=dump_site)
            ## Unhaul Road
            for j in range(self.road.dump_site_num):
                for i in range(self.road.load_site_num):
                    load_site = self.load_sites[i]
                    dump_site = self.dump_sites[j]
                    self.road.road_jam_sampling(start=dump_site, end=load_site)
            yield env.timeout(monitor_interval)

    def update_road_status(self):
        cur_time = self.env.now
        road = self.road
        road_status = dict()

        # 【优化1】一次性获取 jam 和 repair 事件，并构建基于 (start_location, end_location) 的统计字典
        jam_events_dict = {}
        repair_events_dict = {}

        jam_events = self.random_event_pool.get_even_by_type("RoadEvent:jam")
        repair_events = self.random_event_pool.get_even_by_type("RoadEvent:repair")

        for jam_event in jam_events:
            start_loc = jam_event.info["start_location"]
            end_loc = jam_event.info["end_location"]
            if jam_event.info["start_time"] <= cur_time and jam_event.info["est_end_time"] >= cur_time:
                jam_events_dict[(start_loc, end_loc)] = jam_events_dict.get((start_loc, end_loc), 0) + 1

        for repair_event in repair_events:
            start_loc = repair_event.info["start_location"]
            end_loc = repair_event.info["end_location"]
            if repair_event.info["repair_start_time"] <= cur_time and repair_event.info["repair_end_time"] >= cur_time:
                repair_events_dict[(start_loc, end_loc)] = repair_events_dict.get((start_loc, end_loc), 0) + 1

        # 【优化2】一次性统计卡车所在道路
        truck_count_dict = {}
        for truck in self.trucks:
            if not truck.current_location or not truck.target_location:
                continue
            pair = (truck.current_location.name, truck.target_location.name)
            truck_count_dict[pair] = truck_count_dict.get(pair, 0) + 1

        # 下方三个循环逻辑与原来一致，只是由预先构造的 jam/repair/truck 字典来统计
        # 1. charging_site -> load_sites
        for i in range(self.road.load_site_num):
            charging_site_name = self.charging_site.name
            load_site_name = self.load_sites[i].name
            road_status_key = (charging_site_name, load_site_name)
            road_status[road_status_key] = {
                "truck_jam_count": jam_events_dict.get(road_status_key, 0),
                "repair_count": repair_events_dict.get(road_status_key, 0),
                "truck_count": truck_count_dict.get(road_status_key, 0)
            }

        # 2. load_sites -> dump_sites
        for i in range(self.road.load_site_num):
            for j in range(self.road.dump_site_num):
                load_site_name = self.load_sites[i].name
                dump_site_name = self.dump_sites[j].name
                road_status_key = (load_site_name, dump_site_name)
                road_status[road_status_key] = {
                    "truck_jam_count": jam_events_dict.get(road_status_key, 0),
                    "repair_count": repair_events_dict.get(road_status_key, 0),
                    "truck_count": truck_count_dict.get(road_status_key, 0)
                }

        # 3. dump_sites -> load_sites
        for j in range(self.road.dump_site_num):
            for i in range(self.road.load_site_num):
                dump_site_name = self.dump_sites[j].name
                load_site_name = self.load_sites[i].name
                road_status_key = (dump_site_name, load_site_name)
                road_status[road_status_key] = {
                    "truck_jam_count": jam_events_dict.get(road_status_key, 0),
                    "repair_count": repair_events_dict.get(road_status_key, 0),
                    "truck_count": truck_count_dict.get(road_status_key, 0)
                }

        self.road.road_status = road_status

    def add_load_site(self, load_site:LoadSite):
        load_site.set_env(self.env)
        self.load_sites.append(load_site)

    def add_charging_site(self, charging_site:ChargingSite):
        charging_site.set_mine(self)
        self.charging_site = charging_site
        self.trucks = charging_site.trucks
        for truck in self.trucks:
            truck.set_env(self)

    def add_dump_site(self, dump_site:DumpSite):
        dump_site.set_env(self.env)
        self.dump_sites.append(dump_site)

    def add_road(self, road:Road):
        assert road is not None, "road can not be None"
        road.set_env(self)
        self.road = road

    def add_dispatcher(self, dispatcher:BaseDispatcher):
        assert dispatcher is not None, "dispatcher can not be None"
        self.dispatcher = dispatcher

    def get_load_site_status(self):
        for load_site in self.load_sites:
            print(f'{load_site.name} has {len(load_site.res.users)} trucks')

    def get_dest_index_by_name(self,name:str):
        """
        使用名字去匹配铲子和卸载区 并返回其标号
        标号定义如下：
        None 没有匹配
        0～N-1 装载区标号
        N～N+M-1 卸载区标号

        :param name:
        :return:
        """
        assert name is not None, "name can not be None"
        # 查找装载区
        for i,loadsite in enumerate(self.load_sites):
            if loadsite.name == name:
                return i
        # 查找卸载区
        for i,dumpsite in enumerate(self.dump_sites):
            if dumpsite.name == name:
                return i+len(self.load_sites)
        return None

    def get_dest_obj_by_index(self,index:int):
        """
        使用标号去匹配铲子和卸载区 并返回其对象
        :param index:
        :return:
        """
        assert index is not None, "index can not be None"
        if index < len(self.load_sites):
            return self.load_sites[index]
        elif index < len(self.load_sites) + len(self.dump_sites):
            return self.dump_sites[index - len(self.load_sites)]
        else:
            return None

    def get_dest_obj_by_name(self,name:str):
        """
        使用名字去匹配装载区和卸载区 并返回其对象
        :param name:
        :return:
        """
        assert name is not None, "name can not be None"
        # 查找装载区
        for loadsite in self.load_sites:
            if loadsite.name == name:
                return loadsite
        # 查找卸载区
        for dumpsite in self.dump_sites:
            if dumpsite.name == name:
                return dumpsite
        # 查找充电站
        if self.charging_site.name == name:
            return self.charging_site
        return None

    def get_service_vehicle_by_name(self,name:str):
        """
        使用名字匹配铲子和dumper并返回其对象
        :param name:
        :return:
        """
        assert name is not None, "name can not be None"
        for loadsite in self.load_sites:
            for shovel in loadsite.shovel_list:
                if shovel.name == name:
                    return shovel
        for dumpsite in self.dump_sites:
            for dumper in dumpsite.dumper_list:
                if dumper.name == name:
                    return dumper
        return None

    def summary(self):
        """在仿真结束后，统计各个装载区和卸载区的产量、MatchingFactor等并记录在文件中
        目前统计的对象：
        1. 卸载区产量
        2. TruckCycleTime
        3. MatchingFactor
        4. TruckWaitTime、TotalWaitTime

        :return:
        """
        pass

    def start(self, total_time:float=60*8)->dict:
        """
        普通策略算法的仿真入口(包括RL策略推理时)
        :param total_time:
        :return:
        """
        assert self.road is not None, "road can not be None"
        assert self.dispatcher is not None, "dispatcher can not be None"
        assert self.charging_site is not None, "charging_site can not be None"
        assert len(self.load_sites) > 0, "load_sites can not be empty"
        assert len(self.dump_sites) > 0, "dump_sites can not be empty"
        assert len(self.trucks) > 0, "trucks can not be empty"
        assert total_time > 0, "total_time can not be negative"
        self.total_time = total_time
        self.mine_logger.info(f"simulation started with dispatcher {self.dispatcher.__class__.__name__}")

        # start some monitor process for summary
        for load_site in self.load_sites:
            # 对停车场队列的监控
            self.env.process(load_site.parking_lot.monitor_resources(env=self.env,
                                                                    resources=[shovel.res for shovel in load_site.shovel_list],
                                                                     res_objs=load_site.shovel_list))
            for shovel in load_site.shovel_list:
                self.env.process(load_site.parking_lot.monitor_resource(env=self.env,res_obj=shovel,
                                                                        resource=shovel.res))
            # 对铲车产出的监控
            for shovel in load_site.shovel_list:
                self.env.process(shovel.monitor_status(env=self.env))
            # 对装载区产出、装载区队列的监控
            self.env.process(load_site.monitor_status(env=self.env))

        for dump_site in self.dump_sites:
            # 对停车场队列的监控
            self.env.process(dump_site.parking_lot.monitor_resources(env=self.env,
                                                                    resources=[dumper.res for dumper in dump_site.dumper_list],
                                                                     res_objs=dump_site.dumper_list))
            for dumper in dump_site.dumper_list:
                self.env.process(dump_site.parking_lot.monitor_resource(env=self.env,res_obj=dumper,
                                                                        resource=dumper.res))
            # 对卸载区产出、卸载区队列的监控
            self.env.process(dump_site.monitor_status(env=self.env))
            # 对dumper产出的监控
            for dumper in dump_site.dumper_list:
                self.env.process(dumper.monitor_status(env=self.env))
        # 对矿山整体监控
        self.env.process(self.monitor_status(env=self.env))

        # log in the truck as process
        for truck in self.trucks:
            self.env.process(truck.run())

        self.env.run(until=total_time)
        self.mine_logger.info(f"simulation finished with dispatcher {self.dispatcher.__class__.__name__}")
        self.summary()
        ticks = self.dump_frames(total_time=total_time)
        return ticks

    def start_rl(self, obs_queue:Queue, act_queue:Queue, reward_mode:str = "dense", total_time:float=60*8, ticks:bool=False)->dict:
        """
        使用RL算法的仿真入口
        :param total_time:
        :return:
        """
        assert self.road is not None, "road can not be None"
        assert self.dispatcher is not None, "dispatcher can not be None"
        assert self.charging_site is not None, "charging_site can not be None"
        assert len(self.load_sites) > 0, "load_sites can not be empty"
        assert len(self.dump_sites) > 0, "dump_sites can not be empty"
        assert len(self.trucks) > 0, "trucks can not be empty"
        assert total_time > 0, "total_time can not be negative"
        self.total_time = total_time
        self.mine_logger.info("simulation started")
        # pass queue to dispatcher
        self.dispatcher.obs_queue = obs_queue
        self.dispatcher.act_queue = act_queue

        # start some monitor process for summary
        for load_site in self.load_sites:
            # 对停车场队列的监控
            self.env.process(load_site.parking_lot.monitor_resources(env=self.env,
                                                                     resources=[shovel.res for shovel in
                                                                                load_site.shovel_list],
                                                                     res_objs=load_site.shovel_list))
            for shovel in load_site.shovel_list:
                self.env.process(load_site.parking_lot.monitor_resource(env=self.env, res_obj=shovel,
                                                                        resource=shovel.res))
            # 对铲车产出的监控
            for shovel in load_site.shovel_list:
                self.env.process(shovel.monitor_status(env=self.env))
            # 对装载区产出、装载区队列的监控
            self.env.process(load_site.monitor_status(env=self.env))

        for dump_site in self.dump_sites:
            # 对停车场队列的监控
            self.env.process(dump_site.parking_lot.monitor_resources(env=self.env,
                                                                     resources=[dumper.res for dumper in
                                                                                dump_site.dumper_list],
                                                                     res_objs=dump_site.dumper_list))
            for dumper in dump_site.dumper_list:
                self.env.process(dump_site.parking_lot.monitor_resource(env=self.env, res_obj=dumper,
                                                                        resource=dumper.res))
            # 对卸载区产出、卸载区队列的监控
            self.env.process(dump_site.monitor_status(env=self.env))
            # 对dumper产出的监控
            for dumper in dump_site.dumper_list:
                self.env.process(dumper.monitor_status(env=self.env))
        # 对矿山整体监控
        self.env.process(self.monitor_status(env=self.env))
        # log in the truck as process
        for truck in self.trucks:
            self.env.process(truck.run())
        self.env.run(until=total_time)  # 在这里开一个独立的进程用于执行函数，后面大妈

        # 当模拟结束的时候 最后发送一个ob和done等信息
        ob = self.dispatcher.current_observation
        info = ob["info"]
        if reward_mode == "dense":
            reward = self.dispatcher._get_reward_dense(self)
        elif reward_mode == "sparse":
           reward = self.dispatcher._get_reward_dense(self)
        else:
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        done = True
        self.env.done = done
        trucated = False
        out = {
            "ob": ob,
            "info": info,
            "reward": reward,
            "truncated": trucated,
            "done": done
        }
        self.mine_logger.info("simulation finished")
        self.summary()
        if ticks:
            self.dump_frames(total_time=total_time, rl=True)
        obs_queue.put(out, timeout=5)  # 将观察值放入队列

    def dump_frames(self, total_time, rl=False):
        """使用TickGenerator记录仿真过程中的数据
        将数据写入到文件中
        :return:
        """
        # tick generator
        self.tick_generator = TickGenerator(mine=self, tick_num=total_time)
        assert self.tick_generator is not None, "tick_generator can not be None"

        print("dumping frames...")
        self.tick_generator.run()

        # 获得年月日时分秒的字符串表示
        time_str = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

        if rl:
            # 使用指定的TickGenerator.result_path作为文件保存路径
            result_path = self.tick_generator.result_path
            if not os.path.exists(result_path):
                os.makedirs(result_path)  # 如果路径不存在，则创建

            # 查找当前目录下所有相关的文件，使用绝对路径进行查找
            file_pattern = os.path.join(result_path, f'MINE-{self.name}-EP-*.json')
            files = glob.glob(file_pattern)

            # 解析文件名中的时间戳并排序
            episodes = []
            for file in files:
                try:
                    # 假设文件名格式为 'MINE—{self.name}-EP-{episode}-TIME-{time_str}.json'
                    filename = os.path.basename(file)  # 只解析文件名部分
                    parts = filename.split('-')
                    episode = int(parts[3])  # 提取回合数
                    file_time_str = '-'.join(parts[-5:]).split('.')[0]  # 提取时间戳
                    file_time = datetime.strptime(file_time_str, "%Y-%m-%d %H-%M-%S")
                    episodes.append((episode, file_time))
                except (ValueError, IndexError):
                    continue  # 文件名格式不正确，跳过

            # 按时间戳排序并推断当前回合数
            current_episode = 1
            if episodes:
                episodes.sort(key=lambda x: x[1])
                current_episode = episodes[-1][0] + 1

            # 写入RL仿真数据文件
            ticks = self.tick_generator.write_to_file(
                file_name=f'MINE-{self.name}-EP-{current_episode}-TIME-{time_str}.json')
        else:
            # 非RL部分代码保持不变
            ticks = self.tick_generator.write_to_file(
                file_name=f'MINE-{self.name}-ALGO-{self.dispatcher.name}-TIME-{time_str}.json')

        return ticks

    @property
    def match_factor(self):
        # 统计MatchingFactor
        # paper:Match factor for heterogeneous truck and loader fleets
        shovels = [shovel for load_site in self.load_sites for shovel in load_site.shovel_list]
        trucks = self.trucks
        truck_cycle_time_avg = np.mean([truck.truck_cycle_time for truck in trucks])

        num_trucks = len(self.trucks)
        num_shovels = len(shovels)
        loading_time: np.array = np.zeros((num_trucks, num_shovels))
        for i, truck in enumerate(trucks):
            for j, shovel in enumerate(shovels):
                loading_time[i, j] = round((truck.truck_capacity / shovel.shovel_tons),
                                           1) * shovel.shovel_cycle_time  # in mins

        # 按行按列去重复
        loading_time = np.unique(loading_time, axis=0)
        unique_loading_time = np.unique(loading_time, axis=1).astype(int)
        # 对异构铲车的子构数量进行统计
        shovel_type_count = dict()
        for i in range(unique_loading_time.shape[0]):  # truck type index
            int_data = np.array(loading_time[i, :]).astype(int)
            for value in set(int_data):
                shovel_type_count[f'{i}_{value}'] = list(int_data).count(
                    value)  # it means truck type i w.r.t shovel type num

        # unique_loading_time = np.ones_like(unique_loading_time) + unique_loading_time
        # 对每一行求lcm
        lcm_load_time = np.lcm.reduce(unique_loading_time, axis=1)
        upside_down_sum = 0
        for i in range(unique_loading_time.shape[0]):
            for j in range(unique_loading_time.shape[1]):
                upside_down_sum += shovel_type_count[f'{i}_{unique_loading_time[i, j]}'] * (
                            lcm_load_time[i] / unique_loading_time[i, j])
        match_factor = (num_trucks * np.sum(lcm_load_time)) / (upside_down_sum * truck_cycle_time_avg)
        return match_factor