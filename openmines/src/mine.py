"""模拟的核心对象Mine

使用的时候需要先初始化Mine，
然后添加
LoadSite(Shovel)、DumpSite(Dumper)、
ChargingSite(Truck)，Road，Dispatcher等对象
最后调用Mine.run()方法开始仿真
"""
import os
import time

import simpy,logging,math
from functools import reduce

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
    def __init__(self, name:str, log_path=LOG_FILE_PATH, log_file_level=logging.DEBUG, log_console_level=logging.INFO):
        self.env = simpy.Environment()
        self.name = name
        self.load_sites = []
        self.dump_sites = []
        self.trucks = []
        self.road = None
        self.dispatcher = None
        self.random_event_pool = EventPool()  # 随机事件池
        # summary
        self.produce_tons = 0  # the produced tons of this dump site
        self.service_count = 0  # the number of shovel-vehicle cycle in this dump site
        self.status = dict()  # the status of shovel
        # LOGGER配置
        print(log_path)
        self.global_logger = MineLogger(log_path=log_path, file_level=log_file_level, console_level=log_console_level)
        self.mine_logger = self.global_logger.get_logger(name)

    def monitor_status(self, env, monitor_interval=1):
        """监控卸载区的产量、服务次数等信息
            监控道路的状态
        """
        while True:
            """
            1.监控卸载区的产量、服务次数等信息
            """
            # 获取每个dumper信息并统计
            for dump_site in self.dump_sites:
                # 获取每个dumper信息并统计
                for dumper in dump_site.dumper_list:
                    self.produce_tons += dumper.dumper_tons
                    self.service_count += dumper.service_count
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
            # reset
            self.produce_tons = 0
            self.service_count = 0
            """
            2.监控道路的状态
            """
            cur_time = env.now
            # 获取mine的路网
            road = self.road
            charging_to_load_road = road.charging_to_load
            road_matrix = road.road_matrix
            # 获取每条路的状态
            road_status = dict()
            ## 统计初始化init过程中的道路情况
            for i in range(self.road.load_site_num):
                charging_site_name = self.charging_site.name
                load_site_name = self.load_sites[i].name
                road_status[(charging_site_name, load_site_name)] = {"truck_jam_count": 0, "repair_count": 0,
                                                                 "truck_count": 0}
                # 统计道路上当前正在发生的随机事件个数、历史上发生的随机事件个数
                ## 获取jam事件
                jam_events = self.random_event_pool.get_even_by_type("RoadEvent:jam")
                for jam_event in jam_events:
                    """
                    info={"name": self.name, "status": "jam", "speed": 0,
                                                          "start_location": self.current_location.name,
                                                            "end_location": self.target_location.name,
                                                          "start_time": self.env.now, "est_end_time":
                    """
                    if jam_event.info["start_location"] == charging_site_name and jam_event.info[
                        "end_location"] == load_site_name \
                            and jam_event.info["start_time"] <= cur_time and jam_event.info["est_end_time"] >= cur_time:
                        road_status[(charging_site_name, load_site_name)]["truck_jam_count"] += 1
                ## 获取repair事件
                repair_events = self.random_event_pool.get_even_by_type("RoadEvent:repair")
                for repair_event in repair_events:
                    if repair_event.info["start_location"] == charging_site_name and repair_event.info[
                        "end_location"] == load_site_name \
                            and repair_event.info["repair_start_time"] <= cur_time and repair_event.info[
                        "repair_end_time"] >= cur_time:
                        road_status[(charging_site_name, load_site_name)]["repair_count"] += 1
                ## 统计道路上的truck
                for truck in self.trucks:
                    if truck.current_location == charging_site_name and truck.target_location == load_site_name:
                        road_status[(charging_site_name, load_site_name)]["truck_count"] += 1

            ## 统计haul过程中的道路情况
            for i in range(self.road.load_site_num):
                for j in range(self.road.dump_site_num):
                    load_site_name = self.load_sites[i].name
                    dump_site_name = self.dump_sites[j].name
                    road_status[(load_site_name, dump_site_name)] = {"truck_jam_count": 0, "repair_count": 0, "truck_count": 0}

                    # 统计道路上当前正在发生的随机事件个数、历史上发生的随机事件个数
                    ## 获取jam事件
                    jam_events = self.random_event_pool.get_even_by_type("RoadEvent:jam")
                    for jam_event in jam_events:
                        """
                        info={"name": self.name, "status": "jam", "speed": 0,
                                                              "start_location": self.current_location.name,
                                                                "end_location": self.target_location.name,
                                                              "start_time": self.env.now, "est_end_time":
                        """
                        if jam_event.info["start_location"] == load_site_name and jam_event.info["end_location"] == dump_site_name \
                            and jam_event.info["start_time"] <= cur_time and jam_event.info["est_end_time"] >= cur_time:
                            road_status[(load_site_name, dump_site_name)]["truck_jam_count"] += 1
                    ## 获取repair事件
                    repair_events = self.random_event_pool.get_even_by_type("RoadEvent:repair")
                    for repair_event in repair_events:
                        if repair_event.info["start_location"] == load_site_name and repair_event.info["end_location"] == dump_site_name \
                            and repair_event.info["repair_start_time"] <= cur_time and repair_event.info["repair_end_time"] >= cur_time:
                            road_status[(load_site_name, dump_site_name)]["repair_count"] += 1
                    ## 统计道路上的truck
                    for truck in self.trucks:
                        if truck.current_location == load_site_name and truck.target_location == dump_site_name:
                            road_status[(load_site_name, dump_site_name)]["truck_count"] += 1

            ## 统计unhaul过程中的道路情况
            for j in range(self.road.dump_site_num):
                for i in range(self.road.load_site_num):
                    load_site_name = self.load_sites[i].name
                    dump_site_name = self.dump_sites[j].name
                    road_status[(dump_site_name,load_site_name)] = {"truck_jam_count": 0, "repair_count": 0, "truck_count": 0}

                    # 统计道路上当前正在发生的随机事件个数、历史上发生的随机事件个数
                    ## 获取jam事件
                    jam_events = self.random_event_pool.get_even_by_type("RoadEvent:jam")
                    for jam_event in jam_events:
                        if jam_event.info["start_location"] == dump_site_name and jam_event.info["end_location"] == load_site_name \
                            and jam_event.info["start_time"] <= cur_time and jam_event.info["est_end_time"] >= cur_time:
                            road_status[(dump_site_name,load_site_name)]["truck_jam_count"] += 1
                    ## 获取repair事件
                    repair_events = self.random_event_pool.get_even_by_type("RoadEvent:repair")
                    for repair_event in repair_events:
                        if repair_event.info["start_location"] == dump_site_name and repair_event.info["end_location"] == load_site_name \
                            and repair_event.info["repair_start_time"] <= cur_time and repair_event.info["repair_end_time"] >= cur_time:
                            road_status[(dump_site_name,load_site_name)]["repair_count"] += 1
                    ## 统计道路上的truck
                    for truck in self.trucks:
                        if truck.current_location == dump_site_name and truck.target_location == load_site_name:
                            road_status[(dump_site_name,load_site_name)]["truck_count"] += 1

            self.road.road_status = road_status

            # 等待下一个监控时间点
            yield env.timeout(monitor_interval)

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
        self.mine_logger.info("simulation started")
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
        self.mine_logger.info("simulation finished")
        self.summary()
        ticks = self.dump_frames(total_time=total_time)
        return ticks

    def start_rl(self, total_time:float=60*8)->dict:
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
            self.env.process(truck.run(is_rl_training=True))


    def dump_frames(self,total_time):
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
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        ticks = self.tick_generator.write_to_file(file_name=f'MINE:{self.name}_ALGO:{self.dispatcher.name}_TIME:{time_str}.json')
        return ticks