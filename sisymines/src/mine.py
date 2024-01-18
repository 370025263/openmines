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

from sisymines.src.charging_site import ChargingSite
from sisymines.src.dispatcher import BaseDispatcher
from sisymines.src.dump_site import DumpSite
from sisymines.src.load_site import LoadSite
from sisymines.src.road import Road
from sisymines.src.truck import Truck
from sisymines.src.utils.logger import MineLogger, LOG_FILE_PATH
from sisymines.src.utils.ticker import TickGenerator
from sisymines.src.utils.event import EventPool

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
        # LOGGER配置
        self.global_logger = MineLogger(log_path=log_path, file_level=log_file_level, console_level=log_console_level)
        self.mine_logger = self.global_logger.get_logger(name)

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
        # 统计卸载区产量
        out_tons = []
        for dump_site in self.dump_sites:
            out_tons.append(dump_site.get_produce_tons())
        total_out_tons = sum(out_tons)
        print(f'{self.name} summary: {out_tons} tons')

        # 统计TruckCycleTime
        for truck in self.trucks:
            truck_cycle_time = truck.get_cycle_time()
            print(f'{truck.name} cycle time: {truck_cycle_time}')

        # 统计MatchingFactor
        shovels = [shovel for load_site in self.load_sites for shovel in load_site.shovel_list]
        num_trucks = len(self.trucks)
        load_times = [shovel.shovel_cycle_time for shovel in shovels]
        unique_loading_times = set([shovel.shovel_cycle_time for shovel in shovels])
        lcm_load_time = reduce(math.lcm, unique_loading_times)
        LSR = sum([lcm_load_time / shovel.shovel_cycle_time for shovel in shovels]) / lcm_load_time
        TAR = num_trucks**2 / sum(load_times)
        match_factor = TAR / LSR
        print(f'MatchingFactor: {match_factor}')

        # 统计TruckWaitTime
        for truck in self.trucks:
            truck_wait_time = truck.get_wait_time()
            print(f'{truck.name} wait time: {truck_wait_time}')
        # 统计TotalWaitTime
        total_wait_time = sum([truck.get_wait_time() for truck in self.trucks])
        print(f'TotalWaitTime: {total_wait_time}')

    def start(self, total_time:float=60*8):
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
            self.env.process(load_site.parking_lot.monitor_resources(env=self.env,
                                                                    resources=[shovel.res for shovel in load_site.shovel_list]))
            for shovel in load_site.shovel_list:
                self.env.process(load_site.parking_lot.monitor_resource(env=self.env,res_name=shovel.name,
                                                                        resource=shovel.res))
        for dump_site in self.dump_sites:
            self.env.process(dump_site.parking_lot.monitor_resources(env=self.env,
                                                                    resources=[dumper.res for dumper in dump_site.dumper_list]))
            for dumper in dump_site.dumper_list:
                self.env.process(dump_site.parking_lot.monitor_resource(env=self.env,res_name=dumper.name,
                                                                        resource=dumper.res))
        # log in the truck as process
        for truck in self.trucks:
            self.env.process(truck.run())
        self.env.run(until=total_time)
        self.mine_logger.info("simulation finished")
        self.summary()
        self.dump_frames(total_time=total_time)

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
        self.tick_generator.write_to_file(file_name=f'MINE:{self.name}_ALGO:{self.dispatcher.name}_TIME:{time_str}.json')
