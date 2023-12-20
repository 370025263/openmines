"""模拟的核心对象Mine

使用的时候需要先初始化Mine，
然后添加
LoadSite(Shovel)、DumpSite(Dumper)、
ChargingSite(Truck)，Road，Dispatcher等对象
最后调用Mine.run()方法开始仿真
"""
import simpy,logging,math
from functools import reduce

from sisymines.src.charging_site import ChargingSite
from sisymines.src.dispatcher import BaseDispatcher
from sisymines.src.dump_site import DumpSite
from sisymines.src.load_site import LoadSite
from sisymines.src.road import Road
from sisymines.src.truck import Truck
from sisymines.src.utils.logger import MineLogger, LOG_FILE_PATH

class Mine:
    def __init__(self, name:str, log_path=LOG_FILE_PATH, log_file_level=logging.DEBUG, log_console_level=logging.INFO):
        self.env = simpy.Environment()
        self.name = name
        self.load_sites = []
        self.dump_sites = []
        self.trucks = []
        self.road = None
        self.dispatcher = None
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
        # log in the truck as process
        for truck in self.trucks:
            self.env.process(truck.run())
        self.env.run(until=total_time)
        self.mine_logger.info("simulation finished")
        self.summary()