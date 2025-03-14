import simpy

from openmines.src.load_site import ParkingLot


class Dumper:
    def __init__(self, name:str, dumper_cycle_time:float, position_offset:tuple=(0, 0.01)):
        self.name = name
        self.type = "dumper"
        self.dump_site = None
        self.position = None
        self.position_offset = position_offset
        self.dumper_tons:float = 0.0  #  the tons of material the dumper has received
        self.service_count = 0
        self.dump_time = dumper_cycle_time  # shovel-vehicle time took for one shovel of mine
        self.status = dict()  # the status of shovel
        self.last_service_time = 0  # the last service time of shovel, the moment that the shovel start loading a truck
        self.last_service_done_time = 0  # the last service done time of shovel, the moment that the shovel finish loading a truck
        self.est_waiting_time = 0  # the estimated waiting time for the next truck

    def set_env(self, env:simpy.Environment):
        self.env = env
        self.res = simpy.Resource(env, capacity=1)

    def monitor_status(self, env, monitor_interval=1):
        """监控卸载位的产量、服务次数等信息
        """
        while True:
            self.status[int(env.now)] = {
                "produced_tons": self.dumper_tons,
                "service_count": self.service_count,
            }
            # 等待下一个监控时间点
            yield env.timeout(monitor_interval)

class DumpSite:
    def __init__(self, name:str, position:tuple):
        self.name = name
        self.position = position
        self.dumper_list = []
        self.parking_lot = None
        self.tons = 0  # 卸载点的总吨数 用于统计
        self.truck_visits = 0  # 卸载点的总车次数 用于统计
        self.status = dict()  # the status of shovel
        self.produce_tons = 0  # the produced tons of this dump site
        self.service_count = 0  # the number of shovel-vehicle cycle in this dump site
        self.service_ability_ratio = 1  # the ability of dump site to serve trucks(0-1), the dumper may be breakdown
        self.estimated_queue_wait_time = 0  # the estimation of total waiting time for coming trucks in queue
        self.avg_queue_wait_time = 0  # the average waiting time for coming trucks in queue
        self.dump_site_productivity = 0
        # service_time = min service time
        self.last_service_time = 0  # 上一次服务Start时间
        self.last_service_done_time = 0  # 上一次服务End时间

    def update_service_time(self):
        self.last_service_time = min([dumper.last_service_time for dumper in self.dumper_list])
        self.last_service_done_time = min([dumper.last_service_done_time for dumper in self.dumper_list])

    def set_env(self, env:simpy.Environment):
        self.env = env
        for dumper in self.dumper_list:
            dumper.set_env(env)

    def monitor_status(self, env, monitor_interval=1):
        """监控卸载区的产量、服务次数等信息
        """
        while True:
            # 获取每个dumper信息并统计
            self.produce_tons = sum(dumper.dumper_tons for dumper in self.dumper_list)
            self.service_count = sum(dumper.service_count for dumper in self.dumper_list)
            self.status[int(env.now)] = {
                "produced_tons": self.produce_tons,
                "service_count": self.service_count,
            }
            # 统计卸载区的卸载能力
            dump_site_productivity = sum(
                dumper.dumper_tons / dumper.dump_time for dumper in self.dumper_list)
            self.dump_site_productivity = dump_site_productivity
            # 等待下一个监控时间点
            yield env.timeout(monitor_interval)

    def add_dumper(self, dumper:Dumper):
        dumper_count = len(self.dumper_list)
        dumper.position = tuple(a + b for a, b in zip(self.position, dumper.position_offset*(dumper_count+1)))
        # 在dumper中添加自己引用
        dumper.dump_site = self
        self.dumper_list.append(dumper)

    def get_produce_tons(self):
        self.tons = sum([dumper.dumper_tons for dumper in self.dumper_list])
        return self.tons

    def show_dumpers(self):
        dumper_names = []
        for dumper in self.dumper_list:
            dumper_names.append(dumper.name)
        print(f'{self.name} has {dumper_names}')

    def add_parkinglot(self, position_offset, name:str=None):
        if name is None:
            name = f'{self.name}_parking_lot'
        park_position = tuple(a + b for a, b in zip(self.position, position_offset))
        self.parking_lot = ParkingLot(name=name, position=park_position)

    def get_available_dumper(self)->Dumper:
        """
        这里是一个简单的贪心算法，返回第一个空闲的卸载点
        TODO：下一个版本，也许可以将铲车暴露出来作为单个卸载区让其决策。目前仅贪心算法。
        :return:
        """
        for dumper in self.dumper_list:
            if dumper.res.count == 0:
                return dumper
        # 如果没有空闲的铲车 则返回最小队列的铲车
        min_queue_dumper = min(self.dumper_list, key=lambda dumper: len(dumper.res.queue))
        return min_queue_dumper