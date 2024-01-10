import simpy

from sisymines.src.load_site import ParkingLot


class Dumper:
    def __init__(self, name:str, dumper_cycle_time:float, position_offset:tuple=(0, 0.01)):
        self.name = name
        self.position = None
        self.position_offset = position_offset
        self.dumper_tons:float = 0.0
        self.dump_time = dumper_cycle_time  # shovel-vehicle time took for one shovel of mine

    def set_env(self, env:simpy.Environment):
        self.env = env
        self.res = simpy.Resource(env, capacity=1)


class DumpSite:
    def __init__(self, name:str, position:tuple):
        self.name = name
        self.position = position
        self.dumper_list = []
        self.parking_lot = None
        self.tons = 0  # 卸载点的总吨数 用于统计
        self.truck_visits = 0  # 卸载点的总车次数 用于统计

    def set_env(self, env:simpy.Environment):
        self.env = env
        for dumper in self.dumper_list:
            dumper.set_env(env)

    def add_dumper(self, dumper:Dumper):
        dumper_count = len(self.dumper_list)
        dumper.position = tuple(a + b for a, b in zip(self.position, dumper.position_offset*(dumper_count+1)))
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