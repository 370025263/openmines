import simpy

from sisymines.src.load_site import ParkingLot


class Dumper:
    def __init__(self, name:str, dumper_cycle_time:float):
        self.name = name
        self.dumper_tons:float = 0.0
        self.dump_time = dumper_cycle_time  # shovel-vehicle time took for one shovel of mine

    def set_env(self, env:simpy.Environment):
        self.env = env
        self.res = simpy.Resource(env, capacity=1)


class DumpSite:
    def __init__(self, name:str):
        self.name = name
        self.dumper_list = []
        self.parking_lot = ParkingLot()  # 这个对象是用于统计卸载点的等待队列长度
        self.tons = 0  # 卸载点的总吨数 用于统计
        self.truck_visits = 0  # 卸载点的总车次数 用于统计

    def set_env(self, env:simpy.Environment):
        self.env = env
        for dumper in self.dumper_list:
            dumper.set_env(env)

    def add_dumper(self, dumper:Dumper):
        self.dumper_list.append(dumper)

    def get_produce_tons(self):
        self.tons = sum([dumper.dumper_tons for dumper in self.dumper_list])
        return self.tons

    def show_dumpers(self):
        dumper_names = []
        for dumper in self.dumper_list:
            dumper_names.append(dumper.name)
        print(f'{self.name} has {dumper_names}')

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