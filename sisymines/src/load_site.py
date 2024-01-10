import simpy

class ParkingLot:
    def __init__(self,name:str,position:tuple):
        self.name = name
        self.position = position
        self.queue_status = dict()


    def get_waiting_trucks(self, resource_list:list[simpy.Resource])->int:
        waiting_trucks = []
        for resource in resource_list:
            waiting_trucks += resource.queue
        return len(waiting_trucks)

    def monitor_resources(self, env, resources, monitor_interval=1):
        """监控停车场对应所有资源的排队长度
            可以做到铲子、dumper单体级别的排队长度
        """
        self.queue_status["total"] = dict()
        while True:
            # 记录当前的排队长度
            all_queue_len = sum([len(resource.queue) for resource in resources])
            self.queue_status["total"][int(env.now)] = all_queue_len
            # 等待下一个监控时间点
            yield env.timeout(monitor_interval)

    def monitor_resource(self, env, res_name, resource, monitor_interval=1):
        """监控停车场对应资源的排队长度
            可以做到铲子、dumper单体级别的排队长度
        """
        self.queue_status[res_name] = dict()
        while True:
            # 记录当前的排队长度
            self.queue_status[res_name][int(env.now)] = len(resource.queue)
            # 等待下一个监控时间点
            yield env.timeout(monitor_interval)

class Shovel:
    def __init__(self, name:str, shovel_tons:float, shovel_cycle_time:float,position_offset:tuple=(0, 0.05)):
        self.name = name
        self.position = None
        self.position_offset = position_offset
        self.shovel_tons = shovel_tons
        self.shovel_cycle_time = shovel_cycle_time  # shovel-vehicle time took for one shovel of mine

    def set_env(self, env:simpy.Environment):
        self.env = env
        self.res = simpy.Resource(env, capacity=1)


class LoadSite:
    def __init__(self, name:str, position:tuple):
        self.name = name
        self.position = position
        self.shovel_list = []
        self.parking_lot = None

    def set_env(self, env:simpy.Environment):
        self.env = env
        for shovel in self.shovel_list:
            shovel.set_env(env)

    def show_shovels(self):
        shovel_names = []
        for shovel in self.shovel_list:
            shovel_names.append(shovel.name)
        print(f'{self.name} has {shovel_names}')

    def add_shovel(self, shovel:Shovel):
        # 铲子的位置是相对于卸载点的位置，和dumper的数目*偏移不同
        shovel.position = tuple(a + b for a, b in zip(self.position, shovel.position_offset))
        self.shovel_list.append(shovel)

    def add_parkinglot(self, position_offset, name: str = None):
        if name is None:
            name = f'{self.name}_parking_lot'
        park_position = tuple(a + b for a, b in zip(self.position, position_offset))
        self.parking_lot = ParkingLot(name=name, position=park_position)

    def get_available_shovel(self)->Shovel:
        """
        这里是一个简单的贪心算法，返回第一个空闲的铲车
        TODO：下一个版本，也许可以将铲车暴露出来作为单个卸载区让其决策。目前仅贪心算法。
        :return:
        """
        for shovel in self.shovel_list:
            if shovel.res.count == 0:
                return shovel
        # 如果没有空闲的铲车 则返回最小队列的铲车
        min_queue_shovel = min(self.shovel_list, key=lambda shovel: len(shovel.res.queue))
        return min_queue_shovel