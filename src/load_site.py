import simpy

class ParkingLot:
    def __init__(self):
        pass

    def get_waiting_trucks(self, resource_list:list[simpy.Resource])->int:
        waiting_trucks = []
        for resource in resource_list:
            waiting_trucks += resource.queue
        return len(waiting_trucks)

class Shovel:
    def __init__(self, name:str, shovel_tons:float, shovel_cycle_time:float):
        self.name = name
        self.shovel_tons = shovel_tons
        self.shovel_cycle_time = shovel_cycle_time  # shovel-vehicle time took for one shovel of mine

    def set_env(self, env:simpy.Environment):
        self.env = env
        self.res = simpy.Resource(env, capacity=1)


class LoadSite:
    def __init__(self, name:str):
        self.name = name
        self.shovel_list = []
        self.parking_lot = ParkingLot()

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

        if "点3" in shovel.name:
            a = self.name
            b = self.shovel_list
            c = []
            for s in b:
                c.append(s.name)
            print("debug")
        self.shovel_list.append(shovel)


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