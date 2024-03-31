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

    def monitor_resources(self, env, resources, res_objs, monitor_interval=1):
        """监控停车场对应所有资源的排队长度
            可以做到铲子、dumper单体级别的排队长度
        """
        self.queue_status["total"] = dict()
        while True:
            # 记录当前的排队长度
            all_queue_len = sum([len(resource.queue) for resource in resources])
            self.queue_status["total"][int(env.now)] = all_queue_len
            # 统计预期等待时间
            # 获取自定义请求包含的信息
            for index,res_obj in enumerate(res_objs):  # 遍历铲车或dumper对象
                res_name = res_obj.name
                load_time_list = []  # 当前装载点所有铲车的队列装载时间
                dump_time_list = []  # 当前卸载点所有dumper的队列卸载时间
                # 如果是铲车，那就对铲车队列等待时间进行统计
                if res_obj.type == "shovel":
                    shovel_name = res_obj.name
                    shovel_tons = res_obj.shovel_tons
                    shovel_cycle_time = res_obj.shovel_cycle_time
                    truck_load_time = 0
                    for i,load_request in enumerate(resources[index].queue):  # 遍历某铲车的装载请求
                        truck = load_request.truck
                        load_site = load_request.load_site
                        truck_load_time += (truck.truck_capacity / res_obj.shovel_tons) * res_obj.shovel_cycle_time
                    load_time_list.append(truck_load_time)  # 一个铲车的所有装载时间之和
                # 如果是dumper，那就对dumper队列等待时间进行统计
                if res_obj.type == "dumper":
                    dumper_name = res_obj.name
                    dumper_cycle_time = res_obj.dump_time
                    # 计算队列等待时间
                    dumper_queue_estimated_time = len(resources[index].queue)* dumper_cycle_time
                    dump_time_list.append(dumper_queue_estimated_time)
            if res_objs[-1].type == "shovel":
                EWA_LOAD_TIME = sum(load_time_list) / len(load_time_list)  # 装载区的平均等待时间
                load_site = res_objs[-1].load_site
                load_site.estimated_queue_wait_time = EWA_LOAD_TIME
            if res_objs[-1].type == "dumper":
                EWA_DUMP_TIME = sum(dump_time_list) / len(dump_time_list)  # dumpsite的平均卸载时间
                dump_site = res_objs[-1].dump_site
                dump_site.estimated_queue_wait_time = EWA_DUMP_TIME
            # 等待下一个监控时间点
            yield env.timeout(monitor_interval)

    def monitor_resource(self, env, res_obj, resource, monitor_interval=1):
        """监控铲子、dumper单体级别的排队长度信息
            由于使用了自定义的LoadRequest/DumpRequest,
            所以request中会包含load_site/dumpsite,truck等信息

            res_obj: shovel or dumper
            resource: shovel or dumper resource
        """
        res_name = res_obj.name
        self.queue_status[res_name] = dict()
        while True:
            # 记录当前的排队长度
            self.queue_status[res_name][int(env.now)] = len(resource.queue)
            # 等待下一个监控时间点
            yield env.timeout(monitor_interval)

class Shovel:
    def __init__(self, name:str, shovel_tons:float, shovel_cycle_time:float,position_offset:tuple=(0, 0.05)):
        self.name = name
        self.type = "shovel"
        self.load_site = None
        self.position = None
        self.position_offset = position_offset
        self.shovel_tons = shovel_tons  # the size of shovel bucket
        self.produced_tons = 0  # the produced tons of shovel
        self.service_count = 0  # the number of shovel-vehicle cycle
        self.shovel_cycle_time = shovel_cycle_time  # shovel-vehicle time took for one shovel of mine
        self.status = dict()  # the status of shovel

    def set_env(self, env:simpy.Environment):
        self.env = env
        self.res = simpy.Resource(env, capacity=1)

    def monitor_status(self, env, monitor_interval=1):
        """监控铲车的产量、服务次数等信息
        """
        while True:
            # 获取铲车信息
            self.status[int(env.now)] = {
                "produced_tons": self.produced_tons,
                "service_count": self.service_count,
            }
            # 等待下一个监控时间点
            yield env.timeout(monitor_interval)


class LoadSite:
    def __init__(self, name:str, position:tuple):
        self.name = name
        self.position = position
        self.shovel_list = []
        self.parking_lot = None
        self.status = dict()  # the status of shovel
        self.produced_tons = 0  # the produced tons of shovel
        self.service_count = 0  # the number of shovel-vehicle cycle
        self.estimated_queue_wait_time = 0  # the estimation of total waiting time for coming trucks in queue
        self.load_site_productivity = 0  # the productivity of load site

    def set_env(self, env:simpy.Environment):
        self.env = env
        for shovel in self.shovel_list:
            shovel.set_env(env)

    def monitor_status(self, env, monitor_interval=1):
        """监控装载区的产量、服务次数等信息
        """
        while True:
            # 获取铲车信息
            for shovel in self.shovel_list:
                self.produced_tons += shovel.produced_tons
                self.service_count += shovel.service_count
            self.status[int(env.now)] = {
                "produced_tons": self.produced_tons,
                "service_count": self.service_count,
            }
            # 统计装载区的装载能力
            load_site_productivity = sum(
                shovel.shovel_tons / shovel.shovel_cycle_time for shovel in self.shovel_list)
            self.load_site_productivity = load_site_productivity
            # reset
            self.produced_tons = 0
            self.service_count = 0
            # 等待下一个监控时间点
            yield env.timeout(monitor_interval)


    def show_shovels(self):
        shovel_names = []
        for shovel in self.shovel_list:
            shovel_names.append(shovel.name)
        print(f'{self.name} has {shovel_names}')

    def add_shovel(self, shovel:Shovel):
        # 铲子的位置是相对于卸载点的位置，和dumper的数目*偏移不同
        shovel.position = tuple(a + b for a, b in zip(self.position, shovel.position_offset))
        # 在shovel中添加自己的引用
        shovel.load_site = self
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