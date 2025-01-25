import simpy

from openmines.src.utils.event import EventPool


class ParkingLot:
    def __init__(self,name:str,position:tuple):
        self.name = name
        self.position = position
        self.queue_status = dict()
        self.resources = []
        self.res_objs = []
        self.env = None

    def get_waiting_trucks(self, resource_list:list[simpy.Resource])->int:
        waiting_trucks = []
        for resource in resource_list:
            waiting_trucks += resource.queue
        return len(waiting_trucks)

    def update_queue_wait_status(self):
        # Get Reference
        res_objs = self.res_objs
        resources = self.resources
        env = self.env
        # 统计预期等待时间
        # 获取自定义请求包含的信息
        for index, res_obj in enumerate(res_objs):  # 遍历shovel或dumper对象: eg. shovels or dumpers
            res_name = res_obj.name
            # 如果是铲车，那就对铲车队列等待时间进行统计(shovel.est_waiting_time)
            if res_obj.type == "shovel":
                shovel_name = res_obj.name
                shovel_tons = res_obj.shovel_tons
                shovel_cycle_time = res_obj.shovel_cycle_time
                shovel_queue_wait_total_time = 0
                shovel_last_service_time = max(res_obj.last_service_time,res_obj.last_service_done_time)  # Shovel上一次服务时间
                for i, load_request in enumerate(resources[index].queue + resources[index].users):  # 遍历某铲车的装载请求
                    truck = load_request.truck
                    load_site = load_request.load_site
                    shovel_queue_wait_total_time += (truck.truck_capacity / res_obj.shovel_tons) * res_obj.shovel_cycle_time
                time_used_loading = env.now - shovel_last_service_time if resources[index].queue + resources[index].users else 0
                shovel_estimated_waiting_time = max(shovel_last_service_time + shovel_queue_wait_total_time - env.now, 0)  # 时间段，不是时间点
                res_obj.est_waiting_time = shovel_estimated_waiting_time  # 铲车的预期等待时间
                assert time_used_loading >= 0, "Time used for loading should be greater than 0"
            # 如果是dumper，那就对dumper队列等待时间进行统计(dumper.est_waiting_time)
            if res_obj.type == "dumper":
                dumper_name = res_obj.name
                dumper_cycle_time = res_obj.dump_time
                dumper_queue_wait_total_time = 0
                dumper_last_service_time = res_obj.last_service_time  # Dumper上一次服务时间
                for i, dump_request in enumerate(resources[index].queue):  # 遍历某dumper的卸载请求
                    truck = dump_request.truck
                    dump_site = dump_request.dump_site
                    dumper_queue_wait_total_time += res_obj.dump_time
                time_used_dumping = env.now - dumper_last_service_time
                dumper_estimated_waiting_time = (dumper_queue_wait_total_time - time_used_dumping) if dumper_queue_wait_total_time > time_used_dumping else 0
                res_obj.est_waiting_time = dumper_estimated_waiting_time  # Dumper的预期等待时间
                assert time_used_dumping >= 0, "Time used for dumping should be greater than 0"

        # UPDATE LOAD_SITE AND DUMP_SITE(from the summary by each shovel and dumper, take the mini)
        if res_objs and res_objs[-1].type == "shovel":
            load_site = res_objs[-1].load_site
            load_site.estimated_queue_wait_time = min([shovel.est_waiting_time for shovel in res_objs])  # 装载区的平均等待时间
            load_site.avg_queue_wait_time = sum([shovel.est_waiting_time for shovel in res_objs]) / len(res_objs)
        if res_objs and res_objs[-1].type == "dumper":
            dump_site = res_objs[-1].dump_site
            dump_site.estimated_queue_wait_time = min([dumper.est_waiting_time for dumper in res_objs])  # 卸载区的平均等待时间
            dump_site.avg_queue_wait_time = sum([dumper.est_waiting_time for dumper in res_objs]) / len(res_objs)

    def monitor_resources(self, env, resources, res_objs, monitor_interval=1):
        """监控停车场对应所有资源的排队长度
            可以做到铲子、dumper单体级别的排队长度
        """
        self.res_objs = res_objs
        self.resources = resources
        self.env = env

        self.queue_status["total"] = dict()
        while True:
            # 记录当前的排队长度
            all_queue_len = sum([len(resource.queue) for resource in resources])
            self.queue_status["total"][int(env.now)] = all_queue_len
            self.queue_status["total"]["cur_value"] = all_queue_len
            self.update_queue_wait_status()
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
            self.queue_status[res_name]["cur_value"] = len(resource.queue)

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
        self.last_service_time = 0  # the last service time of shovel, the moment that the shovel start loading a truck
        self.last_service_done_time = 0  # the last service done time of shovel, the moment that the shovel finish loading a truck
        self.est_waiting_time = 0  # the estimated waiting time for the next truck
        # shovel breakdown
        self.event_pool = EventPool()
        self.last_breakdown_time = 0  # 上一次故障时间
        self.repair = False  # 是否正在维修

    def set_env(self, env:simpy.Environment):
        self.env = env
        self.res = simpy.Resource(env, capacity=1)

    def monitor_status(self, env, monitor_interval=1):
        """监控铲车的产量、服务次数等信息
        """
        while True:
            # 获取铲车信息
            self.status[int(env.now)] = {
                "repair": self.repair,
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
        self.service_ability_ratio = 0  # the ability of load site to serve trucks(0-1), the shovel may be breakdown
        self.estimated_queue_wait_time = 0  # the estimation of total waiting time for coming trucks in queue
        self.avg_queue_wait_time = 0  # the average waiting time for coming trucks in queue
        self.load_site_productivity = 0  # the productivity of load site
        # service_time = min service time
        self.last_service_time = 0  # 上一次服务Start时间
        self.last_service_done_time = 0  # 上一次服务End时间

    def update_service_time(self):
        self.last_service_time = min([shovel.last_service_time for shovel in self.shovel_list])
        self.last_service_done_time = min([shovel.last_service_done_time for shovel in self.shovel_list])

    def set_env(self, env:simpy.Environment):
        self.env = env
        for shovel in self.shovel_list:
            shovel.set_env(env)

    def monitor_status(self, env, monitor_interval=1):
        """监控装载区的产量、服务次数等信息
        """
        while True:
            # 获取铲车信息
            self.produced_tons = sum(shovel.produced_tons for shovel in self.shovel_list)
            self.service_count = sum(shovel.service_count for shovel in self.shovel_list)
            self.status[int(env.now)] = {
                "produced_tons": self.produced_tons,
                "service_count": self.service_count,
            }
            # 统计装载区的理论装载能力
            load_site_productivity = sum(
                shovel.shovel_tons / shovel.shovel_cycle_time for shovel in self.shovel_list)
            self.load_site_productivity = load_site_productivity
            self.service_ability_ratio = sum((shovel.shovel_tons / shovel.shovel_cycle_time) * (0 if shovel.repair else 1) for shovel in self.shovel_list) / self.load_site_productivity
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