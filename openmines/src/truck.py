import random

import numpy as np
import simpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from simpy.resources.resource import Request



from openmines.src.charging_site import ChargingSite
from openmines.src.dump_site import DumpSite, Dumper
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.utils.event import Event, EventPool

TRUCK_DEFAULT_SPEED = 25 # km/h
"""在仿真结束后将仿真过程中矿卡的位置数据按帧转换成DataFrame并保存到文件中
        卡车地理标号(start,dest)：
            -1 代表充电区
            0-N 代表装载区
            N+1 N+M 代表卸载区
        卡车状态(state)：
            0 代表空载
            1 代表等载
            2 代表装载
            3 代表满载
            4 代表等卸
            5 代表卸载
        卡车位置数值(position_ratio)：
            0-1 代表从当前点到目标点的比率（仅在0 3过程有用）
        卡车位置坐标(position_x, position_y)：
            装载点 卸载点 铲车等等的位置 我们认为其已经在配置文件中提供了
            这里是计算好的二维坐标位置
            TODO：另外我们可视化的时候需要考虑到多个车辆重叠的情况(引入仿真sumo？)
        卡车时间(time)：
            仿真时间
        """

class LoadRequest(Request):
    """
    自定义的simpy资源请求类，用于在卡车到达装载区时请求铲车资源；
    """
    def __init__(self, resource, truck, load_site):
        super().__init__(resource)
        self.truck = truck
        self.load_site = load_site

class DumpRequest(Request):
    def __init__(self, resource, truck, dump_site):
        super().__init__(resource)
        self.truck = truck
        self.dump_site = dump_site

class Truck:
    def __init__(self, name:str, truck_capacity:float, truck_speed:float=TRUCK_DEFAULT_SPEED):
        self.name = name
        self.truck_capacity = truck_capacity  # truck capacity in tons
        self.truck_speed = truck_speed  # truck speed in km/h
        self.current_location = None
        self.target_location = None
        self.journey_start_time = 0
        self.journey_coverage = None
        self.last_breakdown_time = 0
        self.event_pool = EventPool()
        # the probalistics of the truck
        self.expected_working_time_without_breakdown = 60*6  # in minutes
        self.expected_working_time_until_unrepairable = 60*24*7*7  # in minutes
        self.repair_avg_time = 10
        self.repair_std_time = 3
        # truck status
        self.status = "idle"
        self.truck_load = 0  # in tons, the current load of the truck
        self.service_count = 0  # the number of times the truck has been dumped
        self.total_load_count = 0  # the total load count of the truck
        self.truck_cycle_time = 0
        self.first_order_time = 0
        # RL
        self.current_decision_event = None

    def set_env(self, mine:"Mine"):
        self.mine = mine
        self.env = mine.env
        self.dispatcher = mine.dispatcher

    def move(self, target_location, distance:float, manual_speed:float=None):
        """
        移动函数 需要区分满载和空载
        :param target_location:
        :param distance:
        :param manual_speed:
        :return:
        """
        assert target_location is not None, "target_location can not be None"
        assert target_location is not self.current_location, "target_location can not be the same as current_location"
        assert distance >= 0, "distance can not be negative"
        assert self.truck_speed > 0, "truck_speed can not be negative"
        # 记录当前路程的信息
        self.target_location = target_location
        self.journey_start_time = self.env.now
        if manual_speed is not None:
            assert manual_speed > 0, "manual_speed can not be negative"
            duration = (distance / manual_speed)*60  # in minutes
        else:
            duration = (distance / self.truck_speed)*60
        self.truck_speed = manual_speed if manual_speed is not None else self.truck_speed

        """
        1.车辆维修随机事件的模拟        
        """
        # 检查车辆是否发生故障并获得维修时间
        repair_time = self.check_vehicle_availability()
        if repair_time:
            # 如果发生故障，记录故障事件并进行维修
            breakdown_event = Event(self.last_breakdown_time, "TruckEvent:breakdown", f'Time:<{self.last_breakdown_time}> Truck:[{self.name}] breakdown for {repair_time} minutes',
                      info={"name": self.name, "status": "breakdown",
                            "start_time": self.env.now, "end_time": self.env.now + repair_time})
            self.event_pool.add_event(breakdown_event)
            self.mine.random_event_pool.add_event(breakdown_event)
            self.logger.info(f'Time:<{self.last_breakdown_time}> Truck:[{self.name}] breakdown for {repair_time} minutes at {self.last_breakdown_time}')
            # 进行维修（暂停运行）
            self.status = "repairing"
            yield self.env.timeout(repair_time)
        """
        2.车辆运行过程中的堵车随机事件模拟
        """
        # 分析当前道路中的车辆情况，并随机生成一个延迟时间用来模拟交通堵塞
        # 获取其他车辆的引用
        other_trucks = self.mine.trucks
        # 筛选出出发地和目的地跟本车相同的车辆
        other_trucks_on_road = [truck for truck in other_trucks if truck.current_location == self.current_location and
                                truck.target_location == self.target_location]
        # 为每个同路的车辆根据当前时间计算route_coverage数值
        for truck in other_trucks_on_road:
            truck.journey_coverage = truck.get_route_coverage(distance)
        # 假设每个车辆导致堵车的概率在route_coverage上满足正态分布，均值为route_coverage，方差为0.1
        # other_trucks_on_road中的每个车导致堵车的概率会叠加在一起取平均成为最终的堵车概率分布
        truck_positions = [truck.journey_coverage for truck in other_trucks_on_road]
        # 每个卡车的正态分布的标准差
        sd = 0.1  # 根据3sigma原则，这个值代表这个车辆对堵车的影响会在+-0.3总路程的范围内
        # 位置的范围
        x = np.linspace(0, 1, 1000)
        # 初始化堵车的概率为0
        total_prob = np.zeros_like(x)
        # 对每辆卡车
        for i, position in enumerate(truck_positions):
            # 计算在每个位置的堵车概率
            prob = norm.pdf(x, position, sd)
            # 将这个卡车的堵车概率加到总的堵车概率上
            total_prob = total_prob + prob
        # 正规化总的堵车概率
        total_prob = total_prob / len(truck_positions)
        # 使用蒙特卡洛方法来决定是否会发生堵车
        try:
            if np.sum(total_prob) == 0:
                probabilities = np.ones_like(total_prob) / len(total_prob)
            else:
                probabilities = total_prob / (np.sum(total_prob))
            jam_position = np.random.choice(x, p=probabilities)
        except:
            print("bug")
        # 在jam_position处使用weibull分布，采样一个可能的堵车时间
        jam_time = np.random.weibull(2) * 10
        time_to_jam = jam_position * duration
        if time_to_jam < jam_time:
            # 如果到达堵车区域的时间小于堵车时间，那么就会发生堵车
            is_jam = True if len(truck_positions) > 3 else False
            self.mine.random_event_pool.add_event(Event(self.env.now + time_to_jam, "RoadEvent:jam",
                                                        f'Time:<{self.env.now + time_to_jam}> Truck:[{self.name}] is jammed at road from {self.current_location.name} '
                                                        f'to {self.target_location.name} for {jam_time:.2f} minutes',
                                                        info={"name": self.name, "status": "jam", "speed": 0,
                                                              "start_location": self.current_location.name,
                                                                "end_location": self.target_location.name,
                                                              "start_time": self.env.now, "est_end_time": self.env.now + time_to_jam + jam_time}))
            self.logger.info(f"Time:<{self.last_breakdown_time}> Truck:[{self.name}] is jammed at {self.current_location.name} to {self.target_location.name} for {jam_time:.2f} minutes")
        else:
            is_jam = False


        """
        3.车辆完全损坏的模拟(当车辆完全损坏就会被移除,车辆到指定时间后不会再发生移动，而是停留在原地
        """
        if repair_time is None:
            self.logger.info(f"Time:<{self.last_breakdown_time}> Truck:[{self.name}] is broken down and beyond repair at {self.current_location.name} to "
                             f"{self.target_location.name}")
            unrepairable_event = Event(self.env.now, "TruckEvent:unrepairable", f'Time:<{self.env.now}> Truck:[{self.name}] is broken down and beyond repair'
                                                                     f' at {self.current_location.name} to '
                                                                        f'{self.target_location.name}',
                                        info={"name": self.name, "status": "unrepairable","time": self.env.now})
            self.event_pool.add_event(unrepairable_event)
            self.mine.random_event_pool.add_event(unrepairable_event)
            self.status = "unrepairable"
            return
        """
        4.车辆移动前往目的地的模拟
        """
        # 判断目标地点的类型
        if isinstance(self.current_location, DumpSite) and isinstance(target_location, LoadSite):
            event_name = "unhaul"
        elif isinstance(self.current_location, ChargingSite):
            event_name = "init"
        else:
            event_name = "haul"

        self.event_pool.add_event(Event(self.env.now, event_name, f'Truck:[{self.name}] moves at {target_location.name}',
                                        info={"name": self.name, "status": event_name, "speed": manual_speed if manual_speed is not None else self.truck_speed,
                                              "start_time": self.env.now, "est_end_time": self.env.now+duration, "end_time": None,
                                              "start_location": self.current_location.name,
                                              "target_location": target_location.name,
                                              "distance": distance, "duration": None}))
        self.status = "moving"
        yield self.env.timeout(duration+is_jam*jam_time)
        # 补全数据
        last_move_event = self.event_pool.get_last_event(type=event_name, strict=True)
        last_move_event.info["end_time"] = self.env.now
        last_move_event.info["duration"] = self.env.now - last_move_event.info["start_time"]
        # after arrival set current location
        assert type(self.current_location) != type(target_location), f"current_site and target_site should not be the same type of site "
        self.current_location = target_location

    def load(self, shovel:Shovel):
        shovel_tons = shovel.shovel_tons
        shovel_cycle_time = shovel.shovel_cycle_time
        load_time = (self.truck_capacity/shovel_tons) * shovel_cycle_time
        self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Start loading at shovel {shovel.name}, load time is {load_time}')
        self.status = "loading"
        yield self.env.timeout(load_time)
        # 随机生成一个装载量 +-10%
        self.truck_load = self.truck_capacity*(1+np.random.uniform(-0.1, 0.1))
        shovel.produced_tons += self.truck_load
        shovel.service_count += 1
        self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Finish loading at shovel {shovel.name}')

    def unload(self, dumper:Dumper):
        unload_time:float = dumper.dump_time
        self.status = "unloading"
        yield self.env.timeout(unload_time)
        dumper.dumper_tons += self.truck_load
        self.total_load_count += self.truck_load
        self.service_count += 1
        self.truck_cycle_time = (self.env.now - self.last_breakdown_time) / self.service_count
        self.truck_load = 0
        dumper.service_count += 1
        self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Finish unloading at dumper {dumper.name}, dumper tons is {dumper.dumper_tons}')
        # self.event_pool.add_event(Event(self.env.now, "unload", f'Truck:[{self.name}] Finish unloading at dumper {dumper.name}',
        #                           info={"name": self.name, "status": "unloading",
        #                                 "start_time": self.env.now-unload_time, "end_time": self.env.now,
        #                                 "dumper": dumper.name, "unload_duration": unload_time}))
        # WARN：这里unload和get dumper重复了

    def check_queue_position(self, shovel, request):
        """
        检查并返回矿车在等待队列中的位置
        :param shovel: 要检查队列的铲车
        :param request: 矿车的资源请求对象
        :return: 队列中的位置（从0开始计数）
        """
        try:
            if len(shovel.res.queue) == 0:
                return 0
            elif shovel.res.queue.index(request) == 0:
                return 0
            else:
                return shovel.res.queue.index(request)
        except ValueError:
            return 0  # 如果请求不在队列中，则返回-1

    def wait_for_decision(self):
        # 创建一个等待事件
        self.current_decision_event = self.env.event()
        yield self.current_decision_event  # 等待rl环境决策
        return self.current_decision_event.value  # 返回决策结果

    def run(self, is_rl_training=False):
        """
        矿车运行的主干入口函数
        车辆从充电区开始，然后请求init_order前往装载区
        到达装载区后，装载区分配一个shovel给车辆，然后车辆开始装载
        装载完成后，车辆请求haul_order前往卸载区
        到达卸载区后，卸载区分配一个dumper给车辆，然后车辆开始卸载
        卸载完成后，车辆请求back_order前往装载区
        TODO：1.到达时间的正态建模 2.电车油车的区分，油耗电耗统计 3.其他随机事件 4.统计图表
        :return:
        """
        # 配置日志
        self.logger = self.mine.global_logger.get_logger("Truck")

        # 轮班开始 车辆从充电区域前往装载区
        self.current_location = self.mine.charging_site
        self.status = "waiting for init order"
        if is_rl_training:
            # 等待RL agent给出决策;
            dest_load_index: int = yield self.env.process(self.wait_for_decision())
        else:
            dest_load_index: int = self.dispatcher.give_init_order(truck=self, mine=self.mine)  # TODO:允许速度规划
        self.first_order_time = self.env.now
        self.target_location = self.mine.load_sites[dest_load_index]

        move_distance:float = self.mine.road.charging_to_load[dest_load_index]
        load_site: LoadSite = self.mine.load_sites[dest_load_index]
        self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Activated at {self.env.now}, Target load site is ORDER({dest_load_index}):{load_site.name}, move distance is {move_distance}')
        self.event_pool.add_event(Event(self.env.now, "INIT ORDER", f'Truck:[{self.name}] Activated at {self.env.now}, Target load site is ORDER({dest_load_index}):{load_site.name}, move distance is {move_distance}',
                                        info={"name": self.name, "status": "INIT ORDER",
                                              "start_time": self.env.now, "est_end_time": self.env.now+move_distance/self.truck_speed,
                                              "start_location": self.current_location.name,
                                              "target_location": load_site.name,
                                              "distance": move_distance, "order_index": dest_load_index}))
        yield self.env.process(self.move(target_location=load_site, distance=move_distance))  # 移动时间
        # 检查车辆状态，如果是完全损坏则退出
        if self.status == "unrepairable":
            self.logger.info(f"Truck {self.name} is beyond repair and will no longer participate in operations.")
            return

        while True:
            # 到达装载区开始请求资源并装载
            self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Arrived at {self.mine.load_sites[dest_load_index].name} at {self.env.now}')
            load_site:LoadSite = self.mine.load_sites[dest_load_index]
            shovel = load_site.get_available_shovel()

            with LoadRequest(shovel.res, self, load_site) as req:
                # 申请到资源之前的操作
                truck_queue_index = self.check_queue_position(shovel, req)
                self.event_pool.add_event(Event(self.env.now, "wait shovel", f'Truck:[{self.name}] Wait shovel {shovel.name}',
                                                info={"name": self.name, "status": "waiting for shovel",
                                                      "queue_index": truck_queue_index,
                                                      "start_time": self.env.now, "end_time": None,
                                                      "shovel": shovel.name, "wait_duration": None}))
                self.status = "waiting for shovel"
                # ...
                yield req  # 申请铲车资源
                # 申请到铲车资源
                # 先获取wait的event进行数据补全
                last_wait_event = self.event_pool.get_last_event(type="wait shovel", strict=True)
                last_wait_event.info["end_time"] = self.env.now
                last_wait_event.info["wait_duration"] = self.env.now - last_wait_event.info["start_time"]
                # 添加load的event
                self.event_pool.add_event(Event(self.env.now, "get shovel", f'Truck:[{self.name}] Get shovel {shovel.name}',
                                                info={"name": self.name, "status": "loading on shovel",
                                                      "start_time": self.env.now, "end_time": None,
                                                      "shovel": shovel.name, "load_duration": None}))
                yield self.env.process(self.load(shovel))  # 装载时间 同shovel和truck自身有关系
                # 装载完毕 对之前的Event进行数据补全
                last_load_event = self.event_pool.get_last_event(type="get shovel", strict=True)
                last_load_event.info["end_time"] = self.env.now
                last_load_event.info["load_duration"] = self.env.now - last_load_event.info["start_time"]

            # 装载完毕，请求新的卸载区，并开始移动到卸载区
            self.status = "waiting for haul order"
            if is_rl_training:
                # 等待RL agent给出决策;
                dest_unload_index: int = yield self.env.process(self.wait_for_decision())
            else:
                dest_unload_index: int = self.dispatcher.give_haul_order(truck=self, mine=self.mine)
            dest_unload_site: DumpSite = self.mine.dump_sites[dest_unload_index]
            move_distance: float = self.mine.road.get_distance(truck=self, target_site=dest_unload_site)
            self.target_location = dest_unload_site

            self.logger.debug(f"Time:<{self.env.now}> Truck:[{self.name}] Start moving to ORDER({dest_unload_index}): {dest_unload_site.name}, move distance is {move_distance}, speed: {self.truck_speed}")
            self.event_pool.add_event(Event(self.env.now, "ORDER", f'Truck:[{self.name}] Start moving to ORDER({dest_unload_index}): {dest_unload_site.name}, move distance is {move_distance}, speed: {self.truck_speed}',
                                            info={"name": self.name, "status": "ORDER",
                                                  "start_time": self.env.now, "est_end_time": self.env.now+move_distance/self.truck_speed,
                                                  "start_location": self.current_location.name,
                                                  "target_location": dest_unload_site.name, "speed": self.truck_speed,
                                                  "distance": move_distance, "order_index": dest_unload_index}))

            yield self.env.process(self.move(target_location=dest_unload_site, distance=move_distance))  # 移动时间

            # 检查车辆状态，如果是完全损坏则退出
            if self.status == "unrepairable":
                self.logger.info(f"Truck {self.name} is beyond repair and will no longer participate in operations.")
                return
            # 到达卸载区并开始请求资源并卸载
            self.logger.debug(f'Time:<{self.env.now}> Truck:[{self.name}] Arrived at {dest_unload_site.name} at {self.env.now}')
            dumper:Dumper = dest_unload_site.get_available_dumper()
            with DumpRequest(dumper.res, self, dest_unload_site) as req:
                # 申请到资源之前的操作
                # ...
                self.event_pool.add_event(Event(self.env.now, "wait dumper", f'Truck:[{self.name}] Wait dumper {dumper.name}',
                                                info={"name": self.name, "status": "waiting for dumper",
                                                      "start_time": self.env.now, "end_time": None,
                                                      "dumper": dumper.name, "wait_duration": None}))
                self.status = "waiting for dumper"
                yield req  # 申请卸载位资源
                # 申请到卸载位资源
                # 先获取wait的event进行数据补全
                last_wait_event = self.event_pool.get_last_event(type="wait dumper", strict=True)
                last_wait_event.info["end_time"] = self.env.now
                last_wait_event.info["wait_duration"] = self.env.now - last_wait_event.info["start_time"]
                self.event_pool.add_event(Event(self.env.now, "get dumper", f'Truck:[{self.name}] Get dumper {dumper.name}',
                                                info={"name": self.name, "status": "unloading on dumper",
                                                      "start_time": self.env.now, "end_time": None,
                                                      "dumper": dumper.name, "unload_duration": None}))
                yield self.env.process(self.unload(dumper))
                # 卸载完毕 对之前的Event进行数据补全
                last_unload_event = self.event_pool.get_last_event(type="get dumper", strict=True)
                last_unload_event.info["end_time"] = self.env.now
                last_unload_event.info["unload_duration"] = self.env.now - last_unload_event.info["start_time"]

            # 卸载完毕，请求新的装载区，并开始移动到装载区
            self.status = "waiting for back order"
            if is_rl_training:
                # 等待RL agent给出决策;
                dest_load_index: int = yield self.env.process(self.wait_for_decision())
            else:
                dest_load_index: int = self.dispatcher.give_back_order(truck=self, mine=self.mine)
            dest_load_site: LoadSite = self.mine.load_sites[dest_load_index]
            move_distance: float = self.mine.road.get_distance(truck=self, target_site=dest_load_site)
            self.target_location = dest_load_site
            self.logger.debug(f"Time:<{self.env.now}> Truck:[{self.name}] Start moving to ORDER({dest_load_index}):{dest_load_site.name}, move distance is {move_distance}, speed: {self.truck_speed}")
            self.event_pool.add_event(Event(self.env.now, "ORDER", f'Truck:[{self.name}] Start moving to ORDER({dest_load_index}):{dest_load_site.name}, move distance is {move_distance}, speed: {self.truck_speed}',
                                            info={"name": self.name, "status": "ORDER",
                                                  "start_time": self.env.now, "est_end_time": self.env.now+move_distance/self.truck_speed,
                                                  "start_location": self.current_location.name,
                                                  "target_location": dest_load_site.name, "speed": self.truck_speed,
                                                  "distance": move_distance, "order_index": dest_load_index}))
            yield self.env.process(self.move(target_location=dest_load_site, distance=move_distance))  # 移动时间
            # 检查车辆状态，如果是完全损坏则退出
            if self.status == "unrepairable":
                self.logger.info(f"Truck {self.name} is beyond repair and will no longer participate in operations.")
                return

    def charge(self, duration):
        """
        电矿卡需要充电
        油矿卡不需要充电
        TODO：针对卡车油耗、电耗、充电时间进行建模。并应用。
        :param duration:
        :return:
        """
        self.logger.info(f'{self.name} Start charging at {self.env.now}')
        yield self.env.timeout(duration)

    def get_wait_time(self):
        """
        获取一次循环的等待时间
        目前是计算unload事件的个数和时间，然后计算平均值
        :return:
        """
        # TODO:不能只有接到资源的车才计算wait时间 所有的车都应该计算wait时间 有的根本没有获得资源 15811296389
        wait_shovel_events = self.event_pool.get_even_by_type("wait shovel")
        end_wait_shovel_events = self.event_pool.get_even_by_type("get shovel")
        self.event_pool.add_event(Event(self.mine.total_time, "end", f'Truck:[{self.name}] End'))
        end_event = self.event_pool.get_even_by_type("end")

        wait_shovel_event_count = len(wait_shovel_events)
        end_wait_shovel_event_count = len(end_wait_shovel_events)

        if wait_shovel_event_count > end_wait_shovel_event_count:
            # 如果等待铲车的事件比获得铲车的事件多，说明结束时间打断了等待铲车的事件，为结束等待添加一个结束时间
            end_wait_shovel_events.append(end_event[0]) # end evnet 没有添加成功

        wait_shovel_event_time = sum([event.time_stamp for event in wait_shovel_events])
        end_wait_shovel_event_time = sum([event.time_stamp for event in end_wait_shovel_events])
        wait_time = (end_wait_shovel_event_time - wait_shovel_event_time) / wait_shovel_event_count
        if wait_time == float('inf'):
            wait_time = 0
        return wait_time

    def get_route_coverage(self,distance)->float:
        """
        获取一次循环的路程覆盖率 0-1
        0代表刚刚出发 1代表完成
        这里只是一种近似，没有考虑到当前路途的交通情况
        :return:
        """
        assert distance > 0, "distance must be greater than 0"
        if self.journey_start_time is None:
            print(1)
        assert self.journey_start_time is not None, "journey_start_time must be not None, is the truck journey started?"

        # 获取当前环境时间
        current_time = self.env.now
        # 获取卡车的速度
        speed = self.truck_speed
        # 获取卡车的总共预期行驶时间
        total_travel_time = (distance/speed)*60  # 单位：分钟
        # 获取卡车的路程覆盖率
        coverage = (current_time - self.journey_start_time)/total_travel_time
        return coverage

    def check_vehicle_availability(self):
        """
        使用指数分布对卡车的可用性进行建模并采样，可能出现的状况包括故障需要维修。
        :return: 维修持续时间（如果发生故障)
        """
        # 使用指数分布计算故障发生的时间（载具正常工作时长）
        time_to_breakdown = np.random.exponential(self.expected_working_time_without_breakdown)
        # 判断载具是否发生故障（当前时间超过故障时间）
        if self.env.now >= self.last_breakdown_time + time_to_breakdown:
            # 发生故障，使用正态分布N(mu=10, sigma=3)采样获得维修时间
            repair_time = np.random.normal(self.repair_avg_time, self.repair_std_time)
            repair_time = max(repair_time, 0)  # 确保维修时间为正值
            # 更新最后一次故障时间
            self.last_breakdown_time = self.last_breakdown_time + time_to_breakdown
            """
            TODO: 维修的判断依据有问题 很有可能多次采样叠加后self.last_breakdown_time依然是小于当前时间的
            这里明天再修复 今天弄不了 脑子不好用了
            """
            return repair_time  # 发生故障则返回维修时间
        # 如果车辆发生无法维修的损坏，返回None；使用指数分布计算故障发生的时间
        # self.expected_working_time_until_unrepairable

        if self.env.now >= random.expovariate(1.0 / self.expected_working_time_until_unrepairable):
            return None
        return 0  # 未发生故障则返回0维修时间
