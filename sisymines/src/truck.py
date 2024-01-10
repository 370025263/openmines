import simpy

from sisymines.src.charging_site import ChargingSite
from sisymines.src.dump_site import DumpSite, Dumper
from sisymines.src.load_site import LoadSite
from sisymines.src.utils.event import Event, EventPool

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

class Truck:
    def __init__(self, name:str, truck_capacity:float, truck_speed:float=TRUCK_DEFAULT_SPEED):
        self.name = name
        self.truck_capacity = truck_capacity  # truck capacity in tons
        self.truck_speed = truck_speed  # truck speed in km/h
        self.current_location = None
        self.event_pool = EventPool()

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
        if manual_speed is not None:
            assert manual_speed > 0, "manual_speed can not be negative"
            duration = (distance / manual_speed)*60  # in minutes
        else:
            duration = (distance / self.truck_speed)*60
        self.truck_speed = manual_speed if manual_speed is not None else self.truck_speed
        

        # 判断目标地点的类型
        if isinstance(target_location, LoadSite) or isinstance(target_location, ChargingSite):
            # 如果是装载区
            event_name = "unhaul"
        else:
            # 如果是卸载区
            event_name = "haul"

        self.event_pool.add_event(Event(self.env.now, event_name, f'Truck:[{self.name}] moves at {target_location.name}',
                                        info={"name": self.name, "status": event_name, "speed": manual_speed if manual_speed is not None else self.truck_speed,
                                              "start_time": self.env.now, "est_end_time": self.env.now+duration, "end_time": None,
                                              "start_location": self.current_location.name,
                                              "target_location": target_location.name,
                                              "distance": distance, "duration": None}))
        # TODO: 实际上move的end_time是不确定的，因为move的过程中可能会有其他的event发生，比如堵车、滑坡等
        #       这里暂时先不考虑这种情况，后续可以考虑加入这种情况
        #       可以end_time先留空 然后对当前时间的主干取hash值然后后期遇到突发时间 那么就拿过来修改
        yield self.env.timeout(duration)
        # 补全数据
        last_move_event = self.event_pool.get_last_event(type=event_name, strict=True)
        last_move_event.info["end_time"] = self.env.now
        last_move_event.info["duration"] = self.env.now - last_move_event.info["start_time"]
        # after arrival set current location
        self.current_location = target_location

    def load(self, shovel):
        shovel_tons = shovel.shovel_tons
        shovel_cycle_time = shovel.shovel_cycle_time
        load_time = (self.truck_capacity/shovel_tons) * shovel_cycle_time
        self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Start loading at shovel {shovel.name}, load time is {load_time}')
        yield self.env.timeout(load_time)
        self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Finish loading at shovel {shovel.name}')
        # self.event_pool.add_event(Event(self.env.now, "load", f'Truck:[{self.name}] Finish loading at shovel {shovel.name}',
        #                           info={"name": self.name, "status": "loading",
        #                                 "start_time": self.env.now-load_time, "end_time": self.env.now,
        #                                 "shovel": shovel.name, "load_duration": load_time}))
        # WARN:这里load和get shovel重复了
    def unload(self, dumper:Dumper):
        unload_time:float = dumper.dump_time
        yield self.env.timeout(unload_time)
        dumper.dumper_tons += self.truck_capacity
        self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Finish unloading at dumper {dumper.name}, dumper tons is {dumper.dumper_tons}')
        # self.event_pool.add_event(Event(self.env.now, "unload", f'Truck:[{self.name}] Finish unloading at dumper {dumper.name}',
        #                           info={"name": self.name, "status": "unloading",
        #                                 "start_time": self.env.now-unload_time, "end_time": self.env.now,
        #                                 "dumper": dumper.name, "unload_duration": unload_time}))
        # WARN：这里unload和get dumper重复了

    def run(self):
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
        dest_load_index: int = self.dispatcher.give_init_order(truck=self, mine=self.mine)  # TODO:允许速度规划

        move_distance:float = self.mine.road.charging_to_load[dest_load_index]
        load_site: LoadSite = self.mine.load_sites[dest_load_index]
        self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Activated at {self.env.now}, Target load site is ORDER({dest_load_index}):{load_site.name}, move distance is {move_distance}')
        self.event_pool.add_event(Event(self.env.now, "ORDER", f'Truck:[{self.name}] Activated at {self.env.now}, Target load site is ORDER({dest_load_index}):{load_site.name}, move distance is {move_distance}',
                                    info={"name": self.name, "status": "ORDER",
                                            "start_time": self.env.now, "est_end_time": self.env.now+move_distance/self.truck_speed,
                                            "start_location": self.current_location.name,
                                            "target_location": load_site.name,
                                            "distance": move_distance, "order_index": dest_load_index}))
        yield self.env.process(self.move(target_location=load_site, distance=move_distance))  # 移动时间

        while True:
            # 到达装载区开始请求资源并装载
            self.logger.info(f'Time:<{self.env.now}> Truck:[{self.name}] Arrived at {self.mine.load_sites[dest_load_index].name} at {self.env.now}')
            load_site:LoadSite = self.mine.load_sites[dest_load_index]
            shovel = load_site.get_available_shovel()

            with shovel.res.request() as req:
                # 申请到资源之前的操作
                self.event_pool.add_event(Event(self.env.now, "wait shovel", f'Truck:[{self.name}] Wait shovel {shovel.name}',
                                            info={"name": self.name, "status": "waiting for shovel",
                                                    "start_time": self.env.now, "end_time": None,
                                                    "shovel": shovel.name, "wait_duration": None}))
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
            dest_unload_index: int = self.dispatcher.give_haul_order(truck=self, mine=self.mine)
            dest_unload_site: DumpSite = self.mine.dump_sites[dest_unload_index]
            move_distance: float = self.mine.road.get_distance(truck=self, target_site=dest_unload_site)
            self.logger.debug(f"Time:<{self.env.now}> Truck:[{self.name}] Start moving to ORDER({dest_unload_index}): {dest_unload_site.name}, move distance is {move_distance}, speed: {self.truck_speed}")
            self.event_pool.add_event(Event(self.env.now, "ORDER", f'Truck:[{self.name}] Start moving to ORDER({dest_unload_index}): {dest_unload_site.name}, move distance is {move_distance}, speed: {self.truck_speed}',
                                        info={"name": self.name, "status": "ORDER",
                                                "start_time": self.env.now, "est_end_time": self.env.now+move_distance/self.truck_speed,
                                                "start_location": self.current_location.name,
                                                "target_location": dest_unload_site.name, "speed": self.truck_speed,
                                                "distance": move_distance, "order_index": dest_unload_index}))

            yield self.env.process(self.move(target_location=dest_unload_site, distance=move_distance))  # 移动时间

            # 到达卸载区并开始请求资源并卸载
            self.logger.debug(f'Time:<{self.env.now}> Truck:[{self.name}] Arrived at {dest_unload_site.name} at {self.env.now}')
            dumper:Dumper = dest_unload_site.get_available_dumper()
            with dumper.res.request() as req:
                # 申请到资源之前的操作
                # ...
                self.event_pool.add_event(Event(self.env.now, "wait dumper", f'Truck:[{self.name}] Wait dumper {dumper.name}',
                                            info={"name": self.name, "status": "waiting for dumper",
                                                    "start_time": self.env.now, "end_time": None,
                                                    "dumper": dumper.name, "wait_duration": None}))
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
            dest_load_index: int = self.dispatcher.give_back_order(truck=self, mine=self.mine)
            dest_load_site: LoadSite = self.mine.load_sites[dest_load_index]
            move_distance: float = self.mine.road.get_distance(truck=self, target_site=dest_load_site)
            self.logger.debug(f"Time:<{self.env.now}> Truck:[{self.name}] Start moving to ORDER({dest_load_index}):{dest_load_site.name}, move distance is {move_distance}, speed: {self.truck_speed}")
            self.event_pool.add_event(Event(self.env.now, "ORDER", f'Truck:[{self.name}] Start moving to ORDER({dest_load_index}):{dest_load_site.name}, move distance is {move_distance}, speed: {self.truck_speed}',
                                        info={"name": self.name, "status": "ORDER",
                                                "start_time": self.env.now, "est_end_time": self.env.now+move_distance/self.truck_speed,
                                                "start_location": self.current_location.name,
                                                "target_location": dest_load_site.name, "speed": self.truck_speed,
                                                "distance": move_distance, "order_index": dest_load_index}))
            yield self.env.process(self.move(target_location=dest_load_site, distance=move_distance))  # 移动时间

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

    def get_cycle_time(self):
        """
        获取一次循环的时间
        目前是计算unload事件的个数和时间，然后计算平均值
        :return:
        """
        unload_events = self.event_pool.get_even_by_type("unload")
        unload_event_count = len(unload_events)
        unload_event_time = [event.time_stamp for event in unload_events]

        if unload_event_count == 0:
            return 0

        if unload_event_count == 1:
            return unload_event_time[0]

        cycle_time = (max(unload_event_time) - min(unload_event_time))/abs(unload_event_count - 1)
        return cycle_time

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