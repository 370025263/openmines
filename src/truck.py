import simpy

from sisymines.src.dump_site import DumpSite, Dumper
from sisymines.src.load_site import LoadSite

TRUCK_DEFAULT_SPEED = 25 # km/h

class Truck:
    def __init__(self, name:str, truck_capacity:float, truck_speed:float=TRUCK_DEFAULT_SPEED):
        self.name = name
        self.truck_capacity = truck_capacity  # truck capacity in tons
        self.truck_speed = truck_speed  # truck speed in km/h
        self.current_location = None

    def set_env(self, mine:"Mine"):
        self.mine = mine
        self.env = mine.env
        self.dispatcher = mine.dispatcher

    def move(self, target_location, distance:float, manual_speed:float=None):
        assert target_location is not None, "target_location can not be None"
        assert target_location is not self.current_location, "target_location can not be the same as current_location"
        assert distance >= 0, "distance can not be negative"
        assert self.truck_speed > 0, "truck_speed can not be negative"
        if manual_speed is not None:
            assert manual_speed > 0, "manual_speed can not be negative"
            duration = distance / manual_speed
        else:
            duration = distance / self.truck_speed
        yield self.env.timeout(duration)
        # after arrival set current location
        self.current_location = target_location

    def load(self, shovel):
        shovel_tons = shovel.shovel_tons
        shovel_cycle_time = shovel.shovel_cycle_time
        load_time = (shovel_tons/self.truck_capacity) * shovel_cycle_time
        print(f'Time:<{self.env.now}> Truck:[{self.name}] Start loading at shovel {shovel.name}, load time is {load_time}')
        yield self.env.timeout(load_time)
        print(f'Time:<{self.env.now}> Truck:[{self.name}] Finish loading at shovel {shovel.name}')

    def unload(self, dumper:Dumper):
        unload_time:float = dumper.dump_time
        yield self.env.timeout(unload_time)
        dumper.dumper_tons += self.truck_capacity
        print(f'Time:<{self.env.now}> Truck:[{self.name}] Finish unloading at dumper {dumper.name}, dumper tons is {dumper.dumper_tons}')


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
        # 轮班开始 车辆从充电区域前往装载区
        dest_load_index: int = self.dispatcher.give_init_order(truck=self, mine=self.mine)

        move_distance:float = self.mine.road.charging_to_load[dest_load_index]
        load_site: LoadSite = self.mine.load_sites[dest_load_index]
        print(f'Time:<{self.env.now}> Truck:[{self.name}] Activated at {self.env.now}, Target dump site is ORDER({dest_load_index}):{load_site.name}, move distance is {move_distance}')
        yield self.env.process(self.move(target_location=load_site, distance=move_distance))  # 移动时间

        while True:
            # 到达装载区开始请求资源并装载
            print(f'Time:<{self.env.now}> Truck:[{self.name}] Arrived at {self.mine.load_sites[dest_load_index].name} at {self.env.now}')
            load_site:LoadSite = self.mine.load_sites[dest_load_index]
            shovel = load_site.get_available_shovel()

            with shovel.res.request() as req:
                # 申请到资源之前的操作
                # ...
                yield req  # 申请铲车资源
                # 申请到铲车资源
                yield self.env.process(self.load(shovel))  # 装载时间 同shovel和truck自身有关系

            # 装载完毕，请求新的卸载区，并开始移动到卸载区
            dest_unload_index: int = self.dispatcher.give_haul_order(truck=self, mine=self.mine)
            dest_unload_site: DumpSite = self.mine.dump_sites[dest_unload_index]
            move_distance: float = self.mine.road.get_distance(truck=self, target_site=dest_unload_site)
            print(f"Time:<{self.env.now}> Truck:[{self.name}] Start moving to ORDER({dest_unload_index}): {dest_unload_site.name}, move distance is {move_distance}, speed: {self.truck_speed}")
            yield self.env.process(self.move(target_location=dest_unload_site, distance=move_distance))  # 移动时间

            # 到达卸载区并开始请求资源并卸载
            print(f'Time:<{self.env.now}> Truck:[{self.name}] Arrived at {dest_unload_site.name} at {self.env.now}')
            dumper:Dumper = dest_unload_site.get_available_dumper()
            with dumper.res.request() as req:
                # 申请到资源之前的操作
                # ...
                yield req # 申请卸载位资源
                # 申请到卸载位资源
                yield self.env.process(self.unload(dumper))

            # 卸载完毕，请求新的装载区，并开始移动到装载区
            dest_load_index: int = self.dispatcher.give_back_order(truck=self, mine=self.mine)
            dest_load_site: LoadSite = self.mine.load_sites[dest_load_index]
            move_distance: float = self.mine.road.get_distance(truck=self, target_site=dest_load_site)
            print(f"Time:<{self.env.now}> Truck:[{self.name}] Start moving to ORDER({dest_load_index}):{dest_load_site.name}, move distance is {move_distance}, speed: {self.truck_speed}")
            yield self.env.process(self.move(target_location=dest_load_site, distance=move_distance))  # 移动时间

    def charge(self, duration):
        """
        电矿卡需要充电
        油矿卡不需要充电
        TODO：针对卡车油耗、电耗、充电时间进行建模。并应用。
        :param duration:
        :return:
        """
        print(f'{self.name} Start charging at {self.env.now}')
        yield self.env.timeout(duration)