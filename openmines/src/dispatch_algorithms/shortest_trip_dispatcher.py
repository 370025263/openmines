from __future__ import annotations
import time, numpy as np

from openmines.src.dispatcher import BaseDispatcher


class ShortestTripDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "ShortestTripDispatcher"

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        charging_to_load:list[float] = mine.road.charging_to_load  # 从备停区到各装载点的距禽(千米)
        avg_velocity:float = truck.truck_speed  # 车辆平均配速(千米/小时)
        now_time = mine.env.now  # 当前仿真时间(min). Actually, this is just 0.0 because it's init order which launches at the beginning of the simulation

        # 计算车辆(从备停区)抵达各装载点时间(now_time 为当前仿真时间:Float, avg_velocity 为车辆平均配速:np.array. 1D Float)
        reach_load_time = now_time + 60*np.array(charging_to_load) / avg_velocity

        # 计算车辆在各装载点预期获得服务时间(load_available_time is the time when the last truck finished loading at each load site:np.array)
        trucks_on_roads = [mine.road.truck_on_road(start=mine.charging_site, end=mine.load_sites[i]) for i in range(len(mine.load_sites))]
        load_time_on_road_trucks = [sum([each_truck.truck_capacity for each_truck in each_road])/mine.load_sites[i].load_site_productivity for i,each_road in enumerate(trucks_on_roads)]  # 路上的卡车需要的装载用时
        load_available_time = now_time + np.array([load_site.estimated_queue_wait_time for load_site in mine.load_sites]) + np.array(load_time_on_road_trucks)  # 装载点的预期服务时间， 包含了路上的卡车和正在排队的卡车
        service_available_time = np.maximum(load_available_time, reach_load_time)

        # 计算车辆驶往各装载点, 并最终完成装载的预期任务完成时间, 并取最小值 (loading_time 为各挖机装载时间:np.array)
        assert mine.load_sites[0].load_site_productivity != 0, "load_site_productivity is 0"
        loading_service_time = np.array([truck.truck_capacity/ (load_site.load_site_productivity/len(load_site.shovel_list)) for load_site in mine.load_sites])  # mins
        avl = np.argmin(service_available_time + loading_service_time - now_time)

        return avl

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        current_location = truck.current_location
        assert current_location.__class__.__name__ == "LoadSite", "current_location is not a LoadSite"
        avg_velocity = truck.truck_speed  # 车辆平均配速(千米/小时)
        now_time = mine.env.now  # 当前仿真时间(min)
        cur_index = mine.load_sites.index(current_location)  # 获取当前位置的index

        # 获取当前位置到所有dump site的距离
        cur_to_dump:np.array = mine.road.l2d_road_matrix[cur_index,:]

        # 计算车辆抵达各卸载点时间(now_time 为当前仿真时间:Float, avg_velocity 为车辆平均配速:Float)
        reach_dump_time = now_time + 60*cur_to_dump / avg_velocity

        trucks_on_roads = [mine.road.truck_on_road(start=truck.current_location, end=mine.dump_sites[i]) for i in range(len(mine.dump_sites))]
        dump_time_on_road_trucks = [(sum([1 for each_truck in each_road]) / len(mine.dump_sites[i].dumper_list) )*mine.dump_sites[i].dumper_list[0].dump_time for i, each_road in enumerate(trucks_on_roads)]  # 路上的卡车需要的装载用时
        # 计算车辆在各卸载点预期获得服务时间(dump_available_time 为各卸载点处上一次卸载的完成时间:np.array)
        dump_available_time = now_time + np.array([dump_site.estimated_queue_wait_time for dump_site in mine.dump_sites]) + np.array(dump_time_on_road_trucks)
        service_available_time = np.maximum(dump_available_time, reach_dump_time)

        # 计算车辆驶往各卸载点, 并最终完成卸载的预期任务完成时间, 并取最小值 (loading_time 为各挖机卸载时间:np.array)
        unloading_time = np.array([dump_site.dumper_list[0].dump_time for dump_site in mine.dump_sites])
        target = np.argmin(service_available_time + unloading_time - now_time)

        return target

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        current_location = truck.current_location
        assert current_location.__class__.__name__ == "DumpSite", "current_location is not a DumpSite"
        avg_velocity = truck.truck_speed  # 车辆平均配速(千米/小时)
        now_time = mine.env.now
        cur_index = mine.dump_sites.index(current_location)

        # 获取当前位置到所有装载点的距离
        cur_to_load = mine.road.d2l_road_matrix[:,cur_index]

        # 计算车辆抵达各装载点时间(now_time 为当前仿真时间:Float, avg_velocity 为车辆平均配速:Float)
        reach_load_time = now_time + 60*cur_to_load / avg_velocity

        trucks_on_roads = [mine.road.truck_on_road(start=mine.charging_site, end=mine.load_sites[i]) for i in range(len(mine.load_sites))]
        load_time_on_road_trucks = [sum([each_truck.truck_capacity for each_truck in each_road]) / mine.load_sites[i].load_site_productivity for i, each_road in enumerate(trucks_on_roads)]  # 路上的卡车需要的装载用时
        # 计算车辆在各装载点预期获得服务时间(load_available_time is the time when the last truck finished loading at each load site:np.array)
        load_available_time = now_time + np.array([load_site.estimated_queue_wait_time for load_site in mine.load_sites]) + np.array(load_time_on_road_trucks)
        service_available_time = np.maximum(load_available_time, reach_load_time)

        # 计算车辆驶往各装载点, 并最终完成装载的预期任务完成时间, 并取最小值 (loading_time 为各挖机装载时间:np.array)
        assert mine.load_sites[0].load_site_productivity != 0, "load_site_productivity is 0"
        loading_time = np.array(
            [truck.truck_capacity / (load_site.load_site_productivity / len(load_site.shovel_list)) for load_site in
             mine.load_sites])  # mins
        target = np.argmin(service_available_time + loading_time - now_time)

        return target


'''
注1: 其实 now_time 可以不用, 前面加一个now_time后面减一个, 等于没用
'''

'''
注2: 1) 在调度指令下达后, 需要同步更新装/卸载点的预期任务完成时间 (un)load_available_time, 
        将 (un)load_available_time 更新为: 所下达调度指令的车辆的预期任务完成时间 (在调度计算阶段已经计算得到), 作为下一次调度的依据.
        [DONE] the (un)load_available_time is lazy calculated here. it will always be new.
     
     2) 一个班次刚开始时, 所有设备都可用, 因此  (un)load_available_time 初始化为 now_time [DONE]
'''

# if __name__ == "__main__":
#     dispatcher = TabuSearchDispatcher()
#     print(dispatcher.give_init_order(1,2))
#     print(dispatcher.give_init_order(1, 2))
#     print(dispatcher.give_init_order(1, 2))
#     print(dispatcher.give_init_order(1, 2))
#     print(dispatcher.give_haul_order(1,2))
#     print(dispatcher.give_back_order(1,2))
#
#     print(dispatcher.total_order_count,dispatcher.init_order_count,dispatcher.init_order_time,dispatcher.total_order_time)
