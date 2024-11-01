from __future__ import annotations

import time
from openmines.src.dispatcher import BaseDispatcher


class ShortestTripDisptatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "ShortestTripDispatcher"

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        charging_to_load = mine.road.charging_to_load
        # 计算车辆(从备停区)抵达各装载点时间(now_time 为当前仿真时间:Float, avg_velocity 为车辆平均配速:Float)
        reach_load_time = now_time + charging_to_load / avg_velocity
        # 计算车辆在各装载点预期获得服务时间(load_available_time 为各装载点处上一次装载的完成时间:np.array)
        service_available_time = np.maximum(load_available_time, reach_load_time)
        # 计算车辆驶往各装载点, 并最终完成装载的预期任务完成时间, 并取最小值 (loading_time 为各挖机装载时间:np.array)
        target = np.argmin(service_available_time + loading_time - now_time)

        return target

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        current_location = truck.current_location
        assert isinstance(current_location, LoadSite), "current_location is not a LoadSite"
        # 获取当前位置的index
        cur_index = mine.load_sites.index(current_location)
        # 获取当前位置到所有dump site的距离
        cur_to_dump:np.array = mine.road.road_matrix[cur_index,:]
        # 计算车辆抵达各卸载点时间(now_time 为当前仿真时间:Float, avg_velocity 为车辆平均配速:Float)
        reach_dump_time = now_time + cur_to_dump / avg_velocity
        # 计算车辆在各卸载点预期获得服务时间(dump_available_time 为各卸载点处上一次卸载的完成时间:np.array)
        service_available_time = np.maximum(dump_available_time, reach_dump_time)
        # 计算车辆驶往各卸载点, 并最终完成卸载的预期任务完成时间, 并取最小值 (loading_time 为各挖机卸载时间:np.array)
        target = np.argmin(service_available_time + unloading_time - now_time)

        return target

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        # 获取当前位置
        current_location = truck.current_location
        assert isinstance(current_location, DumpSite), "current_location is not a DumpSite"
        # 获取当前位置的index
        cur_index = mine.dump_sites.index(current_location)
        # 获取当前位置到所有load site的距离
        cur_to_load = mine.road.road_matrix[:, cur_index]
        # 计算车辆抵达各装载点时间(now_time 为当前仿真时间:Float, avg_velocity 为车辆平均配速:Float)
        reach_load_time = now_time + cur_to_load / avg_velocity
        # 计算车辆在各装载点预期获得服务时间(load_available_time 为各装载点处上一次装载的完成时间:np.array)
        service_available_time = np.maximum(load_available_time, reach_load_time)
        # 计算车辆驶往各装载点, 并最终完成装载的预期任务完成时间, 并取最小值 (loading_time 为各挖机装载时间:np.array)
        target = np.argmin(service_available_time + loading_time - now_time)

        return target


'''
注1: 其实 now_time 可以不用, 前面加一个now_time后面减一个, 等于没用
'''

'''
注2: 1) 在调度指令下达后, 需要同步更新装/卸载点的预期任务完成时间 (un)load_available_time, 
        将 (un)load_available_time 更新为: 所下达调度指令的车辆的预期任务完成时间 (在调度计算阶段已经计算得到), 作为下一次调度的依据
     
     2) 一个班次刚开始时, 所有设备都可用, 因此  (un)load_available_time 初始化为 now_time
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
