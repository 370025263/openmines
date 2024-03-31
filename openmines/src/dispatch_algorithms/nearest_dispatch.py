from __future__ import annotations

import time

import numpy as np

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.dump_site import DumpSite
from openmines.src.load_site import LoadSite


class NearestDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "NearestDispatcher"

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        charging_to_load = mine.road.charging_to_load
        # 获取最短数值的index
        min_index = charging_to_load.index(min(charging_to_load))
        return min_index

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        # 获取当前位置
        current_location = truck.current_location
        assert isinstance(current_location, LoadSite), "current_location is not a LoadSite"
        # 获取当前位置的index
        cur_index = mine.load_sites.index(current_location)
        # 获取当前位置到所有dump site的距离
        cur_to_dump:np.array = mine.road.road_matrix[cur_index,:]
        # 获取nparray最短数值的index
        min_index = cur_to_dump.argmin()
        # 从第一个load site开始
        return min_index

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        # 获取当前位置
        current_location = truck.current_location
        assert isinstance(current_location, DumpSite), "current_location is not a DumpSite"
        # 获取当前位置的index
        cur_index = mine.dump_sites.index(current_location)
        # 获取当前位置到所有load site的距离
        cur_to_load = mine.road.road_matrix[:,cur_index]
        # 获取nparray最短数值的index
        min_index = cur_to_load.argmin()
        return min_index

if __name__ == "__main__":
    dispatcher = NearestDispatcher()
    print(dispatcher.give_init_order(1,2))
    print(dispatcher.give_init_order(1, 2))
    print(dispatcher.give_init_order(1, 2))
    print(dispatcher.give_init_order(1, 2))
    print(dispatcher.give_haul_order(1,2))
    print(dispatcher.give_back_order(1,2))

    print(dispatcher.total_order_count,dispatcher.init_order_count,dispatcher.init_order_time,dispatcher.total_order_time)
