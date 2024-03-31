from __future__ import annotations

from typing import List

import numpy as np  # 导入NumPy库
import random,json,time
import openai


from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.dump_site import DumpSite, Dumper


class SPTFDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "SPTFDispatcher"
        self.np_random = np.random.RandomState()  # 创建NumPy的随机状态对象

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        # 获取其他卡车信息
        other_trucks = [each_truck for each_truck in mine.trucks if each_truck.name != truck.name]
        # 统计其他卡车的目的地并进行计数;过滤掉target_location为None的卡车;
        target_location_trucks = [each_truck for each_truck in other_trucks if each_truck.target_location is not None and isinstance(each_truck.target_location, LoadSite)]
        # 统计loadsite信息
        load_sites = mine.load_sites
        for _, each_load_site in enumerate(load_sites):
            each_load_site.coming_truck_list = []
            # 统计target_locations中本loadsite的数量
            for each_coming_truck in target_location_trucks:
                if each_coming_truck.target_location.name == each_load_site.name:
                    each_load_site.coming_truck_list.append(each_coming_truck)
            # processing time = coming truck processing time + estimated queue wait time
            each_load_site.processing_time_est = sum([each_truck.truck_capacity/each_load_site.load_site_productivity for each_truck in each_load_site.coming_truck_list]) + each_load_site.estimated_queue_wait_time

        # 计算index
        loadsite_index = np.argmin([loadsite.processing_time_est for loadsite in load_sites])

        return loadsite_index

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        # 获取其他卡车信息
        other_trucks = [each_truck for each_truck in mine.trucks if each_truck.name != truck.name]
        # 统计其他卡车的目的地并进行计数;过滤掉target_location为None的卡车;
        target_location_trucks = [each_truck for each_truck in other_trucks if truck.target_location is not None and isinstance(each_truck.target_location, DumpSite)]
        # 统计dumpsite信息
        dump_sites = mine.dump_sites
        for each_dump_site in dump_sites:
            each_dump_site.coming_truck_list: List["Truck"] = []
            for each_coming_truck in target_location_trucks:
                if each_coming_truck.target_location.name == each_dump_site.name:
                    each_dump_site.coming_truck_list.append(each_coming_truck)
            each_dump_site.processing_time_est = len(each_dump_site.coming_truck_list) + each_dump_site.estimated_queue_wait_time

        dumpsite_index = np.argmin([dumpsite.processing_time_est for dumpsite in dump_sites])
        return dumpsite_index

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        # 这个方法可以与give_init_order方法完全一样，因为它们的逻辑是相同的
        return self.give_init_order(truck, mine)



if __name__ == "__main__":
    dispatcher = SPTFDispatcher()
    print(dispatcher.give_init_order(1,2))
    print(dispatcher.give_haul_order(1,2))
    print(dispatcher.give_back_order(1,2))

    print(dispatcher.total_order_count,dispatcher.init_order_count)