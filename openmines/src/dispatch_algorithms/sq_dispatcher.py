from __future__ import annotations
import numpy as np  # 导入NumPy库
import random,json,time
import openai


from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.dump_site import DumpSite, Dumper


class SQDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "SQDispatcher"
        self.np_random = np.random.RandomState()  # 创建NumPy的随机状态对象

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        # 获取其他卡车信息
        other_trucks = [each_truck for each_truck in mine.trucks if each_truck.name != truck.name]
        # 统计其他卡车的目的地并进行计数;过滤掉target_location为None的卡车;
        target_locations = [each_truck.target_location for each_truck in other_trucks if each_truck.target_location is not None and isinstance(each_truck.target_location, LoadSite)]
        # 统计loadsite信息
        load_sites = mine.load_sites
        for each_load_site in load_sites:
            each_load_site.coming_truck_num = 0
            # 统计target_locations中本loadsite的数量
            for each_target_location in target_locations:
                if each_target_location.name == each_load_site.name:
                    each_load_site.coming_truck_num += 1
        loadsite_queue_length = [load_site.parking_lot.queue_status["total"][int(mine.env.now)] for load_site in load_sites]
        # estimated_loadsite_queue_wait_times = [load_site.estimated_queue_wait_time for load_site in load_sites]
        # 选取coming_truck+queue_length最小的loadsite
        loadsite_index = np.argmin([loadsite.coming_truck_num + loadsite_queue_length[i] for i, loadsite in enumerate(load_sites)])


        # 获取dumpsite信息
        #avaliable_dumpsites = [dumpsite for dumpsite in mine.dump_sites if dumpsite.parking_lot is not None]
        #dump_site_names = [dumpsite.name for dumpsite in avaliable_dumpsites]
        #dumpsite_queue_length = [dumpsite.parking_lot.queue_status["total"][int(mine.env.now)] for dumpsite in
        #                        avaliable_dumpsites]
        #estimated_dumpsite_queue_wait_times = [dumpsite.estimated_queue_wait_time for dumpsite in avaliable_dumpsites]
        return loadsite_index

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        cur_location:LoadSite = truck.current_location
        # 获取其他卡车信息
        other_trucks = [each_truck for each_truck in mine.trucks if each_truck.name != truck.name]
        # 统计其他卡车的目的地并进行计数;过滤掉target_location为None的卡车;
        target_locations = [each_truck.target_location for each_truck in other_trucks if
                            each_truck.target_location is not None and isinstance(each_truck.target_location, DumpSite)]
        # 统计dumpsite信息
        dump_sites = mine.dump_sites
        for each_dump_site in dump_sites:
            each_dump_site.coming_truck_num = 0
            # 统计target_locations中本dumpsite的数量
            for each_target_location in target_locations:
                if each_target_location.name == each_dump_site.name:
                    each_dump_site.coming_truck_num += 1
        dumpsite_queue_length = [dump_site.parking_lot.queue_status["total"][int(mine.env.now)] for dump_site in
                                 dump_sites]
        # estimated_loadsite_queue_wait_times = [load_site.estimated_queue_wait_time for load_site in load_sites]
        # 选取coming_truck+queue_length最小的loadsite
        dumpsite_index = np.argmin(
            [dumpsite.coming_truck_num + dumpsite_queue_length[i] for i, dumpsite in enumerate(dump_sites)])

        # 获取dumpsite信息
        # avaliable_dumpsites = [dumpsite for dumpsite in mine.dump_sites if dumpsite.parking_lot is not None]
        # dump_site_names = [dumpsite.name for dumpsite in avaliable_dumpsites]
        # dumpsite_queue_length = [dumpsite.parking_lot.queue_status["total"][int(mine.env.now)] for dumpsite in
        #                        avaliable_dumpsites]
        # estimated_dumpsite_queue_wait_times = [dumpsite.estimated_queue_wait_time for dumpsite in avaliable_dumpsites]
        return dumpsite_index

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        # 获取其他卡车信息
        other_trucks = [each_truck for each_truck in mine.trucks if each_truck.name != truck.name]
        # 统计其他卡车的目的地并进行计数;过滤掉target_location为None的卡车;
        target_locations = [each_truck.target_location for each_truck in other_trucks if each_truck.target_location is not None and isinstance(each_truck.target_location, LoadSite)]
        # 统计loadsite信息
        load_sites = mine.load_sites
        for each_load_site in load_sites:
            each_load_site.coming_truck_num = 0
            # 统计target_locations中本loadsite的数量
            for each_target_location in target_locations:
                if each_target_location.name == each_load_site.name:
                    each_load_site.coming_truck_num += 1
        loadsite_queue_length = [load_site.parking_lot.queue_status["total"][int(mine.env.now)] for load_site in load_sites]
        # estimated_loadsite_queue_wait_times = [load_site.estimated_queue_wait_time for load_site in load_sites]
        # 选取coming_truck+queue_length最小的loadsite
        loadsite_index = np.argmin([loadsite.coming_truck_num + loadsite_queue_length[i] for i, loadsite in enumerate(load_sites)])


        # 获取dumpsite信息
        #avaliable_dumpsites = [dumpsite for dumpsite in mine.dump_sites if dumpsite.parking_lot is not None]
        #dump_site_names = [dumpsite.name for dumpsite in avaliable_dumpsites]
        #dumpsite_queue_length = [dumpsite.parking_lot.queue_status["total"][int(mine.env.now)] for dumpsite in
        #                        avaliable_dumpsites]
        #estimated_dumpsite_queue_wait_times = [dumpsite.estimated_queue_wait_time for dumpsite in avaliable_dumpsites]
        return loadsite_index



if __name__ == "__main__":
    dispatcher = SQDispatcher()
    print(dispatcher.give_init_order(1,2))
    print(dispatcher.give_haul_order(1,2))
    print(dispatcher.give_back_order(1,2))

    print(dispatcher.total_order_count,dispatcher.init_order_count)