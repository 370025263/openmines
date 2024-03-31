from __future__ import annotations

import json

import numpy as np

from openmines.src.charging_site import ChargingSite
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.dump_site import DumpSite, Dumper
from openmines.src.road import Road
from openmines.src.truck import Truck


class FixedGroupDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "FixedGroupDispatcher"
        self.group_solution = None  # Solution will be stored here after computation

    def compute_solution(self, mine: "Mine"):
        if self.group_solution is not None:
            return  # Solution already computed

        self.group_solution = {}  # Initialize group_solution
        load_sites = mine.load_sites
        dump_sites = mine.dump_sites

        # Compute load site productivity
        for load_site in load_sites:
            load_site_productivity = sum(
                shovel.shovel_tons / shovel.shovel_cycle_time for shovel in load_site.shovel_list)
            load_site.load_site_productivity = load_site_productivity

        total_load_site_productivity = sum(load_site.load_site_productivity for load_site in load_sites)

        # Calculate load site capacity
        trucks = mine.trucks
        total_truck_capacity = sum(truck.truck_capacity for truck in trucks)
        load_site_capacity = {
            load_site.name: (load_site.load_site_productivity / total_load_site_productivity) * total_truck_capacity
            for load_site in load_sites
        }

        # Initialize the group solution structure
        for load_site in load_sites:
            self.group_solution[load_site.name] = {
                "trucks": [],
                "dumpsite": None
            }

        # Distribute trucks to load sites
        for truck in trucks:
            for load_site in load_sites:
                if sum(t.truck_capacity for t in self.group_solution[load_site.name]["trucks"]) < load_site_capacity[
                    load_site.name]:
                    self.group_solution[load_site.name]["trucks"].append(truck)
                    break

        # Match load sites with the nearest dump site
        for load_site in load_sites:
            distances = [np.linalg.norm(np.array(load_site.position) - np.array(dump_site.position)) for dump_site in
                         dump_sites]
            nearest_dump_site = dump_sites[distances.index(min(distances))]
            self.group_solution[load_site.name]["dumpsite"] = nearest_dump_site

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        if self.group_solution is None:
            self.compute_solution(mine)

        for load_site_name, info in self.group_solution.items():
            if truck in info["trucks"]:
                return mine.load_sites.index([ls for ls in mine.load_sites if ls.name == load_site_name][0])

        return 0

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        前往最近距离的卸载点卸载
        :param truck:
        :param mine:
        :return:
        """
        current_location = truck.current_location
        assert isinstance(current_location, LoadSite), "current_location is not a LoadSite"
        cur_index = mine.load_sites.index(current_location)
        cur_to_dump = mine.road.road_matrix[cur_index, :]
        min_index = cur_to_dump.argmin()
        return min_index

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        查询当前车辆所在的装载点并返回
        :param truck:
        :param mine:
        :return:
        """
        for load_site_name, info in self.group_solution.items():
            if truck in info["trucks"]:
                return mine.load_sites.index([ls for ls in mine.load_sites if ls.name == load_site_name][0])

        return 0  # This can be expanded based on specific logic

# Test the dispatcher
if __name__ == "__main__":
    dispatcher = FixedGroupDispatcher()
    config_file = "/Users/mac/PycharmProjects/truck_shovel_mix/openmines_project/openmines/src/conf/north_pit_mine.json"
    from openmines.src.mine import Mine

    def load_config(filename):
        with open(filename, 'r') as file:
            return json.load(file)

    config = load_config(config_file)
    mine = Mine(config_file)
    # 初始化充电站和卡车
    charging_site = ChargingSite(config['charging_site']['name'], position=config['charging_site']['position'])
    for truck_config in config['charging_site']['trucks']:
        for _ in range(truck_config['count']):
            truck = Truck(
                name=f"{truck_config['type']}{_ + 1}",
                truck_capacity=truck_config['capacity'],
                truck_speed=truck_config['speed']
            )
            charging_site.add_truck(truck)

    # 初始化装载点和铲车
    for load_site_config in config['load_sites']:
        load_site = LoadSite(name=load_site_config['name'], position=load_site_config['position'])
        for shovel_config in load_site_config['shovels']:
            shovel = Shovel(
                name=shovel_config['name'],
                shovel_tons=shovel_config['tons'],
                shovel_cycle_time=shovel_config['cycle_time'],
                position_offset=shovel_config['position_offset']
            )
            load_site.add_shovel(shovel)
        load_site.add_parkinglot(position_offset=load_site_config['parkinglot']['position_offset'],
                                 name=load_site_config['parkinglot']['name'])
        mine.add_load_site(load_site)

    # 初始化卸载点和卸载机
    for dump_site_config in config['dump_sites']:
        dump_site = DumpSite(dump_site_config['name'], position=dump_site_config['position'])
        for dumper_config in dump_site_config['dumpers']:
            for _ in range(dumper_config['count']):
                dumper = Dumper(
                    name=f"{dump_site_config['name']}-点位{_}",
                    dumper_cycle_time=dumper_config['cycle_time'],
                    position_offset=dumper_config['position_offset']
                )
                dump_site.add_dumper(dumper)
        dump_site.add_parkinglot(position_offset=dump_site_config['parkinglot']['position_offset'],
                                 name=dump_site_config['parkinglot']['name'])
        mine.add_dump_site(dump_site)

    # 初始化道路
    road_matrix = np.array(config['road']['road_matrix'])
    road_event_params = config['road'].get('road_event_params', {})  # 从配置中加载道路事件参数

    charging_to_load_road_matrix = config['road']['charging_to_load_road_matrix']
    road = Road(road_matrix=road_matrix, charging_to_load_road_matrix=charging_to_load_road_matrix,
                road_event_params=road_event_params)
    # # 添加充电站和装载区卸载区
    mine.add_road(road)
    mine.add_charging_site(charging_site)
    ## 添加调度
    mine.add_dispatcher(dispatcher)
    dispatcher.compute_solution(mine)
    print(dispatcher.group_solution)
    # Assume mine and trucks are defined somewhere
    # print(dispatcher.give_init_order(truck, mine))
    # print(dispatcher.give_haul_order(truck, mine))
    # print(dispatcher.give_back_order(truck, mine))
