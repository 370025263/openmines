"""
at the beginning of the operation, all the trucks is in a special parking lot called charging site
when a truck is ready to be loaded, it will be sent to the load site

this file is for the charging site
"""
from __future__ import annotations

class ChargingSite:
    def __init__(self, name:str,position:tuple=(0,0)):
        self.name = name
        self.position = position
        self.mine = None
        self.trucks = []

    def set_mine(self, mine:"Mine"):
        assert mine is not None, "mine can not be None"
        self.mine = mine

    def add_truck(self, truck:"Truck"):
        assert truck is not None, "truck can not be None"
        self.trucks.append(truck)

    def get_distance_to_load_site(self, load_site_index:int)->float:
        assert load_site_index >= 0, "load_site_index can not be negative"
        assert load_site_index < self.mine.road.load_site_num, "load_site_index can not be larger than load_site_num"
        return self.mine.road.charging_to_load[load_site_index]