from __future__ import annotations
import simpy
import numpy as np

from sisymines.src.charging_site import ChargingSite
from sisymines.src.dump_site import DumpSite
from sisymines.src.load_site import LoadSite


class Road:
    # 目前，Road本质是一个二维数组
    # 以后可能会添加SUMO的路网
    # 根本作用是用于仿真订单到达时间
    def __init__(self, road_matrix: np.ndarray, charging_to_load_road_matrix: list[float]):
        self.road_matrix: np.ndarray = road_matrix
        self.load_site_num:int = road_matrix.shape[0]
        self.dump_site_num:int = road_matrix.shape[1]
        self.charging_to_load:list = charging_to_load_road_matrix

    def set_env(self, mine:"Mine"):
        self.env = mine.env
        self.mine = mine

    def get_distance(self, truck:"Truck", target_site)->float:
        # 读取Truck的当前位置对象
        current_site = truck.current_location
        # 判断当前位置的对象类型，是DumpSite 还是 LoadSite 还是 ChargingSite
        if isinstance(current_site, DumpSite):
            # 取得标号
            load_site_id = self.mine.load_sites.index(target_site)
            dump_site_id = self.mine.dump_sites.index(current_site)
            distance = self.road_matrix[load_site_id][dump_site_id]
        elif isinstance(current_site, LoadSite):
            load_site_id = self.mine.load_sites.index(current_site)
            dump_site_id = self.mine.dump_sites.index(target_site)
            distance = self.road_matrix[load_site_id][dump_site_id]
        elif isinstance(current_site, ChargingSite):
            load_site_id = self.mine.load_sites.index(target_site)
            distance = self.charging_to_load[load_site_id]
        else:
            raise Exception("current_site is not a DumpSite or LoadSite or ChargingSite")
        return distance
