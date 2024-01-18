from __future__ import annotations
import simpy
import numpy as np
import random

from sisymines.src.charging_site import ChargingSite
from sisymines.src.dump_site import DumpSite
from sisymines.src.load_site import LoadSite


class Road:
    # 目前，Road本质是一个二维数组
    # 以后可能会添加SUMO的路网
    # 根本作用是用于仿真订单到达时间
    def __init__(self, road_matrix: np.ndarray, charging_to_load_road_matrix: list[float], road_event_params=None):
        self.road_matrix: np.ndarray = road_matrix
        self.load_site_num: int = road_matrix.shape[0]
        self.dump_site_num: int = road_matrix.shape[1]
        self.charging_to_load: list = charging_to_load_road_matrix

        # 默认参数
        default_params = {
            'lambda_repair': 1 / (60 * 2),  # 指数分布的lambda值
            'mu_repair_duration': 60,  # 检修持续时间的平均值
            'sigma_repair_duration': 10,  # 检修持续时间的标准差
            'mu_punish_distance': 0.4,  # 绕道百分比的平均值
            'sigma_punish_distance': 0.1  # 绕道百分比的标准差
        }

        # 如果提供了参数，则更新
        if road_event_params:
            default_params.update(road_event_params)

        # 设置参数
        self.lambda_repair = default_params['lambda_repair']
        self.mu_repair_duration = default_params['mu_repair_duration']
        self.sigma_repair_duration = default_params['sigma_repair_duration']
        self.mu_punish_distance = default_params['mu_punish_distance']
        self.sigma_punish_distance = default_params['sigma_punish_distance']
        self.in_repair = False  # 检修状态标志
        self.repair_end_time = 0  # 检修结束的模拟时间
        # 维护每条道路的检修状态
        self.road_repairs = {}  # {(current_site, target_site): (in_repair, repair_end_time)}

    def set_env(self, mine: "Mine"):
        self.env = mine.env
        self.mine = mine
        self.logger = self.mine.global_logger.get_logger("Road")

    def check_availability(self, current_site, target_site):
        """
        检查道路可用性，使用指数分布模拟道路检修事件，一旦检修事件发生，道路上新增车辆将会遭受距离增加惩罚
        :param current_site:
        :param target_site:
        :return:
        """
        # 确定道路的唯一标识
        road_id = (current_site, target_site)

        # 获取当前道路的检修状态
        in_repair, repair_end_time = self.road_repairs.get(road_id, (False, 0))

        # 检查当前时间是否在检修时间内
        if self.env.now < repair_end_time:
            in_repair = True
        else:
            # 检查是否发生检修
            if random.expovariate(self.lambda_repair) < 1:  # 使用指数分布生成随机数，模拟检修事件发生
                # 生成检修持续时间
                repair_duration = max(0, np.random.normal(self.mu_repair_duration, self.sigma_repair_duration))
                in_repair = True
                repair_end_time = self.env.now + repair_duration
                self.logger.info(
                    f"[RoadEvent] Road from {current_site.name} to {target_site.name} is in repair，"
                    f"the repair will last for {repair_duration:.2f} mins, and will end at {repair_end_time:.2f}, "
                    f"vehicle on this road will have to suffer punish_distance.")
            else:
                in_repair = False

        # 更新道路的检修状态
        self.road_repairs[road_id] = (in_repair, repair_end_time)

        return in_repair

    def get_distance(self, truck: "Truck", target_site) -> float:
        current_site = truck.current_location

        # 检查道路是否可用
        in_repair = self.check_availability(current_site, target_site)

        # 正常情况下计算距离
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

        # 如果检修，则增加额外的绕道距离
        if in_repair:
            punish_distance_percentage = max(0, np.random.normal(self.mu_punish_distance, self.sigma_punish_distance))
            distance *= (1 + punish_distance_percentage)
        return distance
