from __future__ import annotations
import numpy as np
from scipy.stats import norm
import random

from openmines.src.charging_site import ChargingSite
from openmines.src.dump_site import DumpSite
from openmines.src.load_site import LoadSite
from openmines.src.utils.event import Event


class Road:
    # 目前，Road本质是一个二维数组
    # 以后可能会添加SUMO的路网
    # 根本作用是用于仿真订单到达时间
    def __init__(self, road_matrix: np.ndarray, charging_to_load_road_matrix: list[float], road_event_params=None):
        self.road_matrix: np.ndarray = road_matrix
        self.load_site_num: int = road_matrix.shape[0]
        self.dump_site_num: int = road_matrix.shape[1]
        self.charging_to_load: list = charging_to_load_road_matrix
        self.road_status = None
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


    def truck_on_road(self, start, end)->list["Truck"]:
        """
        返回从start到end的道路上的车辆列表(状态为在路上而不是已经到达)
        :param start:
        :param end:
        :return:
        """
        if self.env.now < 2:
            return []

        # 从start到end的道路上的车辆列表
        truck_list = []
        for truck in self.mine.trucks:
            if truck.current_location.name == start.name and truck.target_location.name == end.name and truck.status == "moving":
                truck_list.append(truck)
        return truck_list

    def road_jam_sampling(self, start, end):
        """
        从start到end的道路上的堵车采样，将新事件放入mine的event-pool
        :param start: 开始地点对象
        :param end: 结束地点对象
        :return:无
        """
        # 从start到end的道路上的车辆列表
        trucks_on_road = self.truck_on_road(start, end)
        if len(trucks_on_road) == 0:
            return
        else:
            pass
        # 从start到end的道路长度
        distance = self.get_distance(trucks_on_road[0], end, enable_event=False)
        # 为每个同路的车辆根据当前时间计算route_coverage数值
        for truck in trucks_on_road:
            truck.journey_coverage = truck.get_route_coverage(distance)
        # 假设每个车辆导致堵车的概率在route_coverage上满足正态分布，均值为route_coverage，方差为0.1
        # other_trucks_on_road中的每个车导致堵车的概率会叠加在一起取平均成为最终的堵车概率分布
        truck_positions = [truck.journey_coverage for truck in trucks_on_road]
        # 每个卡车的正态分布的标准差
        sd = 0.1  # 根据3sigma原则，这个值代表这个车辆对堵车的影响会在+-0.3总路程的范围内
        # 位置的范围
        x = np.linspace(0, 1, 1000)
        # 初始化堵车的概率为0
        total_prob = np.zeros_like(x)
        # 对每辆卡车
        for i, position in enumerate(truck_positions):
            # 计算在每个位置的堵车概率
            prob = norm.pdf(x, position, sd)
            # 将这个卡车的堵车概率加到总的堵车概率上
            total_prob = total_prob + prob
        # 正规化总的堵车概率
        total_prob = total_prob / len(truck_positions)
        # 使用蒙特卡洛方法来决定是否会发生堵车
        if np.sum(total_prob) == 0:
            probabilities = np.ones_like(total_prob) / len(total_prob)
        else:
            probabilities = total_prob / (np.sum(total_prob))
        jam_position = np.random.choice(x, p=probabilities)
        # 在jam_position处使用weibull分布，采样一个可能的堵车时间
        jam_time = np.random.weibull(2) * 10
        # 采样完成，放入pool
        self.mine.random_event_pool.add_event(Event(self.env.now, "RoadEvent:jam",
                                                    f'Time:<{self.env.now}> New RoadEvent:jam at road({jam_position}) from {start.name} '
                                                    f'to {end.name} for {jam_time:.2f} minutes',
                                                    info={"name": "RoadEvent:jam", "status": "jam", "speed": 0,
                                                          "start_location": start.name,
                                                          "end_location": end.name,
                                                          "jam_position": jam_position,
                                                          "start_time": self.env.now,
                                                          "est_end_time": self.env.now + jam_time}))


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
            # 计算下一次检修可能发生的时间
            next_repair_time = repair_end_time + random.expovariate(self.lambda_repair)
            if self.env.now >= next_repair_time:  # 判断是否发生检修
                # 生成检修持续时间
                repair_duration = max(0, np.random.normal(self.mu_repair_duration, self.sigma_repair_duration))
                in_repair = True
                repair_end_time = self.env.now + repair_duration
                self.mine.random_event_pool.add_event(Event(self.env.now, "RoadEvent:repair",
                                                            f'Road from {current_site.name} '
                                                            f'to {target_site.name} is in repair, '
                                                            f'the repair will last for {repair_duration:.2f} mins, '
                                                            f'and will end at {repair_end_time:.2f}, '
                                                            f'vehicle on this road will have to suffer punish_distance.',
                                                            info={"road_id": road_id,
                                                                  "start_location": current_site.name,
                                                                    "end_location": target_site.name,
                                                                    "repair_start_time": self.env.now,
                                                                  "repair_end_time": repair_end_time,
                                                                  "repair_duration": repair_duration,
                                                                  }))
                self.logger.info(
                    f"[RoadEvent] Road from {current_site.name} to {target_site.name} is in repair，"
                    f"the repair will last for {repair_duration:.2f} mins, and will end at {repair_end_time:.2f}, "
                    f"vehicle on this road will have to suffer punish_distance.")
            else:
                in_repair = False

        # 更新道路的检修状态
        self.road_repairs[road_id] = (in_repair, repair_end_time)

        return in_repair

    def get_distance(self, truck: "Truck", target_site, enable_event:bool=True) -> float:
        current_site = truck.current_location
        assert type(current_site) != type(target_site), f"current_site and target_site should not be the same type of site " \
                                                        f"current_site: {current_site.name}, target_site: {target_site.name},Truck name: {truck.name}"
        # 如果只是检查道路可用性，则不触发随机事件
        if not enable_event:
            in_repair = False
        else:
            # 检查道路是否可用，并对检修状态进行模拟
            in_repair = self.check_availability(current_site, target_site)

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
