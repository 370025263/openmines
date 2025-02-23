from __future__ import annotations
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import random,math

from openmines.src.charging_site import ChargingSite
from openmines.src.dump_site import DumpSite
from openmines.src.load_site import LoadSite
from openmines.src.utils.event import Event


class Road:
    """
    道路类，包含三个主要的距离矩阵：
    - l2d_road_matrix: 装载点到卸载点的距离矩阵，l2d_road_matrix[i][j]表示从装载点i到卸载区j的距离
    - d2l_road_matrix: 卸载点到装载点的距离矩阵，d2l_road_matrix[i][j]表示从卸载区i到装载点j的距离
    - charging_to_load_road_matrix: 充电站到装载点的距离列表，charging_to_load_road_matrix[i]表示从充电站到装载点i的距离
    """
    def __init__(self, l2d_road_matrix: np.ndarray, d2l_road_matrix: np.ndarray, charging_to_load_road_matrix: list[float], road_event_params=None):
        """
        初始化道路对象
        
        Args:
            l2d_road_matrix (np.ndarray): 装载点到卸载点的距离矩阵
            d2l_road_matrix (np.ndarray): 卸载点到装载点的距离矩阵
            charging_to_load_road_matrix (list[float]): 充电站到装载点的距离列表
            road_event_params (dict, optional): 道路事件参数
        """
        self.l2d_road_matrix: np.ndarray = l2d_road_matrix  # 装载点到卸载点的距离矩阵
        self.d2l_road_matrix: np.ndarray = d2l_road_matrix  # 卸载点到装载点的距离矩阵
        self.load_site_num: int = l2d_road_matrix.shape[0]  # 装载点数量
        self.dump_site_num: int = l2d_road_matrix.shape[1]  # 卸载点数量
        self.charging_to_load: list = charging_to_load_road_matrix  # 充电站到装载点的距离列表
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
        # 从start到end的道路上的车辆列表
        truck_list = []
        for truck in self.mine.trucks:
            if truck.current_location is None or truck.target_location is None:
                continue
            if truck.current_location.name == start.name and truck.target_location.name == end.name and truck.status == "moving":
                truck_list.append(truck)
        return truck_list

    def plot_road_jam_events(self, jam_position, total_prob, current_time):
        """
        绘制当前堵车事件及其对应的概率密度函数
        :param jam_position: 当前堵车事件的位置
        :param total_prob: 当前的堵车概率分布
        :param current_time: 当前时间，用于决定颜色的深浅
        """
        # 获取现有事件（在某个时刻已有的堵车事件）
        events = self.mine.random_event_pool.get_even_by_type("RoadEvent:jam")

        # 基于调度算法名称生成颜色
        def generate_color(dispatcher_name):
            """ 根据调度算法名称生成颜色 """
            hash_value = hash(dispatcher_name)  # 使用hash函数生成整数
            np.random.seed(hash_value % (2**32))  # 使用hash值作为随机数生成器的种子
            color = np.random.rand(3, )  # 生成一个RGB颜色
            return color

        # 设置颜色渐变（随着时间的推移，颜色变淡）
        plt.figure(figsize=(8, 6))

        # 为每个事件根据调度算法生成颜色
        for event in events:
            dispatcher_name = self.mine.dispatcher.name
            color = generate_color(dispatcher_name)  # 根据算法名字生成颜色
            time_diff = current_time - event.info["start_time"]
            # 使用时间差来设置颜色的透明度（越老的事件越透明）
            alpha_value = max(0.1, 1 - time_diff / 100)  # 透明度范围 [0.1, 1]
            plt.axvline(event.info["jam_position"], color=color, linestyle='--', alpha=alpha_value,
                        label=f"{dispatcher_name} Jam at {event.info['jam_position']:.3f}")

        # 绘制当前堵车的概率密度函数
        plt.plot(np.linspace(0, 1, 1000), total_prob, color="green", label="Current Probability Density")

        # 绘制最新的堵车位置，用三角形标记
        plt.scatter(jam_position, total_prob[np.argmin(np.abs(np.linspace(0, 1, 1000) - jam_position))],
                    color="red", marker='^', label=f"New Jam Position: {jam_position:.3f}")

        plt.title("Road Jam Sampling with Event Visualization")
        plt.xlabel("Position (x)")
        plt.ylabel("Probability Density")

        # 为每个调度算法添加图例
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        plt.legend(unique_handles, unique_labels)

        plt.grid(True)
        plt.show()

    def road_jam_sampling(self, start, end):
        """
        从start到end的道路上的堵车采样，将新事件放入mine的event-pool
        :param start: 开始地点对象
        :param end: 结束地点对象
        :return: 无
        """
        # ------------------------
        # (0) 收集当前路段上的车辆
        # ------------------------
        trucks_on_road = self.truck_on_road(start, end)
        if len(trucks_on_road) == 0:
            return
        if self.env.now < 2:
            return

        distance = self.get_distance(trucks_on_road[0], end, enable_event=False)

        # 为每个同路的车辆根据当前时间计算 route_coverage 数值
        for truck in trucks_on_road:
            truck.journey_coverage = truck.get_route_coverage(distance)
        truck_positions = [truck.journey_coverage for truck in trucks_on_road]

        # ------------------------
        # (1) 先决定"要不要堵车"
        # ------------------------

        jam_lambda = 0.05  # 可自己调节大小(越大表示路上车辆对"触发堵车"的影响越显著)
        N = len(trucks_on_road)
        # 整体堵车概率: 车辆越多则越大
        jam_chance = 1.0 - math.exp(-jam_lambda * N)

        # 只做个输出提示；可保留可去
        if len(trucks_on_road) > 2:
            # print(f"Trucks on road from {start.name} to {end.name}: {len(trucks_on_road)}")
            pass
        elif len(trucks_on_road) == 1:
            # print(f"Trucks on road from {start.name} to {end.name}: {len(trucks_on_road)}")
            # 如果你仍想"单车就不堵"，可以直接 return
            return

        # 如果"这一次"没抽中堵车，直接退出
        if random.random() >= jam_chance:
            return

        # ------------------------
        # (2) 已判定要堵车 -> 再采样堵车位置
        # ------------------------
        sd = 0.1  # 每辆卡车生成的"影响分布"标准差

        x = np.linspace(0, 1, 1000)
        total_prob = np.zeros_like(x)

        # 将各车的正态分布叠加
        for position in truck_positions:
            prob = norm.pdf(x, position, sd)
            total_prob += prob

        # 这里的 total_prob 只是用来选"拥堵发生位置"，所以先做归一化
        total_prob_sum = np.sum(total_prob)
        if total_prob_sum <= 1e-12:
            return  # 极端情况下，直接不产生事件
        total_prob /= total_prob_sum

        # 再根据总分布选择具体位置
        try:
            jam_position_index = np.random.choice(np.arange(len(x)), p=total_prob)
        except ValueError:
            return
        jam_position = x[jam_position_index]

        # ------------------------
        # (3) 真正生成一个新的堵车事件
        # ------------------------
        # 使用 Weibull 分布采样堵车时间
        jam_time = np.random.weibull(2) * 10

        # 可视化(若你不需要画图，删掉下面这行即可)
        # self.plot_road_jam_events(jam_position, total_prob, self.env.now)

        # 放入事件池
        self.mine.random_event_pool.add_event(Event(
            self.env.now,
            "RoadEvent:jam",
            f"Time:<{self.env.now}> New RoadEvent:jam at road({jam_position:.3f}) "
            f"from {start.name} to {end.name} for {jam_time:.2f} minutes",
            info={
                "name": "RoadEvent:jam",
                "status": "jam",
                "speed": 0,
                "start_location": start.name,
                "end_location": end.name,
                "jam_position": jam_position,
                "start_time": self.env.now,
                "est_end_time": self.env.now + jam_time,
            },
        ))

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
        """
        获取从当前位置到目标位置的距离
        
        根据当前位置和目标位置的类型，从相应的距离矩阵中获取距离：
        - 从卸载点到装载点：使用d2l_road_matrix
        - 从装载点到卸载点：使用l2d_road_matrix
        - 从充电站到装载点：使用charging_to_load_road_matrix
        
        Args:
            truck (Truck): 卡车对象
            target_site: 目标位置
            enable_event (bool): 是否启用道路事件
            
        Returns:
            float: 路径距离
        """
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
            distance = self.d2l_road_matrix[load_site_id][dump_site_id]
        elif isinstance(current_site, LoadSite):
            load_site_id = self.mine.load_sites.index(current_site)
            dump_site_id = self.mine.dump_sites.index(target_site)
            distance = self.l2d_road_matrix[load_site_id][dump_site_id]
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
