# rl_dispatch.py
"""
负责RL算法和矿山环境之间的API交互
"""
from openmines.src.charging_site import ChargingSite
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.dump_site import DumpSite
from openmines.src.load_site import LoadSite
from openmines.src.truck import Truck
from openmines.src.mine import Mine
from multiprocessing import Queue, Lock

# 全局队列和锁，确保多个进程间的数据不会混淆
class RLDispatcher(BaseDispatcher):
    def __init__(self):
        """
        初始化RLDispatcher实例。
        """
        super().__init__()
        self.name = "RLDispatcher"
        self.current_observation = None
        self.obs_queue = Queue()
        self.act_queue = Queue()
        self.done = False
        # reward stuff
        self.last_production = 0

    def _step(self, truck: Truck, mine: Mine) -> int:
        """完成队列交互"""
        self.current_observation = self._get_observation(truck, mine)
        info = self.current_observation["info"]
        reward = self._get_reward(mine)
        done = self._get_done(mine)
        trucated = False
        out = {
            "ob": self.current_observation,
            "info": info,
            "reward": reward,
            "truncated": trucated,
            "done": done
        }
        self.obs_queue.put(out, timeout=5)  # 将观察值放入队列
        action = self.act_queue.get(timeout=5)  # 从队列中获取动作
        return action

    def give_init_order(self, truck: Truck, mine: Mine) -> int:
        """
        将当前状态传递给队列，并获取装载点索引
        :param truck: 当前的卡车对象
        :param mine: 当前的矿山对象
        :return: 装载点的索引
        """
        action = self._step(truck,mine)
        return action

    def give_haul_order(self, truck: Truck, mine: Mine) -> int:
        """
        将当前状态传递给队列，并获取卸载点索引
        :param truck: 当前的卡车对象
        :param mine: 当前的矿山对象
        :return: 卸载点的索引
        """
        action = self._step(truck,mine)
        return action

    def give_back_order(self, truck: Truck, mine: Mine) -> int:
        """
        将当前状态传递给队列，并获取返回装载点索引
        :param truck: 当前的卡车对象
        :param mine: 当前的矿山对象
        :return: 返回装载点的索引
        """
        action = self._step(truck,mine)
        return action

    def _get_done(self, mine: Mine) -> bool:
        """done为true的时候被mine.py的start_rl触发"""
        return False

    def _get_reward(self, mine: Mine) -> int:
        """
        最大化产量方向的reward
        可以自定义配置
        :param mine:
        :return:
        """
        cur_production = mine.produce_tons
        reward = cur_production - self.last_production
        self.last_production = cur_production
        return reward

    def _get_observation(self, truck: Truck, mine: Mine):
        """
        将当前的环境状态编码为一个可传递给队列的观察值
        :param truck: 当前的卡车对象
        :param mine: 当前的矿山对象
        :return: 当前的观察值（可根据需要调整）
        """
        # 这里可以根据实际需求返回一个合适的状态表示
        # 当前车的状态
        the_truck_status = {
            "truck_location": truck.current_location.name,

            "truck_load": truck.truck_load,
            "truck_capacity": truck.truck_capacity,

            "truck_cycle_time": truck.truck_cycle_time,
            "truck_speed": truck.truck_speed,
        }
        # 目标地点状态
        if isinstance(truck.current_location, DumpSite) and isinstance(truck.target_location, LoadSite):
            event_name = "unhaul"
            target_status = {
                # stats and digits
                "queue_lengths": [load_site.parking_lot.queue_status["total"]["cur_value"] for load_site in mine.load_sites],
                "capacities": [load_site.load_site_productivity for load_site in mine.load_sites],
                "est_wait": [load_site.estimated_queue_wait_time for load_site in mine.load_sites],

                # summary info
                "produced_tons": [load_site.produced_tons for load_site in mine.load_sites],
                "service_counts": [load_site.service_count for load_site in mine.load_sites],
            }
        elif isinstance(truck.current_location, ChargingSite):
            event_name = "init"
            target_status = {
                # stats and digits
                "queue_lengths": [load_site.parking_lot.queue_status["total"]["cur_value"] for load_site in
                                  mine.load_sites],
                "capacities": [load_site.load_site_productivity for load_site in mine.load_sites],
                "est_wait": [load_site.estimated_queue_wait_time for load_site in mine.load_sites],

                # summary info
                "produced_tons": [load_site.produced_tons for load_site in mine.load_sites],
                "service_counts": [load_site.service_count for load_site in mine.load_sites],
            }
        else:
            event_name = "haul"
            target_status = {
                # stats and digits
                "queue_lengths": [dump_site.parking_lot.queue_status["total"]["cur_value"] for dump_site in mine.dump_sites],
                "capacities": [dump_site.dump_site_productivity for dump_site in mine.dump_sites],
                "est_wait": [dump_site.estimated_queue_wait_time for dump_site in mine.dump_sites],

                # summary info
                "produced_tons": [dump_site.produce_tons for dump_site in mine.dump_sites],
                "service_counts": [dump_site.service_count for dump_site in mine.dump_sites],
            }

        # 当前道路状态
        cur_road_status = {
            "charging2load": {"truck_count":dict(),"distances":dict(),"truck_jam_count":dict(),"repair_count":dict()},
            "load2dump": {"truck_count":dict(),"distances":dict(),"truck_jam_count":dict(),"repair_count":dict()},
            "dump2load": {"truck_count":dict(),"distances":dict(),"truck_jam_count":dict(),"repair_count":dict()},
        }
        ## 统计初始化init过程中的道路情况
        for i in range(mine.road.load_site_num):
            charging_site_name = mine.charging_site.name
            load_site_name = mine.load_sites[i].name
            cur_road_status["charging2load"]["truck_count"][i] = mine.road.road_status[(charging_site_name, load_site_name)][
                "truck_count"]
            cur_road_status["charging2load"]["truck_jam_count"][i] = mine.road.road_status[(charging_site_name, load_site_name)][
                "truck_jam_count"]
            cur_road_status["charging2load"]["repair_count"][i] = mine.road.road_status[(charging_site_name, load_site_name)][
                "repair_count"]
            cur_road_status["charging2load"]["distances"][i] = mine.road.charging_to_load[i]

        ## 统计haul过程中的道路情况
        for i in range(mine.road.load_site_num):
            for j in range(mine.road.dump_site_num):
                load_site_name = mine.load_sites[i].name
                dump_site_name = mine.dump_sites[j].name
                cur_road_status["load2dump"]["truck_count"][(i,j)] = mine.road.road_status[(load_site_name, dump_site_name)][
                    "truck_count"]
                cur_road_status["load2dump"]["truck_jam_count"][(i,j)] = mine.road.road_status[(load_site_name, dump_site_name)][
                    "truck_jam_count"]
                cur_road_status["load2dump"]["repair_count"][(i,j)] = mine.road.road_status[(load_site_name, dump_site_name)][
                    "repair_count"]
                cur_road_status["load2dump"]["distances"][(i,j)] = mine.road.road_matrix[i][j]

        ## 统计unhaul过程中的道路情况
        for j in range(mine.road.dump_site_num):
            for i in range(mine.road.load_site_num):
                load_site_name = mine.load_sites[i].name
                dump_site_name = mine.dump_sites[j].name
                cur_road_status["dump2load"]["truck_count"][(i,j)] = mine.road.road_status[(dump_site_name,load_site_name)][
                    "truck_count"]
                cur_road_status["dump2load"]["truck_jam_count"][(i,j)] = mine.road.road_status[(dump_site_name,load_site_name)][
                    "truck_jam_count"]
                cur_road_status["dump2load"]["repair_count"][(i,j)] = mine.road.road_status[(dump_site_name,load_site_name)][
                    "repair_count"]
                cur_road_status["dump2load"]["distances"][(i, j)] = mine.road.road_matrix[j][i]

        # 其他车状态
        # todo: 添加其他车状态

        # 矿山统计信息
        ## 当前可用的装载点、卸载点数量
        load_num = len(mine.load_sites)
        unload_num = len(mine.dump_sites)  # todo:装卸点的随机失效事件
        info = {"produce_tons": mine.produce_tons, "time": mine.env.now, "load_num":load_num, "unload_num":unload_num}

        # OBSERVATION
        observation = {
            "truck_name":truck.name,
            "event_name": event_name,# order type
            "info": info,# total tons, time
            "the_truck_status": the_truck_status,# the truck stats
            "target_status": target_status,# loading/unloading site stats
            "cur_road_status": cur_road_status,# road net status
            "mine_status": mine.status["cur"],  # 当前矿山KPI、事件总数等统计信息
        }
        return observation

