# rl_dispatch.py
"""
负责RL算法和矿山环境之间的API交互
"""
import time
from asyncore import dispatcher
from queue import Empty, Full
from multiprocessing import Queue, Lock

from docutils.nodes import target

from openmines.src.charging_site import ChargingSite
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.dump_site import DumpSite
from openmines.src.load_site import LoadSite
from openmines.src.truck import Truck
from openmines.src.mine import Mine
from openmines.src.dispatch_algorithms.fixed_group_dispatch import FixedGroupDispatcher



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
        self.last_time = 0
        self.last_reward = 0

    def _step(self, truck: Truck, mine: Mine) -> int:
        """完成队列交互"""
        start_time = time.time()
        timeout = 30  # 设置总超时时间为10秒
        try:
            # 断点 1: 获取观察值之前
            self.current_observation = self._get_observation(truck, mine)
            info = self.current_observation["info"]
            reward = self.last_reward
            done = self._get_done(mine)
            truncated = False
            # suggest action
            info["sug_action"] = self._get_dispatch_action(truck, mine)
            out = {
                "ob": self.current_observation,
                "info": info,
                "reward": reward,
                "truncated": truncated,
                "done": done
            }

            # 断点 2: 将观察值放入队列之前
            try:
                mine.mine_logger.debug(f"PUTTING {truck.name} at {mine.env.now}")
                self.obs_queue.put(out, timeout=5)  # 将观察值放入队列

            except Full:
                mine.mine_logger.error("Observation queue is full. Possible deadlock.")
                raise TimeoutError("Observation queue is full")

            # 断点 3: 从队列中获取动作之前
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                raise TimeoutError("Timeout occurred before getting action")

            action = self.act_queue.get()  # 从队列中获取动作.这里不设置超时。
            mine.mine_logger.debug(f"RECEIVED order of {truck.name} at {mine.env.now}, action is {action}")  # debug
            self.last_reward = self._get_reward(mine=mine, truck=truck, action=action, dense=True)

            # 断点 4: 成功获取动作之后
            return action

        except TimeoutError as e:
            mine.mine_logger.error(f"Timeout error in _step: {str(e)}")
            # 这里可以添加一些恢复逻辑，比如返回一个默认动作
            return 0  # 或者其他默认动作

        except Exception as e:
            import traceback
            mine.mine_logger.error(f"Unexpected error in _step: {str(e)}")
            traceback.print_exc()
            raise

        finally:
            end_time = time.time()
            if end_time - start_time > 1:
                mine.mine_logger.debug(f"Step execution time: {end_time - start_time:.2f} seconds. obs-lock: {self.obs_queue._rlock} act-lock: {self.act_queue._rlock}. Env time: {mine.env.now} Done: {mine.done} Truckname:{truck.name} Location: {truck.current_location.name} OBS_queue_len:{self.obs_queue.qsize()}, act_queue_len: {self.act_queue.qsize()}")

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

    def _get_dispatch_action(self, truck: Truck, mine: Mine) -> int:
        """
        """
        # use group dispatch reward
        dispatch = FixedGroupDispatcher()
        if isinstance(truck.current_location, DumpSite):
            decision = dispatch.give_back_order(truck, mine)
        elif isinstance(truck.current_location, ChargingSite):
            decision = dispatch.give_init_order(truck, mine)
        else:
            # LoadSite
            decision = dispatch.give_haul_order(truck, mine)
        return decision

    def _get_reward(self, mine: Mine, truck:"Truck"=None, action:int=None, dense:bool=False) -> int:
        """
        最大化产量方向的reward
        可以自定义配置
        RewardShaping
        :param mine:
        :return:
        """

        # # funcs
        # productivity_reward_func = lambda x : -8*x**2/3 + 14*x/3 - 1  # (0,-1) (0.25,0) (1,1)
        # distance_reward_func = lambda x : -15*x**2/4 + 7*x/4 + 1  # (0,1) (0.8,0) (1,-1)
        # traffic_reward_func = lambda x : 1/4 - 25*x**2/64 if x < 0.8 else -25*x**2 + 40*x - 16
        # # weight
        # static_weight = [0.33,0.33,0.33]
        # if action is None and truck is None:
        #     # THIS IS LAST STEP, return total reward
        #     baseline_production = 15000
        #     reward:int = (self.last_production - baseline_production)/1000
        #     match_factor = mine.match_factor
        #     # if match_factor is inf
        #     if match_factor == float("inf"):
        #         raise Exception("Match factor is inf at end")
        #     match_factor_r = match_factor if match_factor > 1 else match_factor - 1
        #     # print("MatchFactor:",match_factor)
        #     reward += match_factor_r*10
        #     return reward
        # # INIT PHASE REWARD
        # if isinstance(truck.current_location, ChargingSite):
        #     load_site_productivities = normalize_list_inplace([load_site.load_site_productivity for load_site in mine.load_sites])
        #     load_site_distances = normalize_list_inplace([mine.road.charging_to_load[i] for i in range(mine.road.load_site_num)])
        #     load_site_traffic = normalize_list_inplace([mine.road.road_status[(mine.charging_site.name, load_site.name)]["truck_count"] for load_site in mine.load_sites])
        #     reward = static_weight[0]*productivity_reward_func(load_site_productivities[action]) + static_weight[1]*distance_reward_func(load_site_distances[action]) + static_weight[2]*traffic_reward_func(load_site_traffic[action])
        #
        # # HAUL PHASE REWARD
        # elif isinstance(truck.current_location, LoadSite):
        #     dump_site_productivities = normalize_list_inplace([dump_site.dump_site_productivity for dump_site in mine.dump_sites])
        #     dump_site_distances = normalize_list_inplace([mine.road.road_matrix[action][i] for i in range(mine.road.dump_site_num)])
        #     dump_site_traffic = normalize_list_inplace([mine.road.road_status[(truck.current_location.name, dump_site.name)]["truck_count"] for dump_site in mine.dump_sites])
        #     reward = static_weight[0]*productivity_reward_func(dump_site_productivities[action]) + static_weight[1]*distance_reward_func(dump_site_distances[action]) + static_weight[2]*traffic_reward_func(dump_site_traffic[action])
        # # UNHAUL PHASE REWARD
        # elif isinstance(truck.current_location, DumpSite):
        #     load_site_productivities = normalize_list_inplace([load_site.load_site_productivity for load_site in mine.load_sites])
        #     load_site_distances = normalize_list_inplace([mine.road.road_matrix[i][action] for i in range(mine.road.load_site_num)])
        #     load_site_traffic = normalize_list_inplace([mine.road.road_status[(truck.current_location.name,load_site.name)]["truck_count"] for load_site in mine.load_sites])
        #     reward = static_weight[0]*productivity_reward_func(load_site_productivities[action]) + static_weight[1]*distance_reward_func(load_site_distances[action]) + static_weight[2]*traffic_reward_func(load_site_traffic[action])
        # else:
        #     raise Exception(f"Truck {truck.name} at {truck.current_location.name} order is unkown: {action}")
        #
        # # reward += 1
        # # POTENTIAL REWARD SHAPING
        # cur_production = mine.produce_tons
        # # reward = cur_production - self.last_production
        # self.last_production = cur_production
        # if dense:
        #     # use group dispatch reward
        #     dispatch = FixedGroupDispatcher()
        #     if isinstance(truck.current_location, DumpSite):
        #         decision = dispatch.give_back_order(truck, mine)
        #     elif isinstance(truck.current_location, ChargingSite):
        #         decision = dispatch.give_init_order(truck, mine)
        #     else:
        #         # LoadSite
        #         decision = dispatch.give_haul_order(truck, mine)
        #     # if action == decision, give bonus
        #     # reward = reward + 0.1 if action == decision else reward - 0.1  # just supervise
        # return reward
        return 1

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
            "truck_location_onehot": truck.get_location_onehot(),

            "truck_load": truck.truck_load,
            "truck_capacity": truck.truck_capacity,

            "truck_cycle_time": truck.truck_cycle_time,
            "truck_speed": truck.truck_speed,
        }

        # 目标地点状态
        if isinstance(truck.current_location, DumpSite):  # and isinstance(truck.target_location, LoadSite):
            event_name = "unhaul"
            sig_wait_est =  [load_site.estimated_queue_wait_time for load_site in mine.load_sites]
        elif isinstance(truck.current_location, ChargingSite):
            event_name = "init"
            sig_wait_est = [load_site.estimated_queue_wait_time for load_site in mine.load_sites]
        else:
            event_name = "haul"
            sig_wait_est = [dump_site.estimated_queue_wait_time for dump_site in mine.dump_sites]
        target_status = {
            # stats and digits
            "queue_lengths": [load_site.parking_lot.queue_status["total"]["cur_value"] for load_site in
                              mine.load_sites] + [dump_site.parking_lot.queue_status["total"]["cur_value"] for dump_site in mine.dump_sites],
            "capacities": [load_site.load_site_productivity for load_site in mine.load_sites]+[dump_site.dump_site_productivity for dump_site in mine.dump_sites],
            "est_wait": [load_site.estimated_queue_wait_time for load_site in mine.load_sites]+[dump_site.estimated_queue_wait_time for dump_site in mine.dump_sites],
            "single_est_wait": sig_wait_est,
            # summary info
            "produced_tons": [load_site.produced_tons for load_site in mine.load_sites] + [dump_site.produce_tons for dump_site in mine.dump_sites],
            "service_counts": [load_site.service_count for load_site in mine.load_sites] + [dump_site.service_count for dump_site in mine.dump_sites],
        }

        # 当前道路状态
        cur_road_status = {
            "charging2load": {"truck_count":dict(),"distances":dict(),"truck_jam_count":dict(),"repair_count":dict()},
            "load2dump": {"truck_count":dict(),"distances":dict(),"truck_jam_count":dict(),"repair_count":dict()},
            "dump2load": {"truck_count":dict(),"distances":dict(),"truck_jam_count":dict(),"repair_count":dict()},
            "oh_truck_count": [],
            "oh_distances": [],
            "oh_truck_jam_count": [],
            "oh_repair_count": [],
            # ob from current location
            "truck_counts": [],  # road from cur 2 targets
            "distances": [],
            "truck_jam_counts": [],
        }
        ## 统计初始化init过程global_step中的道路情况
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
        # One-hot encoding
        for road_id in range(mine.road.load_site_num):
            cur_road_status["oh_truck_count"].append(cur_road_status["charging2load"]["truck_count"][road_id])
            cur_road_status["oh_truck_jam_count"].append(cur_road_status["charging2load"]["truck_jam_count"][road_id])
            cur_road_status["oh_repair_count"].append(cur_road_status["charging2load"]["repair_count"][road_id])
            cur_road_status["oh_distances"].append(cur_road_status["charging2load"]["distances"][road_id])

        for i in range(mine.road.load_site_num):
            for j in range(mine.road.dump_site_num):
                cur_road_status["oh_truck_count"].append(cur_road_status["load2dump"]["truck_count"][(i,j)])
                cur_road_status["oh_truck_jam_count"].append(cur_road_status["load2dump"]["truck_jam_count"][(i,j)])
                cur_road_status["oh_repair_count"].append(cur_road_status["load2dump"]["repair_count"][(i,j)])
                cur_road_status["oh_distances"].append(cur_road_status["load2dump"]["distances"][(i,j)])

        for j in range(mine.road.dump_site_num):
            for i in range(mine.road.load_site_num):
                cur_road_status["oh_truck_count"].append(cur_road_status["dump2load"]["truck_count"][(i,j)])
                cur_road_status["oh_truck_jam_count"].append(cur_road_status["dump2load"]["truck_jam_count"][(i,j)])
                cur_road_status["oh_repair_count"].append(cur_road_status["dump2load"]["repair_count"][(i,j)])
                cur_road_status["oh_distances"].append(cur_road_status["dump2load"]["distances"][(i,j)])
        # ob from current location
        if event_name == "init":
            for i in range(mine.road.load_site_num):
                cur_road_status["truck_counts"].append(cur_road_status["charging2load"]["truck_count"][i])
                cur_road_status["truck_jam_counts"].append(cur_road_status["charging2load"]["truck_jam_count"][i])
                cur_road_status["distances"].append(cur_road_status["charging2load"]["distances"][i])
        elif event_name == "haul":
            location_idx = the_truck_status["truck_location_onehot"].index(1) - 1
            assert str(location_idx+1) in truck.current_location.name, f"Truck {truck.name} at {truck.current_location.name} location index is not match: {location_idx}"
            for j in range(mine.road.dump_site_num):
                cur_road_status["truck_counts"].append(cur_road_status["load2dump"]["truck_count"][(location_idx,j)])
                cur_road_status["truck_jam_counts"].append(cur_road_status["load2dump"]["truck_jam_count"][(location_idx,j)])
                cur_road_status["distances"].append(cur_road_status["load2dump"]["distances"][(location_idx,j)])
        elif event_name == "unhaul":
            location_idx = the_truck_status["truck_location_onehot"].index(1) - 1 - mine.road.load_site_num
            for i in range(mine.road.load_site_num):
                cur_road_status["truck_counts"].append(cur_road_status["dump2load"]["truck_count"][(i,location_idx)])
                cur_road_status["truck_jam_counts"].append(cur_road_status["dump2load"]["truck_jam_count"][(i,location_idx)])
                cur_road_status["distances"].append(cur_road_status["dump2load"]["distances"][(i,location_idx)])
        else:
            raise Exception(f"Truck {truck.name} at {truck.current_location.name} order is unkown: {event_name}")

        # 其他车状态
        # todo: 添加其他车状态

        # 矿山统计信息
        ## 当前可用的装载点、卸载点数量
        load_num = len(mine.load_sites)
        unload_num = len(mine.dump_sites)  # todo:装卸点的随机失效事件
        info = {"produce_tons": mine.produce_tons, "time": mine.env.now, "delta_time": mine.env.now - self.last_time,"load_num":load_num, "unload_num":unload_num}
        self.last_time = mine.env.now
        # OBSERVATION
        observation = {
            "truck_name":truck.name,
            "event_name": event_name,# order type
            "info": info, # total tons, time
            "the_truck_status": the_truck_status,# the truck stats
            "target_status": target_status,# loading/unloading site stats
            "cur_road_status": cur_road_status,# road net status
            "mine_status": mine.status["cur"],  # 当前矿山KPI、事件总数等统计信息
        }
        return observation

def normalize_list_inplace(lst):
    """
    Normalize a list inplace
    :param lst: list to normalize
    :return: None
    """
    max_val = max(lst)
    min_val = min(lst)
    for i in range(len(lst)):
        lst[i] = (lst[i] - min_val) / (max_val - min_val + 1e-6)
    return lst