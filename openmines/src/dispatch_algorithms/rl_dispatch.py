# rl_dispatch.py
"""
负责RL算法和矿山环境之间的API交互
"""
import time
from asyncore import dispatcher
from queue import Empty, Full
from multiprocessing import Queue, Lock

import numpy as np
from docutils.nodes import target

from openmines.src.charging_site import ChargingSite
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.dump_site import DumpSite
from openmines.src.load_site import LoadSite
from openmines.src.truck import Truck
from openmines.src.mine import Mine

# 删除所有具体dispatcher的import，改用动态导入
import importlib

# 全局队列和锁，确保多个进程间的数据不会混淆
class RLDispatcher(BaseDispatcher):
    def __init__(self, sug_dispatcher:str, reward_mode:str):
        """
        初始化RLDispatcher实例。
        """
        super().__init__()
        self.name = "RLDispatcher"
        self.sug_dispatcher = sug_dispatcher
        self.reward_mode = reward_mode
        # 动态导入dispatcher类
        try:
            def camel_to_snake(name):
                import re
                # 在大写字母前添加下划线，并转换为小写
                name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
                name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
                return name.lower()
            module = importlib.import_module("openmines.src.dispatch_algorithms." + camel_to_snake(sug_dispatcher))
            self.dispatcher_cls = getattr(module, sug_dispatcher)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Cannot import dispatcher class {sug_dispatcher}: {str(e)}")
            
        self.current_observation = None
        self.obs_queue = Queue()
        self.act_queue = Queue()
        self.done = False
        # reward stuff
        self.last_production = 0
        self.last_time = 0
        self.last_reward = 0
        self.last_order_time = 0
        self.last_truck = None
        self.last_action = None

    def _step(self, truck: Truck, mine: Mine) -> int:
        """完成队列交互"""
        start_time = time.time()
        timeout = 30  # 设置总超时时间为10秒
        try:
            # 1: 获取观察值
            self.current_observation = self._get_observation(truck, mine)
            info = self.current_observation["info"]
            # reward = self.last_reward  # 读取上一次的reward并进行计算
            if self.reward_mode == "dense":
                reward = self._get_reward_dense(mine=mine, truck=self.last_truck, action=self.last_action)  # 预估当前决策的奖励，下一步作为reward返回
            elif self.reward_mode == "sparse":
                reward = self._get_reward_sparse(mine=mine, truck=self.last_truck, action=self.last_action)
            else:
                raise ValueError(f"Unknown reward mode: {self.reward_mode}")

            done = self._get_done(mine)
            truncated = False
            # suggest action
            info["sug_action"] = self._get_dispatch_action(truck, mine)  # 根据当前观察获取建议动作
            out = {
                "ob": self.current_observation,
                "info": info,
                "reward": reward,
                "truncated": truncated,
                "done": done
            }

            # 2: 将观察值放入Agent
            try:
                mine.mine_logger.debug(f"PUTTING {truck.name} at {mine.env.now}")
                self.obs_queue.put(out, timeout=5)  # 将观察值放入队列

            except Full:
                mine.mine_logger.error("Observation queue is full. Possible deadlock.")
                raise TimeoutError("Observation queue is full")

            # 3: 获取Agent动作
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                raise TimeoutError("Timeout occurred before getting action")

            action = self.act_queue.get()  # 从队列中获取动作.这里不设置超时。
            mine.mine_logger.debug(f"RECEIVED order of {truck.name} at {mine.env.now}, action is {action}")  # debug
            # self.last_reward = self._get_reward(mine=mine, truck=truck, action=action, dense=True)  # 预估当前决策的奖励，下一步作为reward返回
            self.update_order_info(truck, mine, action)  # 记录当前订单信息
            self.last_order_time = mine.env.now
            # 断点 4: 成功获取Agent动作之后
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
                mine.mine_logger.debug(
                    f"Step execution time: {end_time - start_time:.2f} seconds. obs-lock: {self.obs_queue._rlock} act-lock: {self.act_queue._rlock}. Env time: {mine.env.now} Done: {mine.done} Truckname:{truck.name} Location: {truck.current_location.name} OBS_queue_len:{self.obs_queue}, act_queue_len: {self.act_queue}")

    def update_order_info(self, truck: Truck, mine: Mine, action: int):
        """
        更新订单信息
        :param truck: 当前的卡车对象
        :param mine: 当前的矿山对象
        :param action: 当前的动作
        """
        self.last_truck = truck
        self.last_action = action

    def give_init_order(self, truck: Truck, mine: Mine) -> int:
        """
        将当前状态传递给队列，并获取装载点索引
        :param truck: 当前的卡车对象
        :param mine: 当前的矿山对象
        :return: 装载点的索引
        """
        action = self._step(truck, mine)
        return action

    def give_haul_order(self, truck: Truck, mine: Mine) -> int:
        """
        将当前状态传递给队列，并获取卸载点索引
        :param truck: 当前的卡车对象
        :param mine: 当前的矿山对象
        :return: 卸载点的索引
        """
        action = self._step(truck, mine)
        return action

    def give_back_order(self, truck: Truck, mine: Mine) -> int:
        """
        将当前状态传递给队列，并获取返回装载点索引
        :param truck: 当前的卡车对象
        :param mine: 当前的矿山对象
        :return: 返回装载点的索引
        """
        action = self._step(truck, mine)
        return action

    def _get_done(self, mine: Mine) -> bool:
        """done为true的时候被mine.py的start_rl触发"""
        return False

    def _get_dispatch_action(self, truck: Truck, mine: Mine) -> int:
        """
        使用指定的调度器获取建议动作
        """
        # 实例化指定的调度器
        dispatch = self.dispatcher_cls()
        
        # 根据当前位置调用对应的决策方法
        if isinstance(truck.current_location, DumpSite):
            decision = dispatch.give_back_order(truck, mine)
        elif isinstance(truck.current_location, ChargingSite):
            decision = dispatch.give_init_order(truck, mine)
        else:
            # LoadSite
            decision = dispatch.give_haul_order(truck, mine)
        return decision

    def _get_reward_sparse(self, mine: Mine, truck: "Truck" = None, action: int = None) -> int:
        """
        稀疏奖励，仅仅在最后一步返回总吨数作为reward
        """
        if action is None and truck is None and mine.env.now != 0: 
            return mine.produce_tons
        return 0

    def _get_reward_dense(self, mine: Mine, truck: "Truck" = None, action: int = None) -> int:
        """
        最大化产量方向的reward
        可以自定义配置
        这部分我需要重新设计
        惩罚：
            车队的总排队时间
            车辆的总卸载时间
            系统堵车用时
            车辆路上的运行时间
            （铲车维护导致车辆等待，这一部分其实包含在了排队时间中，这里不再添加）
        奖励：
            车辆的中间产出log(delta + 1) 而不是直接输出高方差
            最终车辆产出
        """
        # 必要信息
        action = action
        # order_type = "init" if isinstance(truck.current_location, ChargingSite) else "haul" if isinstance(truck.current_location, LoadSite) else "unhaul"
        reward = 0
        # trip_reward = 0
        final_tons_reward = 0
        delta_tons_reward = 0

        # 惩罚项目
        # 1. 车队的总排队时间
        # est_wait_times_in_queue = np.array([dump_site.estimated_queue_wait_time for dump_site in mine.dump_sites]) if order_type == "unhaul" else np.array([load_site.estimated_queue_wait_time for load_site in mine.load_sites])
        # st_wait_time_in_queue = est_wait_times_in_queue[action]
        last_order_time, cur_time = self.last_order_time, mine.env.now
        wait_truck_event_all = [event for t in mine.trucks for event in t.event_pool.get_even_by_type("wait shovel")]
        wait_duration = 0
        for wait_event in wait_truck_event_all:
            wait_start_time = wait_event.info["start_time"]
            wait_end_time = wait_event.info.get("end_time") or cur_time
            if wait_end_time is None:
                wait_end_time = cur_time
            if max(wait_start_time, last_order_time) < min(wait_end_time, cur_time):
                # 统计wait shovel事件 last_order_time~mine.env.now中的持续时间
                wait_duration += min(wait_end_time, cur_time) - max(wait_start_time, last_order_time)

        # 2. 车辆的装卸载时间(按照平均铲车生产力计算)
        # service_time_cost = 1 if order_type == "unhaul" else np.array([truck.truck_capacity/(load_site.load_site_productivity/len(load_site.shovel_list)) for load_site in mine.load_sites])
        get_shovel_event_all = [event for t in mine.trucks for event in t.event_pool.get_even_by_type("get shovel")]
        shovel_duration = 0
        for shovel_event in get_shovel_event_all:
            shovel_start_time = shovel_event.info["start_time"]
            shovel_end_time = shovel_event.info.get("end_time") or cur_time
            if max(shovel_start_time, last_order_time) < min(shovel_end_time, cur_time):
                # 统计get shovel事件 last_order_time~mine.env.now中的持续时间
                shovel_duration += min(shovel_end_time, cur_time) - max(shovel_start_time, last_order_time)
        get_dumper_event_all = [event for t in mine.trucks for event in t.event_pool.get_even_by_type("get dumper")]
        dumper_duration = 0
        for dumper_event in get_dumper_event_all:
            dumper_start_time = dumper_event.info["start_time"]
            dumper_end_time = dumper_event.info.get("end_time") or cur_time
            if max(dumper_start_time, last_order_time) < min(dumper_end_time, cur_time):
                # 统计get dumper事件 last_order_time~mine.env.now中的持续时间
                dumper_duration += min(dumper_end_time, cur_time) - max(dumper_start_time, last_order_time)
        service_duration = shovel_duration + dumper_duration

        # 3. 堵车事件数量（堵车事件没有办法预测，只是对当前状态的惩罚）（每个单位时间对正在进行中的堵车进行惩罚）
        truck_jams_all = mine.random_event_pool.get_even_by_type("RoadEvent:jam")
        jam_duration = 0
        for jam_event in truck_jams_all:
            jam_start_time = jam_event.info["start_time"]
            est_end_time = jam_event.info["est_end_time"] or cur_time
            if max(jam_start_time, last_order_time) < min(est_end_time, cur_time):
                # 统计jam事件在last_order_time~mine.env.now中的持续时间
                jam_duration += min(est_end_time, cur_time) - max(jam_start_time, last_order_time)

        # 4. 车辆路上的运行时间
        truck_move_all = [event for t in mine.trucks for event in
                          t.event_pool.get_even_by_type("init") + t.event_pool.get_even_by_type(
                              "haul") + t.event_pool.get_even_by_type("unhaul")]
        move_duration = 0
        for move_event in truck_move_all:
            move_start_time = move_event.info["start_time"]
            move_end_time = move_event.info.get("end_time") or cur_time
            if max(move_start_time, last_order_time) < min(move_end_time, cur_time):
                # 统计move事件在last_order_time~mine.env.now中的持续时间
                move_duration += min(move_end_time, cur_time) - max(move_start_time, last_order_time)

        if action is None and truck is None and mine.env.now != 0:
            # THIS IS LAST STEP, return total reward
            baseline_production = 10000
            final_tons = (self.last_production - baseline_production)
            if final_tons > 0:
                final_tons_reward = 0.5 * final_tons
            # match_factor = mine.match_factor
            # print("FIN-MatchFactor:", match_factor)
            # reward += (match_factor-0.5)*1000

        # 5. 车辆的中间产出log(delta + 1) 而不是直接输出高方差
        cur_production = mine.produce_tons
        delta_tons = (cur_production - self.last_production)
        self.last_production = cur_production

        # (2) 负向惩罚项：让系数更小
        # 例如改成 -0.01 或 -0.005
        wait_penalty = -0.1 * wait_duration
        service_penalty = -0.05 * service_duration
        jam_penalty = -0.05 * jam_duration
        move_penalty = -0.01 * move_duration

        delta_tons_reward = 5.0 * np.log(delta_tons + 1)

        reward = wait_penalty + service_penalty + jam_penalty + move_penalty \
                 + delta_tons_reward + final_tons_reward
        # 基于delta_tons和delta_time进行奖励，保留最终tons奖励。
        # 基于delta_tons和delta_time计算奖励率
        delta_time = cur_time - last_order_time
        if delta_time > 0:
            tons_per_min = delta_tons / (delta_time + 1e-6)
            reward += 0.1 * tons_per_min  # 每分钟产量的奖励系数为0.1

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
            "truck_location_onehot": truck.get_location_onehot(),

            "truck_load": truck.truck_load,
            "truck_capacity": truck.truck_capacity,

            "truck_cycle_time": truck.truck_cycle_time,
            "truck_speed": truck.truck_speed,
        }

        # 目标地点状态
        if isinstance(truck.current_location, DumpSite):  # and isinstance(truck.target_location, LoadSite):
            event_name = "unhaul"
            trucks_on_roads = [mine.road.truck_on_road(start=truck.current_location, end=mine.load_sites[i]) for i in
                               range(len(mine.load_sites))]
            load_time_on_road_trucks = [
                sum([each_truck.truck_capacity for each_truck in each_road]) / mine.load_sites[i].load_site_productivity
                for i, each_road in enumerate(trucks_on_roads)]  # 路上的卡车需要的装载用时
            sig_wait_est = np.array([load_site.estimated_queue_wait_time for load_site in mine.load_sites]) + np.array(
                load_time_on_road_trucks)
        elif isinstance(truck.current_location, ChargingSite):
            event_name = "init"
            trucks_on_roads = [mine.road.truck_on_road(start=mine.charging_site, end=mine.load_sites[i]) for i in
                               range(len(mine.load_sites))]
            load_time_on_road_trucks = [
                sum([each_truck.truck_capacity for each_truck in each_road]) / mine.load_sites[i].load_site_productivity
                for i, each_road in enumerate(trucks_on_roads)]  # 路上的卡车需要的装载用时
            sig_wait_est = np.array([load_site.estimated_queue_wait_time for load_site in mine.load_sites]) + np.array(
                load_time_on_road_trucks)
        else:
            event_name = "haul"
            trucks_on_roads = [mine.road.truck_on_road(start=truck.current_location, end=mine.dump_sites[i]) for i in
                               range(len(mine.dump_sites))]
            dump_time_on_road_trucks = [(sum([1 for each_truck in each_road]) / len(mine.dump_sites[i].dumper_list)) *
                                        mine.dump_sites[i].dumper_list[0].dump_time for i, each_road in
                                        enumerate(trucks_on_roads)]  # 路上的卡车需要的装载用时
            sig_wait_est = np.array([dump_site.estimated_queue_wait_time for dump_site in mine.dump_sites]) + np.array(
                dump_time_on_road_trucks)
        target_status = {
            # stats and digits
            "queue_lengths": [load_site.parking_lot.queue_status["total"]["cur_value"] for load_site in
                              mine.load_sites] + [dump_site.parking_lot.queue_status["total"]["cur_value"] for dump_site
                                                  in mine.dump_sites],
            "capacities": [load_site.load_site_productivity for load_site in mine.load_sites] + [
                dump_site.dump_site_productivity for dump_site in mine.dump_sites],
            "est_wait": [load_site.estimated_queue_wait_time for load_site in mine.load_sites] + [
                dump_site.estimated_queue_wait_time for dump_site in mine.dump_sites],
            "single_est_wait": sig_wait_est,
            "service_ratio": [load_site.service_ability_ratio for load_site in mine.load_sites] + [
                dump_site.service_ability_ratio for dump_site in mine.dump_sites],
            # summary info
            "produced_tons": [load_site.produced_tons for load_site in mine.load_sites] + [dump_site.produce_tons for
                                                                                           dump_site in
                                                                                           mine.dump_sites],
            "service_counts": [load_site.service_count for load_site in mine.load_sites] + [dump_site.service_count for
                                                                                            dump_site in
                                                                                            mine.dump_sites],
        }

        # 当前道路状态
        cur_road_status = {
            "charging2load": {"truck_count": dict(), "distances": dict(), "truck_jam_count": dict(),
                              "repair_count": dict()},
            "load2dump": {"truck_count": dict(), "distances": dict(), "truck_jam_count": dict(),
                          "repair_count": dict()},
            "dump2load": {"truck_count": dict(), "distances": dict(), "truck_jam_count": dict(),
                          "repair_count": dict()},
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
            cur_road_status["charging2load"]["truck_count"][i] = \
            mine.road.road_status[(charging_site_name, load_site_name)][
                "truck_count"]
            assert cur_road_status["charging2load"]["truck_count"][i] == len(
                mine.road.truck_on_road(start=mine.charging_site, end=mine.load_sites[
                    i])), f"Truck count is not match: {cur_road_status['charging2load']['truck_count'][i]} vs {mine.road.truck_on_road(start=mine.charging_site, end=mine.load_sites[i])}"
            cur_road_status["charging2load"]["truck_jam_count"][i] = \
            mine.road.road_status[(charging_site_name, load_site_name)][
                "truck_jam_count"]
            cur_road_status["charging2load"]["repair_count"][i] = \
            mine.road.road_status[(charging_site_name, load_site_name)][
                "repair_count"]
            cur_road_status["charging2load"]["distances"][i] = mine.road.charging_to_load[i]

        ## 统计haul过程中的道路情况
        for i in range(mine.road.load_site_num):
            for j in range(mine.road.dump_site_num):
                load_site_name = mine.load_sites[i].name
                dump_site_name = mine.dump_sites[j].name
                cur_road_status["load2dump"]["truck_count"][(i, j)] = \
                mine.road.road_status[(load_site_name, dump_site_name)][
                    "truck_count"]
                cur_road_status["load2dump"]["truck_jam_count"][(i, j)] = \
                mine.road.road_status[(load_site_name, dump_site_name)][
                    "truck_jam_count"]
                cur_road_status["load2dump"]["repair_count"][(i, j)] = \
                mine.road.road_status[(load_site_name, dump_site_name)][
                    "repair_count"]
                cur_road_status["load2dump"]["distances"][(i, j)] = mine.road.l2d_road_matrix[i][j]

        ## 统计unhaul过程中的道路情况
        for j in range(mine.road.dump_site_num):
            for i in range(mine.road.load_site_num):
                load_site_name = mine.load_sites[i].name
                dump_site_name = mine.dump_sites[j].name
                cur_road_status["dump2load"]["truck_count"][(i, j)] = \
                mine.road.road_status[(dump_site_name, load_site_name)][
                    "truck_count"]
                cur_road_status["dump2load"]["truck_jam_count"][(i, j)] = \
                mine.road.road_status[(dump_site_name, load_site_name)][
                    "truck_jam_count"]
                cur_road_status["dump2load"]["repair_count"][(i, j)] = \
                mine.road.road_status[(dump_site_name, load_site_name)][
                    "repair_count"]
                cur_road_status["dump2load"]["distances"][(i, j)] = mine.road.d2l_road_matrix[i][j]
        # One-hot encoding
        for road_id in range(mine.road.load_site_num):
            cur_road_status["oh_truck_count"].append(cur_road_status["charging2load"]["truck_count"][road_id])
            cur_road_status["oh_truck_jam_count"].append(cur_road_status["charging2load"]["truck_jam_count"][road_id])
            cur_road_status["oh_repair_count"].append(cur_road_status["charging2load"]["repair_count"][road_id])
            cur_road_status["oh_distances"].append(cur_road_status["charging2load"]["distances"][road_id])

        for i in range(mine.road.load_site_num):
            for j in range(mine.road.dump_site_num):
                cur_road_status["oh_truck_count"].append(cur_road_status["load2dump"]["truck_count"][(i, j)])
                cur_road_status["oh_truck_jam_count"].append(cur_road_status["load2dump"]["truck_jam_count"][(i, j)])
                cur_road_status["oh_repair_count"].append(cur_road_status["load2dump"]["repair_count"][(i, j)])
                cur_road_status["oh_distances"].append(cur_road_status["load2dump"]["distances"][(i, j)])

        for j in range(mine.road.dump_site_num):
            for i in range(mine.road.load_site_num):
                cur_road_status["oh_truck_count"].append(cur_road_status["dump2load"]["truck_count"][(i, j)])
                cur_road_status["oh_truck_jam_count"].append(cur_road_status["dump2load"]["truck_jam_count"][(i, j)])
                cur_road_status["oh_repair_count"].append(cur_road_status["dump2load"]["repair_count"][(i, j)])
                cur_road_status["oh_distances"].append(cur_road_status["dump2load"]["distances"][(i, j)])
        # ob from current location
        if event_name == "init":
            for i in range(mine.road.load_site_num):
                cur_road_status["truck_counts"].append(cur_road_status["charging2load"]["truck_count"][i])
                cur_road_status["truck_jam_counts"].append(cur_road_status["charging2load"]["truck_jam_count"][i])
                cur_road_status["distances"].append(cur_road_status["charging2load"]["distances"][i])
        elif event_name == "haul":
            location_idx = the_truck_status["truck_location_onehot"].index(1) - 1
            assert str(
                location_idx + 1) in truck.current_location.name, f"Truck {truck.name} at {truck.current_location.name} location index is not match: {location_idx}"
            for j in range(mine.road.dump_site_num):
                cur_road_status["truck_counts"].append(cur_road_status["load2dump"]["truck_count"][(location_idx, j)])
                cur_road_status["truck_jam_counts"].append(
                    cur_road_status["load2dump"]["truck_jam_count"][(location_idx, j)])
                cur_road_status["distances"].append(cur_road_status["load2dump"]["distances"][(location_idx, j)])
        elif event_name == "unhaul":
            location_idx = the_truck_status["truck_location_onehot"].index(1) - 1 - mine.road.load_site_num
            for i in range(mine.road.load_site_num):
                cur_road_status["truck_counts"].append(cur_road_status["dump2load"]["truck_count"][(i, location_idx)])
                cur_road_status["truck_jam_counts"].append(
                    cur_road_status["dump2load"]["truck_jam_count"][(i, location_idx)])
                cur_road_status["distances"].append(cur_road_status["dump2load"]["distances"][(i, location_idx)])
        else:
            raise Exception(f"Truck {truck.name} at {truck.current_location.name} order is unkown: {event_name}")

        # 其他车状态
        # todo: 添加其他车状态

        # 矿山统计信息
        ## 当前可用的装载点、卸载点数量
        load_num = len(mine.load_sites)
        unload_num = len(mine.dump_sites)  # todo:装卸点的随机失效事件
        info = {"produce_tons": mine.produce_tons, "time": mine.env.now, "delta_time": mine.env.now - self.last_time,
                "load_num": load_num, "unload_num": unload_num}
        self.last_time = mine.env.now
        # OBSERVATION
        observation = {
            "truck_name": truck.name,
            "event_name": event_name,  # order type
            "info": info,  # total tons, time
            "the_truck_status": the_truck_status,  # the truck stats
            "target_status": target_status,  # loading/unloading site stats
            "cur_road_status": cur_road_status,  # road net status
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