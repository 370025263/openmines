from __future__ import annotations
import os
import time
import numpy as np
import onnxruntime as ort
from typing import Optional
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.mine import Mine
from openmines.src.truck import Truck

# 导入 rl_dispatch.py 中的 preprocess_observation 函数
from openmines.src.dispatch_algorithms.rl_dispatch import RLDispatcher # 导入 preprocess_observation 和 RLDispatcher

class PPODispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "PPODispatcher"
        self.model_path = os.path.join(os.path.dirname(__file__), "checkpoints", "mine", "Mine-v1__ppo_single_dt__1__1738313304", "best_model.onnx")
        self.ort_session = self.load_onnx_model(self.model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name_action_probs = self.ort_session.get_outputs()[0].name # action_probs output name
        self.output_name_value = self.ort_session.get_outputs()[1].name # value output name
        self.obs_shape = 194 # 假设 observation shape 是 194，如果不同请调整
        self.rl_dispatcher_helper = RLDispatcher() # 创建 RLDispatcher 的实例，以便复用 _get_observation 方法
        self.max_sim_time = 240 #  **重要**: 需要设置 max_sim_time，与训练时一致。 从你的训练配置或者 `rl_env.py` 中获取。

    def load_onnx_model(self, onnx_path: str) -> ort.InferenceSession:
        """
        Load an ONNX model for inference.
        """
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")
        session = ort.InferenceSession(onnx_path)
        return session

    def give_init_order(self, truck: Truck, mine: Mine) -> int:
        """
        Given the current truck state and mine, choose an action (initial loading).
        """
        return self._dispatch_action(truck, mine)

    def give_haul_order(self, truck: Truck, mine: Mine) -> int:
        """
        Given the current truck state and mine, choose an action (hauling).
        """
        return self._dispatch_action(truck, mine)

    def give_back_order(self, truck: Truck, mine: Mine) -> int:
        """
        Given the current truck state and mine, choose an action (returning to charging or loading site).
        """
        return self._dispatch_action(truck, mine)

    def _dispatch_action(self, truck: Truck, mine: Mine) -> int:
        """
        Dispatch the truck to the next action based on ONNX model inference.
        """

        def preprocess_observation(observation, max_sim_time):
            """预处理原始观察，使其符合observation_space的格式"""

            """
            0.订单信息
            """
            # 1.订单类型,时间信息
            event_name = observation['event_name']
            if event_name == "init":
                event_type = [1, 0, 0]
                action_space_n = observation['info']['load_num']
            elif event_name == "haul":
                event_type = [0, 1, 0]
                action_space_n = observation['info']['unload_num']
            else:
                event_type = [0, 0, 1]
                action_space_n = observation['info']['load_num']
            # 2.当前订单时间绝对位置和相对位置
            time_delta = float(observation['info']['delta_time'])  # 距离上次调度的时间
            time_now = float(observation['info']['time']) / max_sim_time  # 当前时间(正则化）
            time_left = 1 - time_now  # 距离结束时间
            order_state = np.array([event_type[0], event_type[1], event_type[2], time_delta, time_now, time_left], dtype=np.float32) # dtype=np.float32

            """
            1.车辆自身信息
            """
            # 矿山总卡车数目（用于正则化）
            truck_num = observation['mine_status']['truck_count']
            # 4.车辆当前位置One-hot编码
            truck_location_onehot = np.array(observation["the_truck_status"]["truck_location_onehot"], dtype=np.float32) # dtype=np.float32
            # 车辆装载量，车辆循环时间（正则化）
            truck_features = np.array([
                np.log(observation['the_truck_status']['truck_load'] + 1),
                np.log(observation['the_truck_status']['truck_cycle_time'] + 1),
            ], dtype=np.float32) # dtype=np.float32
            truck_self_state = np.concatenate([truck_location_onehot, truck_features]).astype(np.float32) # .astype(np.float32)

            """
            2.道路相关信息
            """
            # 车预期行驶时间
            travel_time = np.array(observation['cur_road_status']['distances'], dtype=np.float32) * 60 / 25 # dtype=np.float32
            # 道路上卡车数量
            truck_counts = np.array(observation['cur_road_status']['truck_counts'], dtype=np.float32) / (truck_num + 1e-8) # dtype=np.float32
            # 道路距离信息
            road_dist = np.array(observation['cur_road_status']['oh_distances'], dtype=np.float32) # dtype=np.float32
            # 道路拥堵信息
            road_jam = np.array(observation['cur_road_status']['oh_truck_jam_count'], dtype=np.float32) # dtype=np.float32

            road_states = np.concatenate([travel_time, truck_counts, road_dist, road_jam]).astype(np.float32) # .astype(np.float32)
            """
            3.目标点相关信息
            """
            # 预期等待时间
            est_wait = np.log(observation['target_status']['single_est_wait'] + 1).astype(np.float32)  # 包含了路上汽车+队列汽车的目标装载点等待时间 # .astype(np.float32)
            tar_wait_time = np.log(np.array(observation['target_status']['est_wait'], dtype=np.float32) + 1).astype(np.float32)  # 不包含路上汽车 # .astype(np.float32)
            # 队列长度（正则化）
            queue_lens = np.array(observation['target_status']['queue_lengths'], dtype=np.float32) / (truck_num + 1e-8) # dtype=np.float32
            # 装载量
            tar_capa = np.log(np.array(observation['target_status']['capacities'], dtype=np.float32) + 1).astype(np.float32) # .astype(np.float32)
            # 各个目标点当前的产能系数(维护导致的产能下降）
            ability_ratio = np.array(observation['target_status']['service_ratio'], dtype=np.float32) # dtype=np.float32
            # 已经生产的矿石量（正则化）
            produced_tons = np.log(np.array(observation['target_status']['produced_tons'], dtype=np.float32) + 1).astype(np.float32) # .astype(np.float32)

            tar_state = np.concatenate([est_wait, tar_wait_time, queue_lens, tar_capa, ability_ratio, produced_tons]).astype(np.float32) # .astype(np.float32)

            state = np.concatenate([order_state, truck_self_state, road_states, tar_state]).astype(np.float32) # .astype(np.float32)
            assert not np.isnan(state).any(), f"NaN detected in state: {state}"
            assert not np.isnan(order_state).any(), f"NaN detected in order_state: {order_state}" # Debugging
            assert not np.isnan(truck_self_state).any(), f"NaN detected in truck_self_state: {truck_self_state}" # Debugging
            assert not np.isnan(road_states).any(), f"NaN detected in road_states: {road_states}" # Debugging
            assert not np.isnan(tar_state).any(), f"NaN detected in tar_state: {tar_state}" # Debugging

            return state

        current_observation_raw = self._get_raw_observation(truck, mine) # 获取原始观察
        processed_obs = preprocess_observation(current_observation_raw, self.max_sim_time).astype(np.float32).reshape(1, -1) # 使用 preprocess_observation 进行预处理, 并reshape and .astype(np.float32)

        # 使用 ONNX 模型预测动作
        ort_inputs = {self.input_name: processed_obs.astype(np.float32)} # .astype(np.float32)
        ort_outputs = self.ort_session.run([self.output_name_action_probs, self.output_name_value], ort_inputs)
        action_probs = ort_outputs[0] # action_probs 是第一个输出
        action = np.argmax(action_probs, axis=1)[0]  # 选择概率最高的动作

        return action

    def _get_raw_observation(self, truck: Truck, mine: Mine):
        """
        获取原始的、未经预处理的观察值，直接复用 RLDispatcher 中的 _get_observation 方法
        """
        return self.rl_dispatcher_helper._get_observation(truck, mine) # 直接调用 RLDispatcher 实例的 _get_observation

# Example usage (for testing - you'd integrate this into your simulation):
if __name__ == "__main__":
    # This is a placeholder for a Mine and Truck object - you need to create
    # actual instances of Mine and Truck as defined in your openmines simulation.
    class MockLocation:
        def __init__(self, name):
            self.name = name
    class MockTruck:
        def __init__(self, name="Truck1", current_location_name="charging_site", truck_load=0, truck_capacity=40, truck_speed=40):
            self.name = name
            self.current_location = MockLocation(current_location_name)
            self.truck_load = truck_load
            self.truck_capacity = truck_capacity
            self.truck_speed = truck_speed
            self.truck_cycle_time = 0

        def get_status(self):
            return {} # Placeholder

    class MockMine:
        def __init__(self):
            self.env = MockEnv()
            self.load_sites = [MockLocation("load_site_1"), MockLocation("load_site_2")]
            self.dump_sites = [MockLocation("dump_site_1"), MockLocation("dump_site_2")]

        def get_status(self):
            return {} # Placeholder
    class MockEnv:
        def __init__(self):
            self.now = 10.0


    dispatcher = PPODispatcher()
    mock_mine = MockMine()
    mock_truck = MockTruck()

    # Example of getting orders:
    init_order = dispatcher.give_init_order(mock_truck, mock_mine)
    haul_order = dispatcher.give_haul_order(mock_truck, mock_mine)
    back_order = dispatcher.give_back_order(mock_truck, mock_mine)

    print(f"Init Order: {init_order}")
    print(f"Haul Order: {haul_order}")
    print(f"Back Order: {back_order}")