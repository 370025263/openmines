import numpy as np

def preprocess_observation(observation, max_sim_time):
    """特征预处理逻辑"""
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
    order_state = np.array([event_type[0], event_type[1], event_type[2], time_delta, time_now, time_left])

    """
    1.车辆自身信息
    """
    # 矿山总卡车数目（用于正则化）
    truck_num = observation['mine_status']['truck_count']
    # 4.车辆当前位置One-hot编码
    truck_location_onehot = np.array(observation["the_truck_status"]["truck_location_onehot"])
    # 车辆装载量，车辆循环时间（正则化）
    truck_features = np.array([
        np.log(observation['the_truck_status']['truck_load'] + 1),
        np.log(observation['the_truck_status']['truck_cycle_time'] + 1),
    ])
    truck_self_state = np.concatenate([truck_location_onehot, truck_features])

    """
    2.道路相关信息
    """
    # 车预期行驶时间
    travel_time = np.array(observation['cur_road_status']['distances']) * 60 / 25
    # 道路上卡车数量
    truck_counts = np.array(observation['cur_road_status']['truck_counts']) / (truck_num + 1e-8)
    # 道路距离信息
    road_dist = np.array(observation['cur_road_status']['oh_distances'])
    # 道路拥堵信息
    road_jam = np.array(observation['cur_road_status']['oh_truck_jam_count'])

    road_states = np.concatenate([travel_time, truck_counts, road_dist, road_jam])

    """
    3.目标点相关信息
    """
    # 预期等待时间
    est_wait = np.log(observation['target_status']['single_est_wait'] + 1)  # 包含了路上汽车+队列汽车的目标装载点等待时间
    tar_wait_time = np.log(np.array(observation['target_status']['est_wait']) + 1)  # 不包含路上汽车
    # 队列长度（正则化）
    queue_lens = np.array(observation['target_status']['queue_lengths']) / (truck_num + 1e-8)
    # 装载量
    tar_capa = np.log(np.array(observation['target_status']['capacities']) + 1)
    # 各个目标点当前的产能系数(维护导致的产能下降）
    ability_ratio = np.array(observation['target_status']['service_ratio'])
    # 已经生产的矿石量（正则化）
    produced_tons = np.log(np.array(observation['target_status']['produced_tons']) + 1)

    tar_state = np.concatenate([est_wait, tar_wait_time, queue_lens, tar_capa, ability_ratio, produced_tons])

    state = np.concatenate([order_state, truck_self_state, road_states, tar_state])
    assert not np.isnan(state).any(), f"NaN detected in state: {state}"
    assert not np.isnan(time_delta), f"NaN detected in time_delta: {time_delta}"
    assert not np.isnan(time_now), f"NaN detected in time_now: {time_now}"

    return state.astype(np.float32) 