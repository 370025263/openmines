import json
import logging
import time, os
from pathlib import Path
import numpy as np

from sisymines.src.utils.event import EventPool
from sisymines.src.utils.logger import MineLogger


# 项目主路径
PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent
LOG_FILE_PATH = PROJECT_ROOT_PATH / "src" / "data" / "logs"

# 车辆状态
ON_CHARGING_SITE = -1  # 在充电站
ON_ROAD_TO_FIRST_LOAD = -2  # 从充电站到第一个装载点
UNHAULING = 0  # 从卸载区到装载区
WAIT_FOR_LOAD = 1  # 等待装载
LOADING = 2  # 装载中
HAULING = 3  # 从装载区到卸载区
WAIT_FOR_DUMP = 4  # 等待卸载
DUMPING = 5  # 卸载中
REPAIRING = 6  # 维修中
UNREPAIRABLE = 7  # 无法维修，返回充电场

class TickGenerator:
    """
    读取矿山下每个Truck的EventPool进行分析统计
    每一分钟相当于一个tick
    """

    def __init__(self, mine:"Mine", tick_num=60*8):
        self.mine = mine
        self.tick_num = tick_num
        self.ticks = dict()
        self.result_path = os.getcwd() + '/results'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)


    def run(self):
        """
        获取矿山下每个Truck的EventPool并进行总体分析
        1.获取需要dump的对象
        2.在每个tick中按照对象类别分别进行统计，形成字典格式
        :return:
        """
        # 读取mine的配置文件 获取 装载点(铲车、停车场)、卸载点（卸载位、停车场）、充电区 在图像中的位置
        load_sites = self.mine.load_sites
        dump_sites = self.mine.dump_sites
        charging_site = self.mine.charging_site
        trucks = self.mine.trucks

        # 遍历产生每个tick
        for tick in range(self.tick_num):
            # print(f"tick: {tick}")
            cur_time = tick
            truck_states = dict()
            load_site_states = dict()
            dump_site_states = dict()
            shovel_states = dict()
            dumper_states = dict()

            for truck in self.mine.trucks:
                event_pool = truck.event_pool
                # 判断卡车状态
                past_events = event_pool.get_event_by_time(cur_time+0.1)
                future_events = event_pool.get_event_by_time(cur_time+0.1, mode="future")
                status = None
                position = None
                truck_position = np.array([0.0, 0.0])
                ## 车辆在充电场
                if len(past_events)==0:
                    status = -1  # 没有事件、仅有一个调度事件==还在空载状态，位置处于充电站
                    truck_position = np.array(charging_site.position) + np.array([0.0, 0.0])
                ## 车辆在从充电场到装载点的路上
                if len(past_events)==2 and past_events[-1].event_type=="init":
                    """init example:
                        info={"name": self.name, "status": event_name, "speed": manual_speed if manual_speed is not None else self.truck_speed,
                                              "start_time": self.env.now, "est_end_time": self.env.now+duration, "end_time": None,
                                              "start_location": self.current_location.name,
                                              "target_location": target_location.name,
                                              "distance": distance, "duration": None}
                    """
                    status = -2  # 卡车已经得到了订单并且已经开始从充电站移动到第一个装载点

                    # 计算卡车位置
                    last_event = past_events[-1]
                    # 时间计算
                    cur_time_delta = cur_time - last_event.info["start_time"]
                    time_ratio = cur_time_delta / (last_event.info["est_end_time"] - last_event.info["start_time"])
                    # 方向计算
                    target_load_site = self.mine.get_dest_obj_by_name(last_event.info["target_location"])
                    direction = np.array(target_load_site.position) - np.array(charging_site.position)
                    truck_position = np.array(charging_site.position) + time_ratio * direction

                ## 车辆在装载点等待装载
                if past_events[-1].event_type=="wait shovel":
                    """wait shovel example:
                    info={"name": self.name, "status": "waiting for shovel",
                                            "start_time": self.env.now, "end_time": None,
                                            "shovel": shovel.name, "wait_duration": None}
                    """
                    status = 1
                    # 计算卡车位置
                    # 等待装载位置=装载区位置+停车场相对位置偏移=停车场位置
                    truck_cur_location_name = truck.current_location.name
                    parkinglot = self.mine.get_dest_obj_by_name(truck_cur_location_name).parking_lot
                    truck_position = np.array(parkinglot.position)  # 已经处理过

                ## 车辆在装载点装载
                if past_events[-1].event_type=="get shovel":
                    """get shovel example:
                    info={"name": self.name, "status": "loading on shovel",
                                                "start_time": self.env.now, "end_time": None,
                                                "shovel": shovel.name, "load_duration": None}
                    """
                    status = 2
                    # 计算卡车位置
                    # 装载位置=装载区位置+铲子相对位置偏移=铲子位置
                    shovel = self.mine.get_service_vehicle_by_name(past_events[-1].info["shovel"])
                    truck_position = np.array(shovel.position)  # 已经处理过

                ## 车辆满载 从装载点前往卸载点中
                if past_events[-1].event_type=="haul":
                    """haul example:
                    info={"name": self.name, "status": event_name, "speed": manual_speed if manual_speed is not None else self.truck_speed,
                                              "start_time": self.env.now, "est_end_time": self.env.now+duration, "end_time": None,
                                              "start_location": self.current_location.name,
                                              "target_location": target_location.name,
                                              "distance": distance, "duration": None}
                    """
                    status = 3

                    # 计算卡车位置
                    # 时间计算
                    last_event = past_events[-1]
                    cur_time_delta = cur_time - last_event.info["start_time"]
                    end_time = last_event.info["end_time"] if last_event.info["end_time"] is not None else last_event.info["est_end_time"]
                    time_ratio = cur_time_delta / (end_time - last_event.info["start_time"])
                    # 方向计算
                    cur_load_site = self.mine.get_dest_obj_by_name(last_event.info["start_location"])
                    target_unload_site = self.mine.get_dest_obj_by_name(last_event.info["target_location"])
                    direction = np.array(target_unload_site.position) - np.array(cur_load_site.position)
                    truck_position = np.array(cur_load_site.position) + time_ratio * direction

                ## 车辆在卸载点等待卸载
                if past_events[-1].event_type=="wait dumper":
                    """wait dumper example:
                    info={"name": self.name, "status": "waiting for dumper",
                                                    "start_time": self.env.now, "end_time": None,
                                                    "dumper": dumper.name, "wait_duration": None}
                    
                    """
                    status = 4
                    # 计算卡车位置
                    # 等待卸载位置=卸载区位置+停车场相对位置偏移=停车场位置
                    truck_cur_location_name = truck.current_location
                    parkinglot = self.mine.get_dest_obj_by_name(truck_cur_location_name).parking_lot
                    truck_position = np.array(parkinglot.position)  # 已经处理过

                ## 车辆在卸载点卸载
                if past_events[-1].event_type=="get dumper":
                    """get dumper example:
                    info={"name": self.name, "status": "unloading on dumper",
                                                    "start_time": self.env.now, "end_time": None,
                                                    "dumper": dumper.name, "unload_duration": None}
                    """
                    status = 5
                    # 计算卡车位置
                    # 卸载位置=卸载区位置+dumper相对位置偏移=dumper位置
                    dumper = self.mine.get_service_vehicle_by_name(past_events[-1].info["dumper"])
                    truck_position = np.array(dumper.position)  # 已经处理过

                ## 车辆空载 从卸载点前往装载点中
                if past_events[-1].event_type=="unhaul":
                    """unhaul example:
                    info={"name": self.name, "status": event_name, "speed": manual_speed if manual_speed is not None else self.truck_speed,
                                              "start_time": self.env.now, "est_end_time": self.env.now+duration, "end_time": None,
                                              "start_location": self.current_location.name,
                                              "target_location": target_location.name,
                                              "distance": distance, "duration": None}
                    """
                    status = 0
                    # 计算卡车位置
                    # 时间计算
                    last_event = past_events[-1]
                    cur_time_delta = cur_time - last_event.info["start_time"]
                    end_time = last_event.info["end_time"] if last_event.info["end_time"] is not None else last_event.info["est_end_time"]
                    time_ratio = cur_time_delta / (end_time - last_event.info["start_time"])
                    # 方向计算
                    cur_unload_site = self.mine.get_dest_obj_by_name(last_event.info["start_location"])
                    target_load_site = self.mine.get_dest_obj_by_name(last_event.info["target_location"])
                    direction = np.array(target_load_site.position) - np.array(cur_unload_site.position)
                    truck_position = np.array(cur_unload_site.position) + time_ratio * direction

                # 车辆损坏事件，则车辆在原地不动
                if past_events[-1].event_type == "breakdown":
                    status = 6
                    # 计算卡车位置
                    # 车辆损坏后位置=上一时间位置
                    truck_position = truck_position  # 已经处理过

                if past_events[-1].event_type == "unrepairable":
                    status = 7
                    # 计算卡车位置
                    # 车辆完全损坏后位置=充电区位置
                    # 获取chargingSite位置
                    truck_position = np.array(self.mine.charging_site.position).astype(float)  # 已经处理过

                truck_state = {
                    "name":truck.name,
                    "time":cur_time,
                    "state":status,
                    "position":list(truck_position if truck_position is not None else (0.5,0.5)),
                }
                truck_states[truck.name] = truck_state

            # 统计每一个loadsite的信息
            for loadsite in self.mine.load_sites:
                # 统计停车场的信息
                ## 获得停车场的排队数量
                total_queue_length = loadsite.parking_lot.queue_status["total"].get(cur_time,0)
                load_site_tick = {
                    "name":loadsite.name,
                    "tons":loadsite.tons,
                    "truck_service_count":loadsite.truck_service_count,
                    "time":cur_time,
                    "total_queue_length":total_queue_length,
                    "shovel_queue_length":{},
                    "position":list(loadsite.position)
                }

                # 统计shovel的信息
                for shovel in loadsite.shovel_list:
                    shovel_queue_length = loadsite.parking_lot.queue_status[shovel.name].get(cur_time,0)
                    load_site_tick["shovel_queue_length"][shovel.name] = shovel_queue_length
                    shovel_tick = {
                        "name":shovel.name,
                        "tons":shovel.tons,
                        "truck_service_count":shovel.truck_service_count,
                        "time":cur_time,
                        "queue_length":shovel_queue_length,
                        "position":(shovel.position),
                    }
                    shovel_states[shovel.name] = shovel_tick
                load_site_states[loadsite.name] = load_site_tick

            # 统计每一个dumpsite的信息
            for dumpsite in self.mine.dump_sites:
                # 统计停车场的信息
                ## 获得停车场的排队数量
                total_queue_length = dumpsite.parking_lot.queue_status["total"].get(cur_time,0)
                dump_site_tick = {
                    "name":dumpsite.name,
                    "tons":dumpsite.tons,
                    "truck_service_count":dumpsite.truck_service_count,
                    "time":cur_time,
                    "total_queue_length":total_queue_length,
                    "dumper_queue_length":{},
                    "position":list(dumpsite.position)
                }

                # 统计dumper的信息
                for dumper in dumpsite.dumper_list:
                    dumper_queue_length = dumpsite.parking_lot.queue_status[dumper.name].get(cur_time,0)
                    dump_site_tick["dumper_queue_length"][dumper.name] = dumper_queue_length
                    dumper_tick = {
                        "name":dumper.name,
                        "tons":dumper.tons,
                        "truck_service_count":dumper.truck_service_count,
                        "time":cur_time,
                        "queue_length":dumper_queue_length,
                        "position":list(dumper.position),
                    }
                    dumper_states[dumper.name] = dumper_tick
                dump_site_states[dumpsite.name] = dump_site_tick

            # 统计矿山整体信息
            mine_tick = {
                "time":cur_time,
                "total_tons":self.mine.total_tons,
                "waiting_truck_count":self.mine.waiting_truck_count,
            }


            tick_packet = {
                "tick":cur_time,
                "truck_states":truck_states,
                "load_site_states":load_site_states,
                "dump_site_states":dump_site_states,
                "shovel_states":shovel_states,
                "dumper_states":dumper_states,
                "mine_states":mine_tick
            }
            self.ticks[cur_time] = tick_packet

    def write_to_file(self, file_name):
        """
        将ticks写入文件
        :param file_path:
        :return:
        """
        file_path = os.path.join(self.result_path, file_name)
        with open(file_path, "w") as f:
            try:
                json.dump(self.ticks,f)
                print("file_name:{} write success".format(file_name))
            except Exception as e:
                print(e)
                print("file_name:{} write failed".format(file_name))