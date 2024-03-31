from __future__ import annotations
import numpy as np  # 导入NumPy库
import random,json,time
import openai


from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.dump_site import DumpSite, Dumper


class LLMDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "LLMDispatcher"
        self.OPENAI = OPENAI(model_name="gpt-3.5-turbo-0613")
        self.order_index = 0
        self.init_order_index = 0
        self.common_order_index = 0
        self.init_order_history = []
        self.haul_order_history = []
        self.back_order_history = []
        self.order_history = []
        self.np_random = np.random.RandomState()  # 创建NumPy的随机状态对象

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        # logger
        self.logger = mine.global_logger.get_logger("LLMDispatcher")
        cur_location = mine.charging_site.name
        # 统计loadsite信息
        load_sites = mine.load_sites
        loadsite_queue_length = [load_site.parking_lot.queue_status["total"][int(mine.env.now)] for load_site in load_sites]
        estimated_loadsite_queue_wait_times = [load_site.estimated_queue_wait_time for load_site in load_sites]

        # 获取dumpsite信息
        avaliable_dumpsites = [dumpsite for dumpsite in mine.dump_sites if dumpsite.parking_lot is not None]
        dump_site_names = [dumpsite.name for dumpsite in avaliable_dumpsites]
        dumpsite_queue_length = [dumpsite.parking_lot.queue_status["total"][int(mine.env.now)] for dumpsite in
                                 avaliable_dumpsites]
        estimated_dumpsite_queue_wait_times = [dumpsite.estimated_queue_wait_time for dumpsite in avaliable_dumpsites]

        # 获取过去的订单信息
        past_orders_all = self.order_history[-10:]
        past_orders_haul = [order for order in self.order_history if order["order_type"] == "haul_order"][-10:]
        # 获取Road距离信息
        road_matrix = mine.road.road_matrix

        prompt = f"""
                You are now an LLM (Large Language Model) scheduler, and your task is to assign an initial task destination for the current truck based on the available information.
                The current truck is in the charging area and needs to go to the loading area for loading.
                Background knowledge:
                    The mine has multiple loading and unloading areas as well as traffic roads. Mining trucks need to transport goods back and forth between these areas.
                    Loading areas have different loading capacities and queue situations, and unloading areas have different unloading capacities and queue situations.
                    The mining trucks are heterogeneous, varying in their running speeds and loading tonnage.
                    If a road has a large number of mining trucks dispatched, the probability of random events such as traffic jams and road repairs increases, leading to longer operation times for the trucks.

                Current mine information:
                        Loading areas: {[{"name": loadsite.name, "type": "loadsite", "load_capability(tons/min)": loadsite.load_site_productivity, "distance": mine.road.charging_to_load[i],
                                          "queue_length": loadsite_queue_length[i],
                                          "estimated_queue_wait_times": estimated_loadsite_queue_wait_times[i]
                                          } for i, loadsite in enumerate(mine.load_sites)]},

                        Current road information:
                            {[{"road_id": f"{cur_location} to {load_site.name}", "road_desc": f"from {cur_location} to {load_site.name}", "distance": mine.road.charging_to_load[j],
                               "trucks_on_this_road": mine.road.road_status[(cur_location, load_site.name)]["truck_count"],
                               "jammed_trucks_on_this_road": mine.road.road_status[(cur_location, load_site.name)]["truck_jam_count"],
                               "is_road_in_repair": mine.road.road_status[(cur_location, load_site.name)]["repair_count"]} for j, load_site in enumerate(mine.load_sites)]}


                Current order information:
                {{
                "cur_time": {mine.env.now},
                "order_type": "init_order",
                "truck_name": "{truck.name}",
                "truck_capacity": {truck.truck_capacity},
                "truck_speed": {truck.truck_speed}
                }}
                Historical scheduling decisions:
                {[{key: val for key, val in order.items() if key not in ['prompt', 'response']} for order in past_orders_all]}


                Please assign an initial task destination for the current truck based on the above information,
                Requirements:
                1. Overall objective: Considering the impact of random events on the road, choose the loading area with the shortest distance as much as possible, while avoiding traffic jams and road repairs.
                2. Finally, based on the overall objective, directly provide the following JSON string as the decision result:
                {{
                    "truck_name": "{truck.name}",
                    "loadingsite_index": an integer from 0 to {len(mine.load_sites) - 1}
                }}

                """

        for i in range(3):
            try:
                response = self.OPENAI.get_response(prompt=prompt)
                self.logger.info(f"LLM 订单{self.order_index +1 }：prompt:{prompt} \n {response}")
                start = response.find('{')
                end = response.rfind('}') + 1
                # 提取 JSON 字符串
                json_str = response[start:end]
                data = json.loads(json_str)
                loadsite_index = data["loadingsite_index"]
                break
            except Exception as e:
                print(e)
                loadsite_index = random.randint(0, len(mine.load_sites) - 1)
                self.logger.error(f"LLM 订单{self.order_index +1 }：parse error，giving random order")


        # logging
        order = {
            "cur_time": mine.env.now,
            "order_type": "init_order",
            "truck_name": truck.name,
            "truck_capacity": truck.truck_capacity,
            "truck_speed": truck.truck_speed,
            "loadingsite_index": loadsite_index,
            "prompt": prompt,
            "response": response}
        self.init_order_history.append(order)
        self.order_history.append(order)
        self.order_index += 1
        self.init_order_index += 1
        self.logger.debug(f"LLM INIT 订单{self.init_order_index}：{order}")
        return loadsite_index

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        # logger
        self.logger = mine.global_logger.get_logger("LLMDispatcher")
        # 获取当前卡车信息
        truck_load = truck.truck_load
        cur_location = truck.current_location.name
        cur_loadsite = mine.get_dest_obj_by_name(cur_location)
        cur_loadsite_index = mine.load_sites.index(cur_loadsite)
        assert isinstance(cur_loadsite, LoadSite), f"the truck {truck.name} is not in a loadsite, it is in {cur_loadsite.name}"
        # 统计loadsite信息
        load_sites = mine.load_sites
        loadsite_queue_length = [load_site.parking_lot.queue_status["total"][int(mine.env.now)] for load_site in load_sites]
        estimated_loadsite_queue_wait_times = [load_site.estimated_queue_wait_time for load_site in load_sites]

        # 获取dumpsite信息
        avaliable_dumpsites = [dumpsite for dumpsite in mine.dump_sites if dumpsite.parking_lot is not None]
        dump_site_names = [dumpsite.name for dumpsite in avaliable_dumpsites]
        dumpsite_queue_length = [dumpsite.parking_lot.queue_status["total"][int(mine.env.now)] for dumpsite in avaliable_dumpsites]
        estimated_dumpsite_queue_wait_times = [dumpsite.estimated_queue_wait_time for dumpsite in avaliable_dumpsites]

        # 获取过去的订单信息
        # past_orders_all = self.order_history[-10:]
        past_orders_haul = [order for order in self.order_history if order["order_type"] == "haul_order"][-10:]
        # 获取Road距离信息
        road_matrix = mine.road.road_matrix

        prompt = f"""
                You are now an LLM scheduler, and your task is to assign a target location for the current truck based on the available information.
                Background knowledge:
                The mine has multiple loading and unloading areas as well as traffic roads, and mining trucks need to transport goods back and forth between these areas.
                The loading capacity and queue situation of each loading area are different, as are the unloading capacity and queue situation of each unloading area.
                The mining trucks are heterogeneous, with differences in their running speeds and loading capacities.
                If a road has many mining trucks, the probability of random events such as traffic jams and road maintenance increases, leading to longer operation times for the trucks.

                Current mine information:
                Loading areas: {[{"name": loadsite.name, "type": "loadsite", "load_capability(tons/min)": loadsite.load_site_productivity, "queue_length": loadsite_queue_length[i]
                                  } for i, loadsite in enumerate(mine.load_sites)]},
                Unloading areas: {[{"name": dumpsite.name, "type": "dumpsite", "index": j, "distance": road_matrix[cur_loadsite_index][j],
                                    "queue_length": dumpsite_queue_length[j]
                                    } for j, dumpsite in enumerate(avaliable_dumpsites)]},
                Current road information:
                    {[{"road_id": f"{cur_location} to {dumpsite.name}", "road_desc": f"from {cur_location} to {dumpsite.name}", "distance": road_matrix[cur_loadsite_index][j],
                       "trucks_on_this_road": mine.road.road_status[(cur_location, dumpsite.name)]["truck_count"],
                       "jammed_trucks_on_this_road": mine.road.road_status[(cur_location, dumpsite.name)]["truck_jam_count"],
                       "is_road_in_repair": mine.road.road_status[(cur_location, dumpsite.name)]["repair_count"]} for j, dumpsite in enumerate(avaliable_dumpsites)]}

                Historical scheduling decisions:
                    {[{key: val for key, val in order.items() if key not in ['prompt', 'response']} for order in past_orders_haul]}

                The current truck is at the loading area {cur_location} and needs to go to the unloading area for unloading.
                Current truck order request information:
                {{
                "cur_time": {mine.env.now},
                "order_type": "haul_order",
                "truck_location": "{truck.current_location.name}",
                "truck_name": "{truck.name}",
                "truck_capacity": {truck.truck_capacity},
                "truck_speed": {truck.truck_speed}
                }}

                Please assign a suitable unloading area as the target location for the current truck based on the information above, allowing it to reach the target location as quickly as possible for unloading:
                Requirements:
                1. Overall objective: Considering the impact of random events on the road, choose the unloading area with the shortest distance as much as possible, while avoiding traffic jams and road repairs.
                2. Finally, based on the overall objective, directly provide the following JSON string as the decision result:
                {{
                    "truck_name": "{truck.name}",
                    "dumpsite_index": an integer from 0 to {len(avaliable_dumpsites) - 1}
                }}

                """

        for i in range(3):
            try:
                response = self.OPENAI.get_response(prompt)
                self.logger.info(f"LLM 订单{self.order_index + 1}：prompt:{prompt} \n {response}")
                start = response.find('{')
                end = response.rfind('}') + 1
                # 提取 JSON 字符串
                json_str = response[start:end]
                data = json.loads(json_str)
                dumpsite_index = data["dumpsite_index"]
                break
            except Exception as e:
                print(e)
                dumpsite_index = random.randint(0, len(avaliable_dumpsites) - 1)
                self.logger.error(f"LLM 订单{self.order_index + 1}：parse error，giving random order")

        # logging
        order = {
            "cur_time": mine.env.now,
            "order_type": "haul_order",
            "truck_name": truck.name,
            "truck_capacity": truck.truck_capacity,
            "truck_speed": truck.truck_speed,
            "truck_location": f"{truck.current_location.name}",
            "dumpsite_index": dumpsite_index,
            "prompt": prompt,
            "response": response}
        self.haul_order_history.append(order)
        self.order_history.append(order)
        self.order_index += 1
        self.logger.debug(f"LLM HAUL 订单{self.order_index}：{order}")
        return dumpsite_index


    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        # logger
        self.logger = mine.global_logger.get_logger("LLMDispatcher")
        # 获取当前卡车信息
        truck_load = truck.truck_load
        cur_location = truck.current_location.name
        cur_dumpsite = mine.get_dest_obj_by_name(cur_location)
        cur_dumpsite_index = mine.dump_sites.index(cur_dumpsite)
        assert isinstance(cur_dumpsite,
                          DumpSite), f"the truck {truck.name} is not in a dumpsite, it is in {cur_dumpsite.name}"

        # 统计dumpsite信息
        dump_sites = mine.dump_sites
        dumpsite_queue_length = [dump_site.parking_lot.queue_status["total"][int(mine.env.now)] for dump_site in
                                 dump_sites]
        estimated_dumpsite_queue_wait_times = [dump_site.estimated_queue_wait_time for dump_site in dump_sites]

        # 获取loadsite信息
        avaliable_loadsites = [loadsite for loadsite in mine.load_sites if loadsite.parking_lot is not None]
        load_site_names = [loadsite.name for loadsite in avaliable_loadsites]
        loadsite_queue_length = [loadsite.parking_lot.queue_status["total"][int(mine.env.now)] for loadsite in
                                 avaliable_loadsites]
        estimated_loadsite_queue_wait_times = [loadsite.estimated_queue_wait_time for loadsite in avaliable_loadsites]

        # 获取Road距离信息
        road_matrix = mine.road.road_matrix

        # 历史
        past_orders_back = [order for order in self.order_history if order["order_type"] == "back_order"][-10:]

        prompt = f"""
                    You are now an LLM scheduler, and your task is to assign the current truck a task to return to the loading area.
                    Background knowledge:
                    The mine has multiple loading and unloading areas as well as traffic roads. After being loaded with ore at a loading area, the mining truck needs to go to an unloading area to unload, and then return to the loading area to repeat the process.
                    Each unloading area has different unloading capabilities and queue situations, and each loading area also has different loading capabilities and queue situations.
                    The mining trucks are heterogeneous, with differences in their running speeds and loading capacities.
                    If a road has many mining trucks, the probability of random events like traffic jams and road maintenance increases, leading to longer operation times for the trucks.

                    Current mine information:
                    Unloading areas: {[{"name": dumpsite.name, "type": "dumpsite", "index": i, "queue_length": dumpsite_queue_length[i]
                                        } for i, dumpsite in enumerate(mine.dump_sites)]},
                    Loading areas: {[{"name": loadsite.name, "type": "loadsite", "index": j, "distance": road_matrix[j][cur_dumpsite_index],
                                      "queue_length": loadsite_queue_length[j]
                                      } for j, loadsite in enumerate(avaliable_loadsites)]},
                    Current road information:
                          {[{"road_id": f"{cur_location} to {loadsite.name}", "road_desc": f"from {cur_location} to {loadsite.name}", "distance": road_matrix[j][cur_dumpsite_index],
                             "trucks_on_this_road": mine.road.road_status[(cur_location, loadsite.name)]["truck_count"],
                             "jammed_trucks_on_this_road": mine.road.road_status[(cur_location, loadsite.name)]["truck_jam_count"],
                             "is_road_in_repair": mine.road.road_status[(cur_location, loadsite.name)]["repair_count"]}
                            for j, loadsite in enumerate(avaliable_loadsites)]}

                    Historical scheduling decisions:
                            {[{key: val for key, val in order.items() if key not in ['prompt', 'response']} for order in past_orders_back]}

                    The current truck is at the unloading area {cur_location} and needs to return to the loading area for loading.
                    Current truck order request information:
                    {{
                    "cur_time": {mine.env.now},
                    "order_type": "back_order",
                    "truck_location": "{truck.current_location.name}",
                    "truck_name": "{truck.name}",
                    "truck_capacity": {truck.truck_capacity},
                    "truck_speed": {truck.truck_speed}
                    }}

                    Please assign a suitable loading area as the target location for the current truck based on the information above, allowing it to return as quickly as possible for loading:
                    Requirements:
                        1. Overall objective: Considering the impact of random events on the road, choose the loading area with the shortest distance as much as possible, while avoiding traffic jams and road repairs.
                        2. Finally, based on the overall objective, directly provide the following JSON string as the decision result:
                    {{
                        "truck_name": "{truck.name}",
                        "loadsite_index": an integer number from 0 to {len(avaliable_loadsites) - 1}
                    }}
                """

        for i in range(3):
            try:
                response = self.OPENAI.get_response(prompt)
                self.logger.info(f"LLM 订单{self.order_index + 1}：prompt:{prompt} \n {response}")
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                data = json.loads(json_str)
                loadsite_index = data["loadsite_index"]
                break
            except Exception as e:
                print(e)
                loadsite_index = random.randint(0, len(avaliable_loadsites) - 1)
                self.logger.error(f"LLM 订单{self.order_index + 1}：parse error，giving random order")

        order = {
            "cur_time": mine.env.now,
            "order_type": "back_order",
            "truck_name": truck.name,
            "truck_capacity": truck.truck_capacity,
            "truck_speed": truck.truck_speed,
            "truck_location": f"{truck.current_location.name}",
            "loadsite_index": loadsite_index,
            "prompt": prompt,
            "response": response}
        self.back_order_history.append(order)
        self.order_history.append(order)
        self.order_index += 1
        self.logger.debug(f"LLM BACK 订单{self.order_index}：{order}")
        return loadsite_index


class OPENAI:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.api_key = "YOUR API KEY"  # token
        self.api_base = "OPENAI BASE HERE"  # you can choose custom api base, like:"https://api.qaqgpt.com/v1"
        self.model_name = model_name
        self.load_model()
    def load_model(self):
        openai.api_key = self.api_key
        openai.api_base = self.api_base

    def get_response(self, prompt):
        message =  [
                    {"role": "user", "content": prompt}]
        # time.sleep(1)
        response = openai.ChatCompletion.create(model=self.model_name, messages=message)
        for re in response["choices"]:
            return re["message"]["content"].strip()
        return ''


if __name__ == "__main__":
    dispatcher = LLMDispatcher()
    print(dispatcher.give_init_order(1,2))
    print(dispatcher.give_haul_order(1,2))
    print(dispatcher.give_back_order(1,2))

    print(dispatcher.total_order_count,dispatcher.init_order_count)