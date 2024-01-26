from __future__ import annotations
import numpy as np  # 导入NumPy库
import random,json,time
import openai


from sisymines.src.dispatcher import BaseDispatcher
from sisymines.src.load_site import LoadSite, Shovel
from sisymines.src.dump_site import DumpSite, Dumper


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
        你现在是一个LLM调度器，你需要根据已有的信息为当前卡车分配一个初始任务的目标地点。
        当前卡车在充电区，需要前往装载区进行装载。
        背景知识：
            矿山中有多个装载区和卸载区还有交通道路，矿卡需要在两地之间往返运货。
            装载区的装载能力和排队情况各自不同，卸载区的卸载能力和排队情况也各自不同。
            矿卡是异构的，其运行速度和装载吨数上存在区别。
            如果某条道路上派出了较多的矿卡，那么随机事件如堵车、道路维修等发生的概率会增大，会导致矿卡的运行时间变长。



        当前矿山信息：
                装载区：{ [{
                    "name": loadsite.name,
                    "type": "loadsite",
                    "load_capability(tons/min)": loadsite.load_site_productivity,
                    "distance": mine.road.charging_to_load[i],
                    "queue_length": loadsite_queue_length[i],
                    "estimated_queue_wait_times":estimated_loadsite_queue_wait_times[i]
                        }  for i,loadsite in enumerate(mine.load_sites)]},

                当前道路信息：
                    { [  {
                    "road_id": f"{cur_location} to {load_site.name}",
                    "road_desc": f"from {cur_location} to {load_site.name}",
                    "distance": mine.road.charging_to_load[j],
                    "trucks_on_this_road": mine.road.road_status[(cur_location,load_site.name)]["truck_count"],
                    "jammed_trucks_on_this_road": mine.road.road_status[(cur_location,load_site.name)]["truck_jam_count"],
                    "is_road_in_repair": mine.road.road_status[(cur_location,load_site.name)]["repair_count"] } for j,load_site in enumerate(mine.load_sites)]}


        当前订单信息：
        {{
        "cur_time": {mine.env.now},
        "order_type": "init_order",
        "truck_name": "{truck.name}",
        "truck_capacity": {truck.truck_capacity},
        "truck_speed": {truck.truck_speed}
        }}
        历史调度决策：
        {[{key: val for key, val in order.items() if key not in ['prompt', 'response']} for order in past_orders_all]}


        请根据以上信息为当前卡车分配一个初始任务的目标地点,
        要求：
        1.总体目标：考虑到道路上随机事件的影响，尽可能选择距离短的装载区，同时避免堵车、道路维修等情况。
        2.首先对从当前地点前往装载区的道路的通行状况进行一步步的总结, 要参照历史决策。
        3.然后根据总结的结果，进一步对装载区的预期用时、拥挤程度进行主观估计。
        2.最终，根据上方总结，参考总体目标，给出如下的json字符串作为决定结果：
        {{
            "truck_name": "{truck.name}",
            "loadingsite_index": a integer from 0 to {len(mine.load_sites) - 1}
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
                你现在是一个LLM调度器，你需要根据已有的信息为当前卡车分配目标地点。
                背景知识：
                矿山中有多个装载区和卸载区还有交通道路，矿卡需要在两地之间往返运货。
                装载区的装载能力和排队情况各自不同，卸载区的卸载能力和排队情况也各自不同。
                矿卡是异构的，其运行速度和装载吨数上存在区别。
                如果某条道路上存在较多的矿卡，那么随机事件如堵车、道路维修等发生的概率会增大，导致矿卡的运行时间变长。
                
                当前矿山信息：
                装载区：{ [{
                    "name": loadsite.name,
                    "type": "loadsite",
                    "load_capability(tons/min)": loadsite.load_site_productivity,
                    "queue_length": loadsite_queue_length[i]
                        }  for i,loadsite in enumerate(mine.load_sites)]},
                卸载区：{ [{
                    "name": dumpsite.name,
                    "type": "dumpsite",
                    "index": j,
                    "distance": road_matrix[cur_loadsite_index][j],
                    "queue_length": dumpsite_queue_length[j]
                        }  for j,dumpsite in enumerate(avaliable_dumpsites)]},
                当前道路信息：
                    { [  {
                    "road_id": f"{cur_location} to {dumpsite.name}",
                    "road_desc": f"from {cur_location} to {dumpsite.name}",
                    "distance": road_matrix[cur_loadsite_index][j],
                    "trucks_on_this_road": mine.road.road_status[(cur_location,dumpsite.name)]["truck_count"],
                    "jammed_trucks_on_this_road": mine.road.road_status[(cur_location,dumpsite.name)]["truck_jam_count"],
                    "is_road_in_repair": mine.road.road_status[(cur_location,dumpsite.name)]["repair_count"] } for j,dumpsite in enumerate(avaliable_dumpsites)]}
                
                历史调度决策：
                    {[{key: val for key, val in order.items() if key not in ['prompt', 'response']} for order in past_orders_haul]}
        
                当前卡车在装载区{cur_location}，需要前往卸载区进行卸载。
                当前卡车请求订单的信息：
                {{
                "cur_time": {mine.env.now},
                "order_type": "haul_order",
                "truck_location": "{truck.current_location.name}",
                "truck_name": "{truck.name}",
                "truck_capacity": {truck.truck_capacity},
                "truck_speed": {truck.truck_speed}
                }}

                请根据以上信息为当前卡车分配一个合适的卸载区作为目标地点,让其尽快到达目标地点进行卸载：
                要求：
                1.总体目标：考虑到道路上随机事件的影响，尽可能选择距离短的卸载区，同时避免堵车、道路维修等情况。
                2.首先对从当前地点前往卸载区的道路的通行状况进行一步步的总结, 要参照上方的历史调度决策。
                3.然后根据总结的结果，进一步对每个卸载区的预期用时、拥挤程度进行主观估计。
                2.最终，根据上方总结，参考总体目标，给出如下的json字符串作为决定结果：
                {{
                    "truck_name": "{truck.name}",
                    "dumpsite_index": a integer from 0 to  {len(avaliable_dumpsites) - 1}
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
            你现在是一个LLM调度器，你需要根据已有的信息为当前卡车分配返回装载区的任务。
            背景知识：
            矿山中有多个装载区和卸载区还有交通道路，矿卡在装载区装满矿石后，需要前往卸载区卸货，然后返回装载区重复这个过程。
            卸载区的卸载能力和排队情况各自不同，装载区的装载能力和排队情况也各自不同。
            矿卡是异构的，其运行速度和装载吨数上存在区别。
            如果某条道路上存在较多的矿卡，那么随机事件如堵车、道路维修等发生的概率会增大，导致矿卡的运行时间变长。

            当前矿山信息：
            卸载区：{[{
                        "name": dumpsite.name,
                        "type": "dumpsite",
                        "index": i,
                        "queue_length": dumpsite_queue_length[i]
                            } for i, dumpsite in enumerate(mine.dump_sites)]},
            装载区：{[{
                        "name": loadsite.name,
                        "type": "loadsite",
                        "index": j,
                        "distance": road_matrix[j][cur_dumpsite_index],
                        "queue_length": loadsite_queue_length[j]
                                } for j, loadsite in enumerate(avaliable_loadsites)]},
            当前道路信息：
                  {[{
                        "road_id": f"{cur_location} to {loadsite.name}",
                        "road_desc": f"from {cur_location} to {loadsite.name}",
                        "distance": road_matrix[j][cur_dumpsite_index],
                        "trucks_on_this_road": mine.road.road_status[(cur_location, loadsite.name)]["truck_count"],
                        "jammed_trucks_on_this_road": mine.road.road_status[(cur_location, loadsite.name)]["truck_jam_count"],
                        "is_road_in_repair": mine.road.road_status[(cur_location, loadsite.name)]["repair_count"]} 
                            for j, loadsite in enumerate(avaliable_loadsites)]}

            历史调度决策：
                    {[{key: val for key, val in order.items() if key not in ['prompt', 'response']} for order in past_orders_back]}
        
            当前卡车在卸载区{cur_location}，需要返回装载区进行装载。
            当前卡车请求订单的信息：
            {{
            "cur_time": {mine.env.now},
            "order_type": "back_order",
            "truck_location": "{truck.current_location.name}",
            "truck_name": "{truck.name}",
            "truck_capacity": {truck.truck_capacity},
            "truck_speed": {truck.truck_speed}
            }}

            请根据以上信息为当前卡车分配一个合适的装载区作为目标地点,让其尽快返回进行装载：
            要求：
                1.总体目标：考虑到道路上随机事件的影响，尽可能选择距离短的装载区，同时避免堵车、道路维修等情况。
                2.首先对从当前地点前往装载区的道路的通行状况进行一步步的总结, 要参照上方的历史调度决策。
                3.然后根据总结的结果，进一步对每个装载区的预期用时、拥挤程度进行主观估计。
                2.最终，根据上方总结，参考总体目标，给出如下的json字符串作为决定结果：
            {{
                "truck_name": "{truck.name}",
                "loadsite_index": a int number from 0 to  {len(avaliable_loadsites) - 1}
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
    def __init__(self, model_name="gpt-3.5-turbo-0613"):
        self.api_key = "sk-c6aQsx5gHenXWaztBa55E9D5D76b43818206A5Ea1f91B204"
        self.api_base = "https://api.qaqgpt.com/v1"
        self.model_name = model_name
        self.load_model()
    def load_model(self):
        openai.api_key = self.api_key
        openai.api_base = self.api_base

    def get_response(self, prompt):
        message =  [
                    {"role": "user", "content": prompt}]
        time.sleep(3)
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