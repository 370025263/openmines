# rl_dispatch.py
"""
负责RL算法和矿山环境之间的API交互
"""

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.truck import Truck
from openmines.src.mine import Mine
from multiprocessing import Queue, Lock

# 全局队列和锁，确保多个进程间的数据不会混淆
obs_queue = Queue()
act_queue = Queue()
queue_lock = Lock()

class RLDispatcher(BaseDispatcher):
    def __init__(self):
        """
        初始化RLDispatcher实例。
        """
        super().__init__()
        self.name = "RLDispatcher"
        self.current_observation = None
        self.done = False

    def give_init_order(self, truck: Truck, mine: Mine) -> int:
        """
        将当前状态传递给队列，并获取装载点索引
        :param truck: 当前的卡车对象
        :param mine: 当前的矿山对象
        :return: 装载点的索引
        """
        self.current_observation = self._get_observation(truck, mine)
        with queue_lock:
            obs_queue.put(self.current_observation)  # 将观察值放入队列
            action = act_queue.get()  # 从队列中获取动作
        return action

    def give_haul_order(self, truck: Truck, mine: Mine) -> int:
        """
        将当前状态传递给队列，并获取卸载点索引
        :param truck: 当前的卡车对象
        :param mine: 当前的矿山对象
        :return: 卸载点的索引
        """
        self.current_observation = self._get_observation(truck, mine)
        with queue_lock:
            obs_queue.put(self.current_observation)  # 将观察值放入队列
            action = act_queue.get()  # 从队列中获取动作
        return action

    def give_back_order(self, truck: Truck, mine: Mine) -> int:
        """
        将当前状态传递给队列，并获取返回装载点索引
        :param truck: 当前的卡车对象
        :param mine: 当前的矿山对象
        :return: 返回装载点的索引
        """
        self.current_observation = self._get_observation(truck, mine)
        with queue_lock:
            obs_queue.put(self.current_observation)  # 将观察值放入队列
            action = act_queue.get()  # 从队列中获取动作
        return action

    def _get_observation(self, truck: Truck, mine: Mine):
        """
        将当前的环境状态编码为一个可传递给队列的观察值
        :param truck: 当前的卡车对象
        :param mine: 当前的矿山对象
        :return: 当前的观察值（可根据需要调整）
        """
        # 这里可以根据实际需求返回一个合适的状态表示
        observation = {
            "truck_location": truck.current_location.name,
            "truck_load": truck.truck_load,
            "mine_status": mine.status,  # 这里只是一个示例，可以根据需要简化
        }
        return observation

    def reset(self):
        """
        重置环境状态
        """
        with queue_lock:
            while not obs_queue.empty():
                obs_queue.get()
            while not act_queue.empty():
                act_queue.get()
        self.current_observation = None
        self.done = False
