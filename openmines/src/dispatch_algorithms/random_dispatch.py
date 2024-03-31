from __future__ import annotations
import numpy as np  # 导入NumPy库

from openmines.src.dispatcher import BaseDispatcher

class RandomDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "RandomDispatcher"
        self.np_random = np.random.RandomState()  # 创建NumPy的随机状态对象

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        # 获取可用的loadsite
        avaliable_loadsites = [loadsite for loadsite in mine.load_sites if loadsite.parking_lot is not None]
        # 随机一个数字
        random_index = self.np_random.randint(0, len(avaliable_loadsites))
        return random_index

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        # 获取可用的dumpsite
        avaliable_dumpsites = [loadsite for loadsite in mine.load_sites if loadsite.parking_lot is not None]
        random_index = self.np_random.randint(0, len(avaliable_dumpsites))
        return random_index

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        # 获取可用的loadsite
        avaliable_loadsites = [loadsite for loadsite in mine.dump_sites if loadsite.parking_lot is not None]
        # 随机一个数字
        random_index = self.np_random.randint(0, len(avaliable_loadsites))
        return random_index

if __name__ == "__main__":
    dispatcher = RandomDispatcher()
    print(dispatcher.give_init_order(1,2))
    print(dispatcher.give_haul_order(1,2))
    print(dispatcher.give_back_order(1,2))

    print(dispatcher.total_order_count,dispatcher.init_order_count)