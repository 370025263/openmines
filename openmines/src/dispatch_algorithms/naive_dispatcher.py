from __future__ import annotations

import time
from openmines.src.dispatcher import BaseDispatcher


class NaiveDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "NaiveDispatcher"

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        # 从第一个load site开始
        return 0

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        # 从第一个load site开始
        return 0

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        # 从第一个dump site开始
        return 0

if __name__ == "__main__":
    dispatcher = NaiveDispatcher()
    print(dispatcher.give_init_order(1,2))
    print(dispatcher.give_init_order(1, 2))
    print(dispatcher.give_init_order(1, 2))
    print(dispatcher.give_init_order(1, 2))
    print(dispatcher.give_haul_order(1,2))
    print(dispatcher.give_back_order(1,2))

    print(dispatcher.total_order_count,dispatcher.init_order_count,dispatcher.init_order_time,dispatcher.total_order_time)