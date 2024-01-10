from __future__ import annotations

from sisymines.src.dispatcher import BaseDispatcher



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

