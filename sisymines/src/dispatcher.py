from __future__ import annotations
import simpy


# 定义一个基类
class BaseDispatcher:
    def __init__(self):
        pass

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        # 需要由子类实现
        raise NotImplementedError("Subclass must implement this method.")

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        # 需要由子类实现
        raise NotImplementedError("Subclass must implement this method.")

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        # 需要由子类实现
        raise NotImplementedError("Subclass must implement this method.")


