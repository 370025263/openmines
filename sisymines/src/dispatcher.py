from __future__ import annotations
import simpy
import time
from functools import wraps

# 定义一个用于跟踪调用次数和执行时间的装饰器
def track_calls_and_time(method_type):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = method(self, *args, **kwargs)
            elapsed_time = time.time() - start_time

            if method_type == "init":
                self.init_order_count += 1
                self.init_order_time += elapsed_time
            elif method_type == "haul":
                self.haul_order_count += 1
                self.haul_order_time += elapsed_time
            elif method_type == "back":
                self.back_order_count += 1
                self.back_order_time += elapsed_time
            self.total_order_count = self.init_order_count + self.haul_order_count + self.back_order_count
            self.total_order_time = self.init_order_time + self.haul_order_time + self.back_order_time

            return result
        return wrapper
    return decorator

# 定义基类
class BaseDispatcher:
    def __init__(self):
        self.init_order_count = 0
        self.haul_order_count = 0
        self.back_order_count = 0
        self.init_order_time = 0
        self.haul_order_time = 0
        self.back_order_time = 0
        self.total_order_count = 0
        self.total_order_time = 0

    @track_calls_and_time("init")
    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        raise NotImplementedError("Subclass must implement this method.")

    @track_calls_and_time("haul")
    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        raise NotImplementedError("Subclass must implement this method.")

    @track_calls_and_time("back")
    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        raise NotImplementedError("Subclass must implement this method.")
