from __future__ import annotations
import time

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

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if name in ["give_init_order", "give_haul_order", "give_back_order"] and callable(attr):
            return self._track_calls_and_time_wrapper(attr, name)
        return attr

    def _track_calls_and_time_wrapper(self, method, method_type):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = method(*args, **kwargs)
            elapsed_time = (time.time() - start_time) * 1000

            if method_type == "give_init_order":
                self.init_order_count += 1
                self.init_order_time += elapsed_time
            elif method_type == "give_haul_order":
                self.haul_order_count += 1
                self.haul_order_time += elapsed_time
            elif method_type == "give_back_order":
                self.back_order_count += 1
                self.back_order_time += elapsed_time

            self.total_order_count = self.init_order_count + self.haul_order_count + self.back_order_count
            self.total_order_time = self.init_order_time + self.haul_order_time + self.back_order_time

            return result
        return wrapper

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        raise NotImplementedError("Subclass must implement this method.")

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        raise NotImplementedError("Subclass must implement this method.")

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        raise NotImplementedError("Subclass must implement this method.")

class TestDispatcher(BaseDispatcher):
    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        # 具体实现
        time.sleep(0.6)
        return 0

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        # 具体实现
        return 0

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        # 具体实现
        return 0

if __name__ == "__main__":
    dispatcher = TestDispatcher()
    dispatcher.give_init_order("truck1", "mine1")
    dispatcher.give_init_order("truck1", "mine1")
    dispatcher.give_init_order("truck1", "mine1")
    dispatcher.give_init_order("truck1", "mine1")
    dispatcher.give_init_order("truck1", "mine1")
    dispatcher.give_haul_order("truck2", "mine2")
    dispatcher.give_back_order("truck3", "mine3")

    print("Init Order Count:", dispatcher.init_order_count)
    print("Haul Order Count:", dispatcher.haul_order_count)
    print("Back Order Count:", dispatcher.back_order_count)
    print("Total Order Count:", dispatcher.total_order_count)
    print("Init Order Time:", dispatcher.init_order_time)
    print("Haul Order Time:", dispatcher.haul_order_time)
    print("Back Order Time:", dispatcher.back_order_time)
    print("Total Order Time:", dispatcher.total_order_time)
