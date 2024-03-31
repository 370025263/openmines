
class Event:
    def __init__(self, time_stamp:float, event_type:str, desc:str, info:dict=None):
        self.time_stamp = time_stamp
        self.event_type = event_type
        self.desc = desc
        self.info = info

    def __str__(self):
        return f"Event(time_stamp={self.time_stamp},event_type={self.event_type},desc={self.desc},info={self.info})"

    def __repr__(self):
        return self.__str__()


class EventPool:
    """
    用来存储卡车运行事件的类，便于后期统计分析
    """
    def __init__(self):
        self.event_set = dict()

    def add_event(self, event:Event):
        if event.time_stamp in self.event_set.keys():
            event.time_stamp = event.time_stamp + 0.0001
        self.event_set[event.time_stamp] = event

    def get_even_by_type(self, name:str, )->list:
        """
        通过事件type获取事件
        :param name:
        :return:
        """
        list_event = []
        for t in sorted(self.event_set.keys()):
            if name in self.event_set[t].event_type:
                list_event.append(self.event_set[t])
        return list_event

    def get_even_by_desc(self, name:str)->list:
        """
        通过事件desc获取事件
        :param name:
        :return:
        """
        list_event = []
        for t in sorted(self.event_set.keys()):
            if name in self.event_set[t].desc:
                list_event.append(self.event_set[t])
        return list_event

    def get_event_by_time(self,time:float,mode="backward")->list:
        """
        给定一个时间，获取当前时间之前的顺序的event列表
        :param time:
        :return:
        """
        if mode == "backward":
            list_event = []
            for t in sorted(self.event_set.keys()):
                if t <= time:
                    list_event.append(self.event_set[t])
            return list_event
        else:
            list_event = []
            for t in sorted(self.event_set.keys()):
                if t > time:
                    list_event.append(self.event_set[t])
            return list_event

    def get_event_by_time_range(self, start_time:float, end_time:float)->list:
        """
        给定一个时间范围，获取当前时间之前的顺序的event列表
        :param time:
        :return:
        """
        list_event = []
        for t in sorted(self.event_set.keys()):
            if start_time <= t <= end_time:
                list_event.append(self.event_set[t])
        return list_event

    def update_last_info(self, type:str, info:dict,strict:bool=True):
        """
        更新最后一个事件的info
        如果strict为True，则
            断言这个事件的type是给定type
        如果strict为False，则
            更新最近的匹配到的event的info

        :param type:
        :param info:
        :param strict:
        :return:
        """
        if strict:
            assert self.event_set[list(self.event_set.keys())[-1]].event_type == type
            self.event_set[list(self.event_set.keys())[-1]].info = info
        else:
            for t in sorted(self.event_set.keys(),reverse=True):
                if type in self.event_set[t].event_type:
                    self.event_set[t].info = info
                    break

    def get_last_event(self,type:str,strict:bool=True):
        """
        获取最后一个事件
        如果strict为True，则
            断言这个事件的type是给定type
        如果strict为False，则
            更新最近的匹配到的event的info
        """
        if strict:
            assert self.event_set[list(self.event_set.keys())[-1]].event_type == type
            return self.event_set[list(self.event_set.keys())[-1]]
        else:
            for t in sorted(self.event_set.keys(),reverse=True):
                if type in self.event_set[t].event_type:
                    return self.event_set[t]
                    break

    def clear(self):
        self.event_set.clear()

class RandomEventPool:
    """
    1.用来记录在模拟运行过程中的随机事件以供后期分析
    2.在算法派车决策时，提供现有的随机事件
    """
    pass


if __name__ == "__main__":
    # test EventPool
    print("==============test EventPool================")
    pool = EventPool()
    pool.add_event(Event(3.0, 'test', 'desc'))
    pool.add_event(Event(1.0, 'test', 'desc'))
    pool.add_event(Event(1.0, 'test2', 'desc'))
    pool.add_event(Event(2.0, 'dasdasd', 'desc'))

    print(pool.get_even_by_type('test'))
    print(pool.get_even_by_desc('desc'))
    # clear
    pool.clear()
    # test update_last_info
    print("==============test update_last_info================")
    pool.add_event(Event(1.0, 'move', 'desc'))
    pool.add_event(Event(1.5, 'wait', 'desc', info={'load_time': 9527}))
    pool.add_event(Event(2.0, 'wait', 'desc',info={'wait_time':0}))
    pool.add_event(Event(3.0, 'load', 'desc',info={'load_time':98}))
    pool.update_last_info('wait', {'wait_time': 10}, strict=False)
    print(pool.get_even_by_type('wait'))

    # test get_last_event
    print("==============test get_last_event================")
    event1 = pool.get_last_event('wait',strict=False)
    print(f"BEFORE: {event1}")
    event1.info['wait_time'] = 100000
    print(f"AFTER: {pool.get_last_event('wait',strict=False)}")

    # test get_event_by_time
    print("==============test get_event_by_time================")
    for t in range(100):
        pool.add_event(Event(time_stamp=t, event_type='test', desc='desc'))
    print(pool.get_event_by_time(21))