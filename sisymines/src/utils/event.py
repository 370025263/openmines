
class Event:
    def __init__(self, time_stamp:float, event_type:str, param:str):
        self.time_stamp = time_stamp
        self.event_type = event_type
        self.param = param

    def __str__(self):
        return f"Event(time_stamp={self.time_stamp},event_type={self.event_type},param={self.param})"

    def __repr__(self):
        return self.__str__()


class EventPool:
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

    def get_even_by_name(self, name:str)->list:
        """
        通过事件param获取事件
        :param name:
        :return:
        """
        list_event = []
        for t in sorted(self.event_set.keys()):
            if name in self.event_set[t].param:
                list_event.append(self.event_set[t])
        return list_event


if __name__ == "__main__":
    # test
    pool = EventPool()
    pool.add_event(Event(3.0, 'test', 'param'))
    pool.add_event(Event(1.0, 'test', 'param'))
    pool.add_event(Event(1.0, 'test2', 'param'))
    pool.add_event(Event(2.0, 'dasdasd', 'param'))

    print(pool.get_even_by_type('test'))
    print(pool.get_even_by_name('param'))