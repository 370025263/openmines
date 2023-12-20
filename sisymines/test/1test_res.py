import simpy

env = simpy.Environment()

res = simpy.Resource(env, capacity=1)

def print_stats(res):
    print(f'{res.count} of {res.capacity} slots are allocated.')
    print(f'  Users: {res.users}')
    print(f'  Queued events: {res.queue}')
    print('####################')


def user(res):
    print("请求资源前：")
    print_stats(res)  # 4 ge zai pai dui
    with res.request() as req:
        yield req
        print("使用资源中：")
        print_stats(res)  # 3 ge zai pai dui
    print("使用后：")
    print_stats(res) # 3 ge zai pai dui
    print('------------------')

procs = [env.process(user(res)), env.process(user(res)),env.process(user(res)),env.process(user(res))]
env.run()