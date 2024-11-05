from openmines_gym.envs.mine_env import GymMineEnv,ThreadMineEnv
from gymnasium.envs.registration import register


# 也可以添加一个显式的注册函数
def register_envs():
    register(
        id='Mine-v0',
        entry_point='openmines_gym.envs.mine_env:GymMineEnv',
    )
    register(
        id='Mine-v1',
        entry_point='openmines_gym.envs.mine_env:ThreadMineEnv',
    )

register_envs()