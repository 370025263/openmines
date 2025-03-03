from openmines.src.utils.gym.openmines_gym.envs.mine_env import *
from gymnasium.envs.registration import register
 

# 也可以添加一个显式的注册函数
def register_envs():
    register(
        id='Mine-v0',
        entry_point='openmines.src.utils.gym.openmines_gym.envs.mine_env:GymMineEnv',
    )
    register(
        id='Mine-v1',
        entry_point='openmines.src.utils.gym.openmines_gym.envs.mine_env:ThreadMineEnv',
    ) # dense default; equivilant to Mine-v1-dense
    register(
        id='Mine-v1-dense',
        entry_point='openmines.src.utils.gym.openmines_gym.envs.mine_env:ThreadMineDenseEnv',
    )
    register(
        id='Mine-v1-sparse',
        entry_point='openmines.src.utils.gym.openmines_gym.envs.mine_env:ThreadMineSparseEnv',
    )  # sparse reward

register_envs()