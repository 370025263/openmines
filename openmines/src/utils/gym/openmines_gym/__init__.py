from openmines_gym.envs.mine_env import GymMineEnv
from gymnasium.envs.registration import register

# 确保这段代码在导入时就执行
register(
    id='Mine-v0',
    entry_point='openmines_gym.envs.mine_env:GymMineEnv',
)

# 也可以添加一个显式的注册函数
def register_envs():
    register(
        id='Mine-v0',
        entry_point='openmines_gym.envs.mine_env:GymMineEnv',
    )