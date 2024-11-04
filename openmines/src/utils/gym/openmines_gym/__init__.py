from openmines_gym.envs.mine_env import GymMineEnv
from gymnasium.envs.registration import register

register(
    id='Mine-v0',
    entry_point='openmines_gym.envs.mine_env:GymMineEnv',
)

__all__ = ['GymMineEnv']