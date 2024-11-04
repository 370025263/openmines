# openmines/src/utils/gym/openmines_gym/__init__.py
from openmines.src.utils.gym.openmines_gym.envs.mine_env import GymMineEnv
from gymnasium.envs.registration import register

register(
    id='Mine-v0',
    entry_point='openmines.src.utils.gym.openmines_gym.envs.mine_env:GymMineEnv',
)