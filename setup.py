from setuptools import setup, find_packages

import openmines

# 读取requirements.txt文件中的内容
with open('openmines/requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='openmines',
    version=openmines.__version__,
    author='stone91',
    description='Mining Fleet Dispatch Algorithm Framework',
    license='MIT',
    packages=find_packages(),
    package_data={
        'openmines': ['./src/utils/visualization/materials/*'],
    },
    entry_points={
        'console_scripts': [
            'openmines=openmines.src.cli.run:main',
        ],
        'gymnasium.envs': [
            'mine=openmines.src.utils.gym.openmines_gym:register_envs',
        ],
    },
    install_requires=required + ['gymnasium==0.28.1'],
)