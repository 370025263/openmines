from setuptools import setup, find_packages

import openmines

# 读取requirements.txt文件中的内容
with open('openmines/requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='openmines',
    version=openmines.__version__,
    author='stone91',  # 添加作者
    description='Mining Fleet Dispatch Algorithm Framework',  # 添加描述（英文）
    license='MIT',  # 添加许可证
    packages=find_packages(),
    package_data={
        # 确保你的包名正确
        'openmines': ['./src/utils/visualization/materials/*'],
    },
    entry_points={
        'console_scripts': [
            'openmines=openmines.src.cli.run:main',
        ],
    },
    install_requires=required,  # 使用requirements.txt中的依赖
    # 可以在此添加更多元数据，如作者邮箱、项目网址等
)