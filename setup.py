from setuptools import setup, find_packages

setup(
    name='sisymines',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'sisymines=sisymines.src.cli.run:main',
        ],
    },
    # 依赖项可以在这里列出，例如：
    install_requires=[
        'numpy',
        # 其他依赖...
    ],
    # 其他元数据，例如作者、描述、许可证等...
)


