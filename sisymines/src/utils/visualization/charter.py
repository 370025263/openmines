import json
import os

import pandas as pd
from matplotlib import pyplot as pl, pyplot as plt
from pathlib import Path

class Charter:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_name = Path(config_file).name
        self.img_name = self.config_name.replace('.json', '.png')
        self.fig = None

        # 使用Path来处理路径，以确保跨平台兼容性
        self.result_path = Path.cwd() / 'results'

        # 确保结果目录存在
        self.result_path.mkdir(exist_ok=True)

        # 使用os.path.join或Path对象组合路径
        self.img_path = self.result_path / self.img_name

    def draw(self, states_dict):
        ## 绘制每一个算法的产量曲线，并添加图例
        # 初始化绘图区域
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        # 存储最终的production数据
        final_productions = []
        # 遍历每个调度器的结果并绘图
        for dispatcher_name, states in states_dict.items():
            # 绘制产量曲线
            axs[0, 0].plot(states['times'], states['produced_tons'], label=dispatcher_name)
            # 绘制waitingTruck曲线
            axs[0, 1].plot(states['times'], states['waiting_truck_count'], label=dispatcher_name)
            # 添加到final_productions
            final_productions.append({
                'Name': dispatcher_name,
                'production': states['produced_tons'][-1]
            })
        # 设置图例
        axs[0, 0].legend()
        axs[0, 1].legend()

        # 设置标题
        axs[0, 0].set_title("Production Over Time")
        axs[0, 1].set_title("Waiting Trucks Over Time")

        # 创建并显示表格
        df = pd.DataFrame(final_productions)
        axs[1, 0].axis('tight')
        axs[1, 0].axis('off')
        table = axs[1, 0].table(cellText=df.values, colLabels=df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # 调整布局并显示图表
        plt.tight_layout()
        plt.show()
        self.fig = fig
    def save(self):
        # 保存图表
        if self.fig is None:
            raise Exception("You must draw the chart first!")
        self.fig.savefig(self.img_path)