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
        # 初始化绘图区域，使用2x2的布局，并合并最后一行的两个子图格
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.delaxes(axs[1, 1])  # 删除第二行第二列的子图
        fig.delaxes(axs[1, 0])  # 删除第二行第一列的子图
        ax_table = plt.subplot2grid((2, 2), (1, 0), colspan=2)  # 在第二行添加一个跨越两列的子图用于表格

        # 存储最终的production数据
        final_table = []
        # 遍历每个调度器的结果并绘图
        for dispatcher_name, states in states_dict.items():
            # 获取必要统计信息
            decision_latency = states['summary']['avg_time_per_order']
            matching_factor = states['summary']['MatchingFactor']
            total_wait_time = states['summary']['TotalWaitTime']
            produced_tons = states['produced_tons_list'][-1]

            # 绘制产量曲线
            axs[0, 0].plot(states['times'], states['produced_tons_list'], label=dispatcher_name)
            # 绘制waitingTruck曲线
            axs[0, 1].plot(states['times'], states['waiting_truck_count_list'], label=dispatcher_name)
            # final_table
            final_table.append({
                'Name': dispatcher_name,
                'Produced Tons': f"{round(produced_tons, 2):.2f}",
                'Decision Latency (ms)': f"{round(decision_latency, 1):.1f}",
                'Matching Factor': f"{round(matching_factor, 2):.2f}",
                'Total Wait Time': f"{round(total_wait_time, 2):.2f}"
            })

        # 设置图例和标题
        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[0, 0].set_title("Production Over Time")
        axs[0, 1].set_title("Waiting Trucks Over Time")

        # 创建并显示表格
        df = pd.DataFrame(final_table)
        ax_table.axis('tight')
        ax_table.axis('off')
        table = ax_table.table(cellText=df.values, colLabels=df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        # 调整布局并显示图表
        # 提高dpi以提高图像质量
        # 调整子图之间的间距
        plt.subplots_adjust(hspace=0.01, wspace=0.3)
        # 调整布局并显示图表
        plt.tight_layout()
        plt.show()
        self.fig = fig
    def save(self):
        # 保存图表
        if self.fig is None:
            raise Exception("You must draw the chart first!")
        self.fig.savefig(self.img_path, dpi=300)