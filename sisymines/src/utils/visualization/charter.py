import json
import os
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

from matplotlib.font_manager import FontProperties

# 设置Times New Roman字体
times_new_roman = FontProperties(family='Times New Roman', size=8)

class Charter:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_name = Path(config_file).name
        self.img_name = self.config_name.replace('.json', '.tiff')
        self.tab_name = self.config_name.replace('.json', '') + "_table" + '.tiff'
        self.fig_img = None
        self.fig_tab = None

        # 使用Path来处理路径，以确保跨平台兼容性
        self.result_path = Path.cwd() / 'results'

        # 确保结果目录存在
        self.result_path.mkdir(exist_ok=True)

        # 使用Path对象组合路径
        self.img_path = self.result_path / self.img_name
        self.tab_path = self.result_path / self.tab_name

    def draw(self, states_dict):
        # 绘制图像
        fig_img, axs = plt.subplots(2, 1, figsize=(3.5, 5))  # 创建两个子图的图像

        # 存储最终的production数据
        final_table = []
        # 遍历每个调度器的结果并绘图
        for dispatcher_name, states in states_dict.items():
            # 获取必要统计信息
            decision_latency = states['summary']['avg_time_per_order']
            matching_factor = states['summary']['MatchingFactor']
            total_wait_time = states['summary']['TotalWaitTime']
            road_jams = states['summary']['RoadJams']
            produced_tons = states['produced_tons_list'][-1]

            # 绘制产量曲线
            axs[0].plot(states['times'], states['produced_tons_list'], label=dispatcher_name, linewidth=0.7)
            # 绘制waitingTruck曲线
            axs[1].plot(states['times'], states['waiting_truck_count_list'], label=dispatcher_name, linewidth=0.7)
            # final_table
            final_table.append({
                'Name': dispatcher_name,
                'Produced Tons': f"{round(produced_tons, 2):.2f}",
                'Decision Latency (ms)': f"{round(decision_latency, 1):.1f}",
                'Matching Factor': f"{round(matching_factor, 2):.2f}",
                'Total Wait Time': f"{round(total_wait_time, 2):.2f}",
                "Road Jams": road_jams
            })
        # 设置坐标轴刻度的字体大小
        axs[0].tick_params(axis='both', which='major', labelsize=6)
        axs[1].tick_params(axis='both', which='major', labelsize=6)

        # 设置图像的图例和标题
        axs[0].legend(prop={'size': 5}, loc='upper left')
        axs[1].legend(prop={'size': 5}, loc='center right')
        axs[0].set_title("Production Over Time", fontproperties=times_new_roman)
        axs[1].set_title("Waiting Trucks Over Time", fontproperties=times_new_roman)
        axs[0].set_xlabel("Time (mins)", fontproperties=times_new_roman)
        axs[0].set_ylabel("Production (tons)", fontproperties=times_new_roman)
        axs[1].set_xlabel("Time (mins)", fontproperties=times_new_roman)
        axs[1].set_ylabel("Number of Waiting Trucks", fontproperties=times_new_roman)

        plt.tight_layout()
        self.fig_img = fig_img


        # 绘制表格
        fig_tab, ax_table = plt.subplots(figsize=(15, 5))  # 创建单独的Figure用于表格
        df = pd.DataFrame(final_table)
        ax_table.axis('tight')
        ax_table.axis('off')
        table = ax_table.table(cellText=df.values, colLabels=df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.tight_layout()
        self.fig_tab = fig_tab

    def save(self):
        # 保存图像
        if self.fig_img is None or self.fig_tab is None:
            raise Exception("You must draw the chart first!")
        self.fig_img.savefig(self.img_path, dpi=300, format='tiff')
        self.fig_tab.savefig(self.tab_path, dpi=300, format='tiff')
