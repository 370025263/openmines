import json
import os

import numpy as np
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

        # Plot the histogram
        load_site_names = [state for state in states_dict.values()][0]['summary']["load_sites_name"]
        dump_site_names = [state for state in states_dict.values()][0]['summary']["dump_sites_name"]

        dispatcher_name_list = list(states_dict.keys())
        init_orders_list = [states['summary']['init_orders'] for states in states_dict.values()]
        haul_orders_list = [states['summary']['haul_orders'] for states in states_dict.values()]
        back_orders_list = [states['summary']['back_orders'] for states in states_dict.values()]
        self.init_orders_fig = self.plot_histogram(init_orders_list, prefix_list=dispatcher_name_list, entre_names=load_site_names, title="LoadSite Orders")
        self.haul_orders_fig = self.plot_histogram(haul_orders_list, prefix_list=dispatcher_name_list, entre_names=dump_site_names, title="DumpSite Orders")
        self.back_orders_fig = self.plot_histogram(back_orders_list, prefix_list=dispatcher_name_list, entre_names=dump_site_names, title="BackSite Orders")

    @staticmethod
    def plot_histogram(data_list, prefix_list, entre_names=None, title="Histogram of SiteOrder", show=False):
        # 统计每个数字出现的次数
        count_dicts = []
        for data in data_list:
            count_dict = {}
            for num in data:
                if num in count_dict:
                    count_dict[num] += 1
                else:
                    count_dict[num] = 1
            count_dicts.append(count_dict)

        # 获取所有唯一的数字
        unique_numbers = sorted(set().union(*[d.keys() for d in count_dicts]))

        # Conditionally set tick labels
        if entre_names is not None and len(entre_names) == len(unique_numbers):
            tick_labels = entre_names
        else:
            tick_labels = [f'Site {num}' for num in unique_numbers]

        # 动态设置柱状图的宽度和位置
        num_groups = len(prefix_list)  # 数据列表的数量
        bar_width = 0.8 / num_groups  # 动态计算每个柱子的宽度
        max_bar_width = 0.4  # 设置最大柱宽
        bar_width = min(bar_width, max_bar_width)  # 保证柱子的最大宽度不会超过max_bar_width
        index = np.arange(len(unique_numbers))  # x轴位置基于 unique_numbers

        # 绘制柱状图
        fig = plt.figure(figsize=(10, 6))
        for i, (count_dict, prefix) in enumerate(zip(count_dicts, prefix_list)):
            counts = [count_dict.get(num, 0) for num in unique_numbers]
            plt.bar(index + i * bar_width, counts, bar_width, label=prefix, color=plt.cm.viridis(i / num_groups),
                    edgecolor='black', linewidth=0.7)

        # 设置x轴标签
        plt.xlabel('Sites', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')

        # 设置刻度位置和标签
        tick_positions = index + (num_groups - 1) * bar_width / 2
        plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=10)
        plt.yticks(fontsize=10)

        # 添加网格线
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 设置图例
        plt.legend(frameon=False, fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

        # 去除顶部和右侧边框
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # 显示图形
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def save(self):
        # 保存图像
        if self.fig_img is None or self.fig_tab is None:
            raise Exception("You must draw the chart first!")
        self.fig_img.savefig(self.img_path, dpi=300, format='tiff')
        self.fig_tab.savefig(self.tab_path, dpi=300, format='tiff')
        self.init_orders_fig.savefig(self.result_path / "init_orders.png", dpi=300, format='png')
        self.haul_orders_fig.savefig(self.result_path / "haul_orders.png", dpi=300, format='png')
        self.back_orders_fig.savefig(self.result_path / "back_orders.png", dpi=300, format='png')


