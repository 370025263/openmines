import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

from matplotlib.font_manager import FontProperties
from tabulate import tabulate  # ★ 用于在终端打印表格


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
            total_order_count = states['summary']['total_order_count']

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
                "Road Jams": road_jams,
                "TruckTrips": total_order_count,
                "Jam Ratio": f"{round(road_jams / states['summary']['total_order_count'], 2):.2f}"
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

        # 最后：在命令行打印出 final_table
        self._print_table_in_terminal(final_table)

    def draw_scene_based_fleet_ablation_experiment(self, results, original_fleet_size):
        """
        results: { dispatcher_name: { 'fleet_sizes': [...], 'productions': [...] } }
        original_fleet_size: int
        只画一张对比图, 不输出表格
        """
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(5,4))

        for dispatcher_name, data in results.items():
            fsizes = data['fleet_sizes']
            prods = data['productions']
            ax.plot(fsizes, prods, marker='o', label=dispatcher_name)
            arr = np.array(prods)
            if len(arr)>0:
                mx_i = arr.argmax()
                mn_i = arr.argmin()
                ax.plot(fsizes[mx_i], prods[mx_i], 'ko', mfc='black')
                ax.plot(fsizes[mn_i], prods[mn_i], 'ko', mfc='black')

        # 画竖线
        ax.axvline(x=original_fleet_size, color='gray', linestyle='--', label='Original Fleet')

        # 调整范围
        all_fs = []
        all_pd = []
        for d_ in results.values():
            all_fs += d_['fleet_sizes']
            all_pd += d_['productions']
        if all_fs:
            fs_min, fs_max = min(all_fs), max(all_fs)
            pd_min, pd_max = min(all_pd), max(all_pd)
            margin_x = 0.05*(fs_max - fs_min) if fs_max>fs_min else 1
            margin_y = 0.05*(pd_max - pd_min) if pd_max>pd_min else 1
            ax.set_xlim(fs_min - margin_x, fs_max + margin_x)
            ax.set_ylim(pd_min - margin_y, pd_max + margin_y)

        ax.set_xlabel("Fleet Size", fontsize=10, fontweight='bold')
        ax.set_ylabel("Production (tons)", fontsize=10, fontweight='bold')
        ax.set_title("Scene-based Fleet Ablation", fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')

        plt.tight_layout()
        self.fig_img = fig  # 覆盖 self.fig_img

    def draw_algo_based_fleet_ablation_experiment(self, scenes_data, baseline, target):
        """
        本函数绘制“多场景 + 双算法”下的消融结果，但仅显示目标算法与基线算法的产量比 (ratio = target / baseline)，
        每个场景一条折线。

        scenes_data 的结构示例:
        {
          "sceneA": { "fleet_sizes": [...], "ratios": [...] },
          "sceneB": { "fleet_sizes": [...], "ratios": [...] },
          ...
        }
        其中 ratio = target_production / baseline_production.
        """

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.font_manager import FontProperties

        # 使用与 draw 函数相同的 Times New Roman 字体与字号
        times_new_roman = FontProperties(family='Times New Roman')

        # 建议的画布尺寸，可以与 scene_ablation 类似
        fig, ax = plt.subplots(figsize=(5, 4))

        # 使用 tab10 调色板给不同场景分配颜色
        color_map = plt.cm.get_cmap('tab10', len(scenes_data))

        # 用来收集全部 x（fleet_sizes）与 ratio，以便后续自动设置坐标范围
        all_fs = []
        all_ratios = []

        # 遍历每个场景的数据
        for i, (scene_name, data) in enumerate(scenes_data.items()):
            fsizes = data['fleet_sizes']
            ratio = data['ratios']

            # 选择对应颜色
            color_ = color_map(i)

            # 绘制该场景的折线
            # marker='o'表示用小圆点, linewidth=0.7表示线宽
            ax.plot(fsizes, ratio, marker='o', linewidth=0.7, color=color_, label=scene_name)

            # 若需要标记最高点/最低点，则如下:
            arr_r = np.array(ratio)
            if arr_r.size > 0:
                rmax_i = arr_r.argmax()
                rmin_i = arr_r.argmin()
                # 在最高点和最低点画实心黑色圆点
                ax.plot(fsizes[rmax_i], ratio[rmax_i], 'ko', mfc='black')
                ax.plot(fsizes[rmin_i], ratio[rmin_i], 'ko', mfc='black')

            # 收集数据用于设置坐标范围
            all_fs.extend(fsizes)
            all_ratios.extend(ratio)

        # 自动设定 x、y 的范围，防止图像过于稀疏或挤压
        if all_fs:
            fs_min, fs_max = min(all_fs), max(all_fs)
            r_min, r_max = min(all_ratios), max(all_ratios)
            margin_x = 0.05 * (fs_max - fs_min) if fs_max > fs_min else 1
            margin_y = 0.05 * (r_max - r_min) if r_max > r_min else 0.1

            ax.set_xlim(fs_min - margin_x, fs_max + margin_x)
            # 若 r_max <= 0 则设置一个默认值，防止 y 轴范围为负或 0
            ax.set_ylim(r_min - margin_y, r_max + margin_y if r_max > 0 else 0.1)

        # 设置坐标轴刻度的字体大小（与 draw 函数一致）
        ax.tick_params(axis='both', which='major', labelsize=6)

        # 图例设置：字号可与 draw 函数保持一致
        ax.legend(prop={'size': 5}, loc='best')

        # 设置坐标轴标签，采用 Times New Roman 字体
        ax.set_xlabel("Fleet Size", fontproperties=times_new_roman, fontsize=8)
        ax.set_ylabel("Production Ratio (Target / Baseline)", fontproperties=times_new_roman, fontsize=8)

        # 不再设置标题，直接移除 ax.set_title(...)，也不使用 fig.text。
        # 如果想简单放一行文字，可以手动调用 fig.text(...)，否则就完全省去。
        ax.set_title("Algorithm-based Fleet Ablation", fontproperties=times_new_roman, fontsize=10)
        # 布局紧凑
        plt.tight_layout()

        # 将绘制的 Figure 存入 self.fig_img
        self.fig_img = fig

    def save_ablation(self, tag="ablation"):
        """
        仅保存 self.fig_img (消融图), 并使用单独命名, 不保存表格/产量图.
        """
        if not self.fig_img:
            print("[save_ablation] No ablation figure found.")
            return
        ablation_filename = self.config_name.replace('.json','') + f"_{tag}.tiff"
        ablation_path = self.result_path / ablation_filename
        self.fig_img.savefig(ablation_path, dpi=300, format='tiff')
        print(f"[save_ablation] Saved ablation figure -> {ablation_path}")

    def _print_table_in_terminal(self, final_table):
        """
        使用 Rich 库在命令行中打印一个自适应、紧凑、支持自动换行的表格。
        """
        try:
            import pandas as pd
            from rich.table import Table
            from rich.console import Console
            print("Rich is installed!")
        except ImportError:
            print("Rich is not installed. Please run 'pip install rich'.")


        df = pd.DataFrame(final_table)

        console = Console()

        # 创建一个 Rich Table，自动展开并显示行分割线
        table = Table(title="Final Metrics Table", show_lines=True, expand=True)

        # 遍历 DataFrame 的列名，动态添加表格列
        # overflow="fold" + no_wrap=False 可以在列内容过长时自动换行
        for col_name in df.columns:
            table.add_column(str(col_name), overflow="fold", no_wrap=False)

        # 依次添加每一行的数据
        for _, row in df.iterrows():
            row_values = [str(val) for val in row.values]
            table.add_row(*row_values)

        # 在终端打印该 Rich 表格
        console.print(table)
        print("----- End of Table -----\n")

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


