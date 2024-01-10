import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation
import numpy as np

plt.rcParams['font.family'] = 'PingFang HK'
plt.rcParams["axes.unicode_minus"] = False

class VisualGrapher:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

        self.truck_colors = {}
        self.fig, self.ax = plt.subplots()

    def create_animation(self, output_path='output.gif'):
        ani = animation.FuncAnimation(self.fig, self.plot_tick, frames=len(self.data), repeat=False)
        ani.save(output_path, writer='pillow', fps=2, dpi=300)


    def plot_tick(self, i):
        tick_data = self.data[str(i)]
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title(f"time_step: {i}/{len(self.data)}")

        rect_width, rect_height = 0.04, 0.02  # 长方形的宽度和高度

        # 绘制装载区
        for load_site, load_data in tick_data['load_site_states'].items():
            rect_x, rect_y = load_data['position'][0] - rect_width / 2, load_data['position'][1] - rect_height / 2
            self.ax.add_patch(Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, edgecolor='sienna'))
            self.ax.text(load_data['position'][0], load_data['position'][1] - rect_height / 2 - 0.025, load_site, ha='center', fontsize=5)
            self.ax.text(load_data['position'][0], load_data['position'][1] + rect_height / 2 + 0.025, str(load_data['total_queue_length']), ha='center', color='sienna', fontsize=5)

        # 绘制电铲
        for shovel, shovel_data in tick_data['shovel_states'].items():
            self.ax.add_patch(Circle(shovel_data['position'], 0.01, fill=True, color='yellow'))
            self.ax.text(shovel_data['position'][0], shovel_data['position'][1] - 0.025, shovel, ha='center', fontsize=5)
            self.ax.text(shovel_data['position'][0], shovel_data['position'][1] + 0.02, str(shovel_data['queue_length']), ha='center', color='yellow', fontsize=5)

        # 绘制卸载区
        for dump_site, dump_data in tick_data['dump_site_states'].items():
            rect_x, rect_y = dump_data['position'][0] - rect_width / 2, dump_data['position'][1] - rect_height / 2
            self.ax.add_patch(Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, edgecolor='brown'))
            self.ax.text(dump_data['position'][0], dump_data['position'][1] - rect_height / 2 - 0.025, dump_site, ha='center', fontsize=5)
            self.ax.text(dump_data['position'][0], dump_data['position'][1] + rect_height / 2 + 0.025, str(dump_data['total_queue_length']), ha='center', color='brown', fontsize=5)

        # 绘制卸载点
        for dumper, dumper_data in tick_data['dumper_states'].items():
            self.ax.add_patch(Circle(dumper_data['position'], 0.01, fill=True, color='black'))
            self.ax.text(dumper_data['position'][0], dumper_data['position'][1] - 0.025, dumper, ha='center', fontsize=5)
            self.ax.text(dumper_data['position'][0], dumper_data['position'][1] + 0.02, str(dumper_data['queue_length']), ha='center', color='black', fontsize=5)

        # 绘制卡车
        for truck, truck_data in tick_data['truck_states'].items():
            color = self.truck_colors.get(truck, np.random.rand(3,))
            self.truck_colors[truck] = color
            self.ax.plot(truck_data['position'][0], truck_data['position'][1], 'o', color=color)
            self.ax.text(truck_data['position'][0], truck_data['position'][1] - 0.025, truck, ha='center', fontsize=5, color=color)

        # 添加图例
        self.ax.legend(['装载点', '铲子', '卸载区', '卸载点', '卡车'], loc='upper right', fontsize=5)


if __name__ == '__main__':
    path = "/Users/mac/PycharmProjects/truck_shovel_mix/sisymines_project/sisymines/src/results/MINE:北露天矿_ALGONaiveDispatcher_TIME:2023-12-23 09:09:29.json"
    visualizer = VisualGrapher(path)
    visualizer.create_animation(output_path='output.gif')
