import json
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.legend_handler import HandlerBase
import matplotlib.image as mpimg
import numpy as np
import matplotlib.animation as animation
import os

from tqdm import tqdm
from matplotlib.font_manager import FontProperties

# 定义字体属性
legend_font = FontProperties(family='Times New Roman', size=8)

plt.rcParams['font.family'] = 'PingFang HK'
plt.rcParams["axes.unicode_minus"] = False

class VisualGrapher:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

        self.truck_colors = {}
        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        # Use os.path.join to construct image paths relative to the script's location
        script_directory = os.path.dirname(__file__)
        self.img_mine = mpimg.imread(os.path.join(script_directory, 'materials', '矿山.png'))
        self.img_shovel = mpimg.imread(os.path.join(script_directory, 'materials', '挖机.png'))
        self.img_truck_hauling = mpimg.imread(os.path.join(script_directory, 'materials', 'truck_hauling.png'))
        self.img_truck_unhauling = mpimg.imread(os.path.join(script_directory, 'materials', 'truck_unhauling.png'))
        self.img_truck_initing = mpimg.imread(os.path.join(script_directory, 'materials', 'truck_initing.png'))
        self.img_dump = mpimg.imread(os.path.join(script_directory, 'materials', 'dump.png'))

    def update_progress(self, pbar):
        pbar.update()

    def create_animation(self, output_path='output.gif'):
        frames = len(self.data)
        with tqdm(total=frames, desc="Generating Animation") as pbar:
            ani = animation.FuncAnimation(self.fig, lambda i: self.plot_tick(i, pbar), frames=frames, repeat=False,
                                          blit=False)
            ani.save(output_path, writer='pillow', fps=10, dpi=300)

    def plot_tick(self, i, pbar):
        self.update_progress(pbar)
        try:
            tick_data = self.data[str(i)]
        except KeyError:
            return
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        mine_data = tick_data['mine_states']
        # 1. 在图片上方展示矿山的总产量和服务次数
        self.ax.set_title(f"Production: {mine_data['produced_tons']:.2f} Tons, TruckCycles: {mine_data['service_count']} time: {i}/{len(self.data)-1}",fontproperties=legend_font)

        # self.ax.text(0.5, 0.95, f"Production: {mine_data['produced_tons']:.2f} Tons, TruckCycles: {mine_data['service_count']}",
        #              ha='center', va='center', fontsize=10, color='red')

        # 2. 准备动态图例的数据
        mine_stats = [
            ("Waiting Trucks", mine_data['waiting_truck_count']),
            ("Loading/Unloading Trucks", mine_data['load_unload_truck_count']),
            ("Moving Trucks", mine_data['moving_truck_count']),
            ("Repairing Trucks", mine_data['repairing_truck_count']),
            ("Road Jams", mine_data['road_jam_count']),
            ("Road Repairs", mine_data['road_repair_count']),
            ("Truck Repairs", mine_data['truck_repair']),
            ("Irreparable Trucks", mine_data['truck_unrepairable']),
            ("Random Events", mine_data['random_event_count'])
        ]

        # 3. 创建动态图例
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'{name}: {value}',
                                      markerfacecolor=colors[i], markersize=5) for i, (name, value) in enumerate(mine_stats)]
        self.ax.legend(handles=legend_elements, loc='upper right', prop=legend_font)

        DUMP_SITE_SCALE = 0.1    # 可以调整卸载区的缩放比例
        LOAD_SITE_SCALE = 0.1
        SHOVEL_SCALE = 0.10
        TRUCK_SCALE = 0.06

        # 3. 绘制装载区并展示总吨数
        for load_site, load_data in tick_data['load_site_states'].items():
            self.place_image(load_data['position'], self.img_mine, zoom=LOAD_SITE_SCALE)
            self.ax.text(load_data['position'][0], load_data['position'][1] - 0.06, load_site,
                         ha='center', fontsize=4)
            stats_text = f"Tons: {load_data['tons']:.2f}," \
                            f"count: {load_data['service_count']}"
            self.ax.text(load_data['position'][0], load_data['position'][1] + 0.05, stats_text,
                            ha='center', fontsize=4)

        # 4. 绘制卸载区并展示总吨数
        for dump_site, dump_data in tick_data['dump_site_states'].items():
            self.place_image(dump_data['position'], self.img_dump, zoom=DUMP_SITE_SCALE)
            self.ax.text(dump_data['position'][0], dump_data['position'][1] - 0.05, dump_site,
                         ha='center', fontsize=4)
            stats_text = f"Tons: {dump_data['tons']:.2f}," \
                         f"count: {dump_data['service_count']}"
            self.ax.text(dump_data['position'][0], dump_data['position'][1] + 0.05, stats_text,
                         ha='center', fontsize=4)

        # 5. 绘制铲车并展示其产量和服务次数
        for shovel, shovel_data in tick_data['shovel_states'].items():
            self.place_image(shovel_data['position'], self.img_shovel, zoom=SHOVEL_SCALE)
            # 在图像底部展示铲车名称
            self.ax.text(shovel_data['position'][0], shovel_data['position'][1] - 0.04, shovel,
                         ha='center', va='bottom', fontsize=3)
            # 在图像顶部展示产量和服务次数
            stats_text = f"tons: {shovel_data['tons']:.2f}, count: {shovel_data['service_count']}"
            self.ax.text(shovel_data['position'][0], shovel_data['position'][1] + 0.04, stats_text,
                         ha='center', va='top', fontsize=4)

        # 绘制矿车
        for truck, truck_data in tick_data['truck_states'].items():
            if truck_data['state'] in [self.ON_CHARGING_SITE, self.ON_ROAD_TO_FIRST_LOAD]:
                img_truck = self.img_truck_initing
            elif truck_data['state'] == self.UNHAULING:
                img_truck = self.img_truck_unhauling
            elif truck_data['state'] == self.HAULING:
                img_truck = self.img_truck_hauling
            else:
                img_truck = self.img_truck_initing
            self.place_image(truck_data['position'], img_truck, zoom=TRUCK_SCALE)
            self.ax.text(truck_data['position'][0], truck_data['position'][1] - 0.035, truck, ha='center', fontsize=3,
                         color='black')


    def place_image(self, xy, img, zoom=1):
        """
        在指定的坐标上放置图像
        :param xy: 放置图像的坐标 (x, y)
        :param img: 图像数组
        :param zoom: 图像的缩放级别
        """
        im_offset = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im_offset, xy, frameon=False, box_alignment=(0.5, 0.5))
        self.ax.add_artist(ab)

# 状态常量定义
VisualGrapher.ON_CHARGING_SITE = -1
VisualGrapher.ON_ROAD_TO_FIRST_LOAD = -2
VisualGrapher.UNHAULING = 0
VisualGrapher.WAIT_FOR_LOAD = 1
VisualGrapher.LOADING = 2
VisualGrapher.HAULING = 3
VisualGrapher.WAIT_FOR_DUMP = 4
VisualGrapher.DUMPING = 5
VisualGrapher.REPAIRING = 6
VisualGrapher.UNREPAIRABLE = 7

class ImageHandler(HandlerBase):
    def __init__(self, image_ref, scale):
        super(ImageHandler, self).__init__()
        self.image_ref = image_ref
        self.scale = scale

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        zoom = self.scale / height  # 缩放图片以适应图例
        im = OffsetImage(self.image_ref, zoom=zoom)
        ab = AnnotationBbox(im, (width/2., height/2.), xycoords='axes fraction', frameon=False)
        return [ab]



if __name__ == '__main__':
    path = "/Users/mac/PycharmProjects/truck_shovel_mix/openmines_project/openmines/src/cli/results/MINE:NorthOpenPitMine_ALGO:RandomDispatcher_TIME:2024-01-21 14:26:04.gif"
    visualizer = VisualGrapher(path)
    visualizer.create_animation(output_path='output.gif')
