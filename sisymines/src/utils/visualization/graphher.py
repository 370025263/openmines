import json
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.legend_handler import HandlerBase
import matplotlib.image as mpimg
import numpy as np
import matplotlib.animation as animation
import os

plt.rcParams['font.family'] = 'PingFang HK'
plt.rcParams["axes.unicode_minus"] = False

class VisualGrapher:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

        self.truck_colors = {}
        self.fig, self.ax = plt.subplots()

        # 路径可能需要根据您的实际文件位置进行调整
        self.img_mine = mpimg.imread(os.path.join('materials', '矿山.png'))
        self.img_shovel = mpimg.imread(os.path.join('materials', '挖机.png'))
        self.img_truck_hauling = mpimg.imread(os.path.join('materials', 'truck_hauling.png'))
        self.img_truck_unhauling = mpimg.imread(os.path.join('materials', 'truck_unhauling.png'))
        self.img_truck_initing = mpimg.imread(os.path.join('materials', 'truck_initing.png'))
        self.img_dump = mpimg.imread(os.path.join('materials', 'dump.png'))

    def create_animation(self, output_path='output.gif'):
        ani = animation.FuncAnimation(self.fig, self.plot_tick, frames=len(self.data), repeat=False)
        ani.save(output_path, writer='pillow', fps=10, dpi=300)

    def plot_tick(self, i):
        tick_data = self.data[str(i)]
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title(f"time_step: {i}/{len(self.data)}")

        DUMP_SITE_SCALE = 0.08  # 可以调整卸载区的缩放比例
        LOAD_SITE_SCALE = 0.08
        SHOVEL_SCALE = 0.10
        TRUCK_SCALE = 0.06

        # 绘制装载区
        for load_site, load_data in tick_data['load_site_states'].items():
            self.place_image(load_data['position'], self.img_mine, zoom=LOAD_SITE_SCALE)
            self.ax.text(load_data['position'][0], load_data['position'][1] - 0.05, load_site, ha='center', fontsize=5)
        # 绘制卸载区
        for dump_site, dump_data in tick_data['dump_site_states'].items():
            self.place_image(dump_data['position'], self.img_dump, zoom=DUMP_SITE_SCALE)
            self.ax.text(dump_data['position'][0], dump_data['position'][1] - 0.05, dump_site, ha='center', fontsize=5)

        # 绘制铲车
        for shovel, shovel_data in tick_data['shovel_states'].items():
            self.place_image(shovel_data['position'], self.img_shovel, zoom=SHOVEL_SCALE)
            self.ax.text(shovel_data['position'][0], shovel_data['position'][1] - 0.03, shovel, ha='center', fontsize=3)

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
            self.ax.text(truck_data['position'][0], truck_data['position'][1] - 0.035, truck, ha='center', fontsize=3, color='black')

            # 手动创建用于图例的代理艺术家（proxy artists）
            legend_elements = [
                (self.img_mine, '装载区'),
                (self.img_shovel, '铲车'),
                (self.img_truck_initing, '矿车'),
                (self.img_dump, '卸载区')
            ]

            # 添加图例
            self.ax.legend(
                [plt.Line2D([0], [0], linestyle='none', marker='_', alpha=0)] * len(legend_elements),
                [label for _, label in legend_elements],
                handler_map={plt.Line2D: ImageHandler(self.img_mine, LOAD_SITE_SCALE)},  # 指定图例处理器
                loc='upper right', fontsize=5
            )
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
    path = "/Users/mac/PycharmProjects/truck_shovel_mix/sisymines_project/sisymines/test/junk/results/MINE:北露天矿_ALGO:NaiveDispatcher_TIME:2024-01-19 17:36:35.json"
    visualizer = VisualGrapher(path)
    visualizer.create_animation(output_path='output.gif')
