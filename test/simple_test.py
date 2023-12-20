import pathlib
import sys
# add the sisymines package to the path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.absolute()))

from sisymines.src.mine import Mine
from sisymines.src.truck import Truck
from sisymines.src.road import Road
from sisymines.src.charging_site import ChargingSite
from sisymines.src.load_site import LoadSite, Shovel
from sisymines.src.dump_site import DumpSite, Dumper
from sisymines.src.dispatcher import BaseDispatcher

from sisymines.src.dispatch_algorithms.naive_dispatch import NaiveDispatcher


def main():
    # init mine
    mine = Mine("北露天矿")

    # init dispatcher
    dispatcher = NaiveDispatcher()
    mine.add_dispatcher(dispatcher)

    # init charging site and trucks
    charging_site = ChargingSite("北露天矿-充电站")
    for i in range(30):
        charging_site.add_truck(Truck(name=f"星火-油卡{i+1}", truck_capacity=30, truck_speed=25))
    for i in range(20):
        charging_site.add_truck(Truck(name=f"抻录-电卡{i+1}", truck_capacity=120, truck_speed=10))
    for i in range(20):
        charging_site.add_truck(Truck(name=f"官方-卡车{i+1}", truck_capacity=50, truck_speed=25))

    # init load_site and shovel
    load_site_1 = LoadSite("北露天矿-装载点1")
    load_site_2 = LoadSite("北露天矿-装载点2")
    load_site_3 = LoadSite("北露天矿-装载点3")

    load_site_1.add_shovel(Shovel(name="北露天矿-装载点1-铲车1", shovel_tons=0.5, shovel_cycle_time=1))
    #load_site_1.add_shovel(Shovel(name="北露天矿-装载点1-铲车2", shovel_tons=2, shovel_cycle_time=2))

    load_site_2.add_shovel(Shovel(name="北露天矿-装载点2-铲车1", shovel_tons=0.5, shovel_cycle_time=1))
    #load_site_2.add_shovel(Shovel(name="北露天矿-装载点2-铲车2", shovel_tons=0.5, shovel_cycle_time=2))

    load_site_3.add_shovel(Shovel(name="北露天矿-装载点3-铲车1", shovel_tons=10, shovel_cycle_time=3))
    #load_site_3.add_shovel(Shovel(name="北露天矿-装载点3-铲车2", shovel_tons=10, shovel_cycle_time=3))

    load_site_1.show_shovels()
    load_site_2.show_shovels()
    load_site_3.show_shovels()

    # init dump_site and dumper
    dump_site_1 = DumpSite("北露天矿-卸载点1")
    dump_site_2 = DumpSite("北露天矿-卸载点2")
    dump_site_3 = DumpSite("北露天矿-卸载点3")

    for i in range(10):
        dump_site_1.add_dumper(Dumper(name=f"北露天矿-卸载点1-点位{i}", dumper_cycle_time=1))
        dump_site_2.add_dumper(Dumper(name=f"北露天矿-卸载点2-点位{i}", dumper_cycle_time=1))
        dump_site_3.add_dumper(Dumper(name=f"北露天矿-卸载点3-点位{i}", dumper_cycle_time=1))
    dump_site_1.show_dumpers()
    dump_site_2.show_dumpers()
    dump_site_3.show_dumpers()

    # init road
    from sisymines.src.data.example_data import road_3_3,charging_to_load_3_3
    road = Road(road_matrix=road_3_3, charging_to_load_road_matrix=charging_to_load_3_3)

    # add all to mine
    mine.add_charging_site(charging_site)
    mine.add_load_site(load_site_1)
    mine.add_load_site(load_site_2)
    mine.add_load_site(load_site_3)
    mine.add_dump_site(dump_site_1)
    mine.add_dump_site(dump_site_2)
    mine.add_dump_site(dump_site_3)
    mine.add_road(road)

    # test
    mine.start(total_time=60*8)

if __name__ == "__main__":
    main()




