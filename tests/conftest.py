"""
Shared fixtures for OpenMines test suite.

Design choices:
- Uses a 2-LoadSite x 3-DumpSite NON-SQUARE matrix to catch index-order bugs
- Asymmetric l2d vs d2l values to catch direction confusion
- Small fleet (3 trucks) for fast test execution
- Deterministic seed for reproducibility
"""
import pytest
import numpy as np
import simpy

from openmines.src.mine import Mine
from openmines.src.truck import Truck
from openmines.src.charging_site import ChargingSite
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.dump_site import DumpSite, Dumper
from openmines.src.road import Road
from openmines.src.utils.event import Event, EventPool


# ---------------------------------------------------------------------------
# Tiny mine configuration — 2 load sites, 3 dump sites, 3 trucks
# ---------------------------------------------------------------------------

# l2d_road_matrix[load_idx][dump_idx]: 2 rows (load sites) x 3 cols (dump sites)
L2D_MATRIX = np.array([
    [5.0, 8.0, 12.0],   # LoadSite0 -> DumpSite0=5, DumpSite1=8, DumpSite2=12
    [7.0, 3.0, 9.0],    # LoadSite1 -> DumpSite0=7, DumpSite1=3, DumpSite2=9
])

# d2l_road_matrix: same shape as l2d (2x3) per current code convention
# d2l_road_matrix[load_idx][dump_idx] = distance from DumpSite(dump_idx) to LoadSite(load_idx)
D2L_MATRIX = np.array([
    [6.0, 9.0, 13.0],   # DumpSite{0,1,2} -> LoadSite0
    [8.0, 4.0, 10.0],   # DumpSite{0,1,2} -> LoadSite1
])

CHARGING_TO_LOAD = [3.0, 5.0]  # ChargingSite -> LoadSite0=3, LoadSite1=5


@pytest.fixture
def simpy_env():
    """A fresh SimPy environment."""
    return simpy.Environment()


@pytest.fixture
def event_pool():
    """A fresh EventPool."""
    return EventPool()


@pytest.fixture
def sample_events():
    """A list of sample events for testing."""
    return [
        Event(1.0, "init", "truck init", info={"name": "T1"}),
        Event(2.0, "haul", "truck haul", info={"name": "T1", "start_time": 2.0, "est_end_time": 5.0}),
        Event(3.0, "RoadEvent:jam", "jam on road", info={
            "start_location": "L1", "end_location": "D1",
            "jam_position": 0.5, "start_time": 3.0, "est_end_time": 6.0
        }),
        Event(4.0, "wait shovel", "waiting", info={"name": "T1", "shovel": "S1", "queue_index": 0}),
        Event(5.5, "get shovel", "loading", info={"name": "T1", "shovel": "S1"}),
    ]


def _build_mini_mine(seed=42):
    """Build a minimal but complete Mine for testing.

    Layout:
    - 1 ChargingSite at (0, 0)
    - 2 LoadSites at (0.1, 0.2) and (0.1, 0.6)
    - 3 DumpSites at (0.8, 0.2), (0.8, 0.5), (0.8, 0.8)
    - 3 Trucks with different capacities
    - NON-SQUARE distance matrix (2x3)
    """
    mine = Mine("TestMine", seed=seed)

    # Charging site with 3 trucks
    cs = ChargingSite("TestChargingSite", position=[0, 0])
    cs.add_truck(Truck("Truck-1", truck_capacity=30, truck_speed=25))
    cs.add_truck(Truck("Truck-2", truck_capacity=50, truck_speed=20))
    cs.add_truck(Truck("Truck-3", truck_capacity=40, truck_speed=25))

    # Load sites
    ls0 = LoadSite("LoadSite-0", position=[0.1, 0.2])
    ls0.add_shovel(Shovel("Shovel-0", shovel_tons=5.0, shovel_cycle_time=1.0, position_offset=[0.05, 0.05]))
    ls0.add_parkinglot(position_offset=[-0.05, 0.0], name="ParkingLot-L0")

    ls1 = LoadSite("LoadSite-1", position=[0.1, 0.6])
    ls1.add_shovel(Shovel("Shovel-1", shovel_tons=10.0, shovel_cycle_time=1.5, position_offset=[0.05, 0.05]))
    ls1.add_parkinglot(position_offset=[-0.05, 0.0], name="ParkingLot-L1")

    # Dump sites
    ds0 = DumpSite("DumpSite-0", position=[0.8, 0.2])
    ds0.add_dumper(Dumper("Dumper-0", dumper_cycle_time=1.0, position_offset=[0.0, 0.05]))
    ds0.add_parkinglot(position_offset=[0.05, 0.05], name="ParkingLot-D0")

    ds1 = DumpSite("DumpSite-1", position=[0.8, 0.5])
    ds1.add_dumper(Dumper("Dumper-1", dumper_cycle_time=1.0, position_offset=[0.0, 0.05]))
    ds1.add_parkinglot(position_offset=[0.05, 0.05], name="ParkingLot-D1")

    ds2 = DumpSite("DumpSite-2", position=[0.8, 0.8])
    ds2.add_dumper(Dumper("Dumper-2", dumper_cycle_time=1.0, position_offset=[0.0, 0.05]))
    ds2.add_parkinglot(position_offset=[0.05, 0.05], name="ParkingLot-D2")

    # Road (non-square: 2 load x 3 dump)
    road = Road(
        l2d_road_matrix=L2D_MATRIX.copy(),
        d2l_road_matrix=D2L_MATRIX.copy(),
        charging_to_load_road_matrix=CHARGING_TO_LOAD.copy(),
    )

    # Assemble
    mine.add_load_site(ls0)
    mine.add_load_site(ls1)
    mine.add_dump_site(ds0)
    mine.add_dump_site(ds1)
    mine.add_dump_site(ds2)
    mine.add_road(road)
    mine.add_charging_site(cs)

    # Ensure all trucks have loggers (normally set in truck.run())
    for truck in mine.trucks:
        truck.logger = mine.global_logger.get_logger("Truck")

    return mine


@pytest.fixture
def mini_mine():
    """A minimal complete Mine with non-square distance matrix."""
    return _build_mini_mine()


@pytest.fixture
def mini_mine_with_dispatcher():
    """A minimal complete Mine with NaiveDispatcher attached."""
    from openmines.src.dispatch_algorithms.naive_dispatcher import NaiveDispatcher
    mine = _build_mini_mine()
    mine.add_dispatcher(NaiveDispatcher())
    # Re-set truck dispatcher references (truck.set_env was called before dispatcher was added)
    for truck in mine.trucks:
        truck.dispatcher = mine.dispatcher
    return mine
