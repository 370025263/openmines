"""Tests for Truck — movement, loading, unloading, and vehicle availability.

These tests run actual SimPy processes to verify truck behavior under simulation.
"""
import pytest
import simpy
import numpy as np

from openmines.src.truck import Truck
from openmines.src.load_site import Shovel
from openmines.src.dump_site import Dumper


class TestTruckConstruction:
    """Test Truck initialization."""

    def test_default_attributes(self):
        t = Truck("T1", truck_capacity=50, truck_speed=25)
        assert t.name == "T1"
        assert t.truck_capacity == 50
        assert t.truck_speed == 25
        assert t.truck_load == 0
        assert t.service_count == 0
        assert t.status == "idle"

    def test_custom_speed(self):
        t = Truck("T2", truck_capacity=30, truck_speed=40)
        assert t.truck_speed == 40


class TestTruckMovement:
    """Test truck movement mechanics via SimPy simulation."""

    def test_move_duration_matches_distance_and_speed(self, mini_mine):
        """Movement time should be (distance / speed) * 60 minutes."""
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.load_sites[0]
        target = mini_mine.dump_sites[0]
        distance = 5.0  # km
        speed = truck.truck_speed  # 25 km/h
        expected_duration = (distance / speed) * 60  # 12 minutes

        def run():
            yield mini_mine.env.process(truck.move(target, distance))

        mini_mine.env.process(run())
        mini_mine.env.run()

        assert mini_mine.env.now == pytest.approx(expected_duration, abs=0.5)

    def test_move_updates_current_location(self, mini_mine):
        """After movement, current_location should be the target."""
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.load_sites[0]
        target = mini_mine.dump_sites[0]

        def run():
            yield mini_mine.env.process(truck.move(target, 5.0))

        mini_mine.env.process(run())
        mini_mine.env.run()

        if truck.status != "unrepairable":
            assert truck.current_location == target

    def test_move_records_event(self, mini_mine):
        """Movement should record a haul/unhaul/init event in the truck's event_pool."""
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.load_sites[0]
        target = mini_mine.dump_sites[0]

        def run():
            yield mini_mine.env.process(truck.move(target, 5.0))

        mini_mine.env.process(run())
        mini_mine.env.run()

        # Should have recorded a "haul" event
        events = truck.event_pool.get_even_by_type("haul")
        assert len(events) >= 1
        last_event = events[-1]
        assert last_event.info["target_location"] == target.name
        assert last_event.info["distance"] == 5.0

    def test_move_with_manual_speed(self, mini_mine):
        """Manual speed override should affect travel time."""
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.load_sites[0]
        target = mini_mine.dump_sites[0]
        distance = 10.0
        manual_speed = 50  # km/h
        expected_duration = (distance / manual_speed) * 60  # 12 minutes

        def run():
            yield mini_mine.env.process(truck.move(target, distance, manual_speed=manual_speed))

        mini_mine.env.process(run())
        mini_mine.env.run()

        if truck.status != "unrepairable":
            assert mini_mine.env.now == pytest.approx(expected_duration, abs=0.5)


class TestTruckLoadUnload:
    """Test loading and unloading mechanics."""

    def test_load_increases_truck_load(self, mini_mine):
        """After loading, truck_load should be approximately truck_capacity."""
        truck = mini_mine.trucks[0]  # capacity=30
        shovel = mini_mine.load_sites[0].shovel_list[0]  # 5 tons, 1.0 cycle

        def run():
            yield mini_mine.env.process(truck.load(shovel))

        mini_mine.env.process(run())
        mini_mine.env.run()

        # Load should be approximately capacity (with ±10% random variation)
        assert truck.truck_load > 0
        assert truck.truck_load == pytest.approx(truck.truck_capacity, rel=0.15)

    def test_load_time_depends_on_capacity_and_shovel(self, mini_mine):
        """Load time = (truck_capacity / shovel_tons) * shovel_cycle_time."""
        truck = mini_mine.trucks[0]  # capacity=30
        shovel = mini_mine.load_sites[0].shovel_list[0]  # 5 tons, 1.0 cycle
        expected_time = (30 / 5.0) * 1.0  # = 6.0 minutes

        def run():
            yield mini_mine.env.process(truck.load(shovel))

        mini_mine.env.process(run())
        mini_mine.env.run()

        assert mini_mine.env.now == pytest.approx(expected_time)

    def test_unload_resets_truck_load(self, mini_mine):
        """After unloading, truck_load should be 0."""
        truck = mini_mine.trucks[0]
        truck.truck_load = 30
        truck.first_order_time = 0
        dumper = mini_mine.dump_sites[0].dumper_list[0]

        def run():
            yield mini_mine.env.process(truck.unload(dumper))

        mini_mine.env.process(run())
        mini_mine.env.run()

        assert truck.truck_load == 0

    def test_unload_increments_service_count(self, mini_mine):
        """Each unload should increment truck's service_count by 1."""
        truck = mini_mine.trucks[0]
        truck.truck_load = 30
        truck.first_order_time = 0
        dumper = mini_mine.dump_sites[0].dumper_list[0]
        initial_count = truck.service_count

        def run():
            yield mini_mine.env.process(truck.unload(dumper))

        mini_mine.env.process(run())
        mini_mine.env.run()

        assert truck.service_count == initial_count + 1

    def test_unload_adds_to_dumper_tons(self, mini_mine):
        """Unloading should transfer truck_load to dumper_tons."""
        truck = mini_mine.trucks[0]
        truck.truck_load = 30
        truck.first_order_time = 0
        dumper = mini_mine.dump_sites[0].dumper_list[0]
        initial_dumper_tons = dumper.dumper_tons

        def run():
            yield mini_mine.env.process(truck.unload(dumper))

        mini_mine.env.process(run())
        mini_mine.env.run()

        assert dumper.dumper_tons == initial_dumper_tons + 30


class TestTruckVehicleAvailability:
    """Test vehicle breakdown and availability mechanics.

    Bug hypothesis: check_vehicle_availability() generates a new random value
    each call instead of comparing against a fixed breakdown time, which may
    cause vehicles to become unrepairable too easily.
    """

    def test_availability_returns_float_or_none(self, mini_mine):
        """check_vehicle_availability() should return a float (repair time) or None."""
        truck = mini_mine.trucks[0]
        result = truck.check_vehicle_availability()
        assert result is None or isinstance(result, (int, float))

    def test_most_trucks_survive_short_simulation(self, mini_mine_with_dispatcher):
        """In a short simulation, most trucks should NOT be unrepairable.

        If check_vehicle_availability is fundamentally broken, a large fraction
        of trucks would be unrepairable even in a short simulation.
        """
        mine = mini_mine_with_dispatcher
        mine.start(total_time=30)  # 30 minutes — very short

        unrepairable_count = sum(
            1 for t in mine.trucks if t.status == "unrepairable"
        )
        total = len(mine.trucks)

        # With 3 trucks and 30 min sim, we expect 0 or at most 1 unrepairable
        assert unrepairable_count <= 1, (
            f"{unrepairable_count}/{total} trucks unrepairable in 30-min sim — "
            "vehicle availability model may be broken"
        )

    def test_location_onehot_encoding(self, mini_mine):
        """get_location_onehot should return correct one-hot vector."""
        truck = mini_mine.trucks[0]

        # At charging site
        truck.current_location = mini_mine.charging_site
        onehot = truck.get_location_onehot()
        # Format: [charging] + [load_sites] + [dump_sites]
        # = [1] + [0, 0] + [0, 0, 0]
        assert onehot[0] == 1
        assert sum(onehot) == 1

        # At LoadSite-0
        truck.current_location = mini_mine.load_sites[0]
        onehot = truck.get_location_onehot()
        assert onehot[0] == 0  # not at charging
        assert onehot[1] == 1  # at load site 0
        assert sum(onehot) == 1

        # At DumpSite-1
        truck.current_location = mini_mine.dump_sites[1]
        onehot = truck.get_location_onehot()
        assert onehot[0] == 0  # not at charging
        assert onehot[4] == 1  # dump site 1 (index: 1 + 2 + 1 = 4)
        assert sum(onehot) == 1
