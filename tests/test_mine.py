"""Tests for Mine — initialization, simulation lifecycle, and monitoring."""
import pytest
import numpy as np

from openmines.src.mine import Mine
from openmines.src.road import Road
from openmines.src.dispatcher import BaseDispatcher


class TestMineInitialization:
    """Test Mine construction and component assembly."""

    def test_mine_has_name(self, mini_mine):
        assert mini_mine.name == "TestMine"

    def test_mine_has_load_sites(self, mini_mine):
        assert len(mini_mine.load_sites) == 2

    def test_mine_has_dump_sites(self, mini_mine):
        assert len(mini_mine.dump_sites) == 3

    def test_mine_has_trucks(self, mini_mine):
        assert len(mini_mine.trucks) == 3

    def test_mine_has_road(self, mini_mine):
        assert mini_mine.road is not None

    def test_mine_has_charging_site(self, mini_mine):
        assert mini_mine.charging_site is not None


class TestMineStartRequirements:
    """Test that Mine.start() enforces all prerequisites."""

    def test_start_without_dispatcher_raises(self, mini_mine):
        """Mine.start() should fail if no dispatcher is set."""
        with pytest.raises(AssertionError, match="dispatcher"):
            mini_mine.start(total_time=10)

    def test_start_without_road_raises(self):
        mine = Mine("EmptyMine")
        with pytest.raises(AssertionError):
            mine.start(total_time=10)


class TestMineSimulation:
    """Test Mine simulation produces correct results."""

    def test_simulation_produces_ticks(self, mini_mine_with_dispatcher):
        """Simulation should return a non-empty ticks dictionary."""
        ticks = mini_mine_with_dispatcher.start(total_time=30)
        assert isinstance(ticks, dict)
        assert len(ticks) > 0

    def test_simulation_produces_tons(self, mini_mine_with_dispatcher):
        """After simulation, total produced_tons should be > 0."""
        mini_mine_with_dispatcher.start(total_time=60)
        assert mini_mine_with_dispatcher.produce_tons > 0

    def test_simulation_records_service_count(self, mini_mine_with_dispatcher):
        """After simulation, service_count should be > 0."""
        mini_mine_with_dispatcher.start(total_time=60)
        assert mini_mine_with_dispatcher.service_count > 0

    def test_status_dict_populated(self, mini_mine_with_dispatcher):
        """Mine.status should have entries after simulation."""
        mini_mine_with_dispatcher.start(total_time=30)
        status = mini_mine_with_dispatcher.status
        assert len(status) > 0
        # Should have "cur" key
        assert "cur" in status

    def test_status_contains_expected_fields(self, mini_mine_with_dispatcher):
        """Each status entry should contain the expected KPI fields."""
        mini_mine_with_dispatcher.start(total_time=30)
        cur = mini_mine_with_dispatcher.status["cur"]
        expected_fields = [
            "produced_tons", "service_count", "truck_count",
            "working_truck_count", "waiting_truck_count",
            "moving_truck_count",
        ]
        for field in expected_fields:
            assert field in cur, f"Missing field: {field}"

    def test_truck_count_matches(self, mini_mine_with_dispatcher):
        """Status truck_count should match actual truck count."""
        mini_mine_with_dispatcher.start(total_time=30)
        cur = mini_mine_with_dispatcher.status["cur"]
        assert cur["truck_count"] == len(mini_mine_with_dispatcher.trucks)


class TestMineRoadStatus:
    """Test road status tracking during simulation."""

    def test_road_status_populated_after_sim(self, mini_mine_with_dispatcher):
        """road.road_status should be populated after simulation."""
        mini_mine_with_dispatcher.start(total_time=30)
        road_status = mini_mine_with_dispatcher.road.road_status
        assert road_status is not None
        assert len(road_status) > 0

    def test_road_status_has_correct_keys(self, mini_mine_with_dispatcher):
        """Road status should contain keys for all valid routes."""
        mini_mine_with_dispatcher.start(total_time=30)
        road_status = mini_mine_with_dispatcher.road.road_status
        # Should have charging->load, load->dump, and dump->load routes
        # Check at least one load->dump route exists
        found_l2d = False
        for key in road_status:
            src, dst = key
            if "LoadSite" in src and "DumpSite" in dst:
                found_l2d = True
                break
        assert found_l2d, "No load-to-dump route found in road_status"


class TestMineDestLookup:
    """Test Mine's destination lookup methods."""

    def test_get_dest_index_by_name_load(self, mini_mine):
        idx = mini_mine.get_dest_index_by_name("LoadSite-0")
        assert idx == 0

    def test_get_dest_index_by_name_dump(self, mini_mine):
        idx = mini_mine.get_dest_index_by_name("DumpSite-0")
        # Dump sites are indexed after load sites: num_load_sites + dump_index
        assert idx == len(mini_mine.load_sites) + 0  # = 2

    def test_get_dest_index_by_name_unknown(self, mini_mine):
        idx = mini_mine.get_dest_index_by_name("NonexistentSite")
        assert idx is None

    def test_get_dest_obj_by_name(self, mini_mine):
        obj = mini_mine.get_dest_obj_by_name("DumpSite-1")
        assert obj is not None
        assert obj.name == "DumpSite-1"

    def test_get_dest_obj_by_name_charging(self, mini_mine):
        obj = mini_mine.get_dest_obj_by_name("TestChargingSite")
        assert obj is not None
        assert obj.name == "TestChargingSite"

    def test_get_service_vehicle_by_name(self, mini_mine):
        shovel = mini_mine.get_service_vehicle_by_name("Shovel-0")
        assert shovel is not None
        assert shovel.name == "Shovel-0"
