"""Tests for Road — distance calculation, road events, and status tracking.

Uses a NON-SQUARE distance matrix (2 load sites x 3 dump sites) to catch
index-order bugs. Asymmetric l2d vs d2l values catch direction confusion.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock

from openmines.src.road import Road
from openmines.src.load_site import LoadSite
from openmines.src.dump_site import DumpSite
from openmines.src.charging_site import ChargingSite
from tests.conftest import L2D_MATRIX, D2L_MATRIX, CHARGING_TO_LOAD


class TestRoadConstruction:
    """Test Road object construction and properties."""

    def test_road_stores_matrices(self):
        road = Road(L2D_MATRIX.copy(), D2L_MATRIX.copy(), CHARGING_TO_LOAD.copy())
        np.testing.assert_array_equal(road.l2d_road_matrix, L2D_MATRIX)
        np.testing.assert_array_equal(road.d2l_road_matrix, D2L_MATRIX)
        assert road.charging_to_load == CHARGING_TO_LOAD

    def test_load_site_num_from_matrix_rows(self):
        road = Road(L2D_MATRIX.copy(), D2L_MATRIX.copy(), CHARGING_TO_LOAD.copy())
        assert road.load_site_num == 2

    def test_dump_site_num_from_matrix_cols(self):
        road = Road(L2D_MATRIX.copy(), D2L_MATRIX.copy(), CHARGING_TO_LOAD.copy())
        assert road.dump_site_num == 3

    def test_custom_road_event_params(self):
        params = {"lambda_repair": 0.01, "mu_repair_duration": 30}
        road = Road(L2D_MATRIX.copy(), D2L_MATRIX.copy(), CHARGING_TO_LOAD.copy(),
                    road_event_params=params)
        assert road.lambda_repair == 0.01
        assert road.mu_repair_duration == 30


class TestRoadGetDistance:
    """Test distance lookup for all site type combinations.

    The mini_mine fixture has:
    - 2 LoadSites: LoadSite-0 (index 0), LoadSite-1 (index 1)
    - 3 DumpSites: DumpSite-0 (index 0), DumpSite-1 (index 1), DumpSite-2 (index 2)
    - l2d[0][1] = 8.0 (LoadSite-0 -> DumpSite-1)
    - d2l[1][0] = 8.0 (DumpSite-0 -> LoadSite-1)
    """

    def test_load_to_dump_distance(self, mini_mine):
        """LoadSite-0 -> DumpSite-1 should be l2d[0][1] = 8.0."""
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.load_sites[0]
        dist = mini_mine.road.get_distance(truck, mini_mine.dump_sites[1], enable_event=False)
        assert dist == pytest.approx(8.0)

    def test_load_to_dump_distance_second_load(self, mini_mine):
        """LoadSite-1 -> DumpSite-0 should be l2d[1][0] = 7.0."""
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.load_sites[1]
        dist = mini_mine.road.get_distance(truck, mini_mine.dump_sites[0], enable_event=False)
        assert dist == pytest.approx(7.0)

    def test_dump_to_load_distance(self, mini_mine):
        """DumpSite-0 -> LoadSite-0.

        Per code convention, d2l_road_matrix[load_idx][dump_idx],
        so d2l[0][0] = 6.0 for DumpSite-0 -> LoadSite-0.
        """
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.dump_sites[0]
        dist = mini_mine.road.get_distance(truck, mini_mine.load_sites[0], enable_event=False)
        assert dist == pytest.approx(6.0)

    def test_dump_to_load_distance_cross(self, mini_mine):
        """DumpSite-2 -> LoadSite-1.

        d2l[1][2] = 10.0 (load_idx=1, dump_idx=2).
        """
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.dump_sites[2]
        dist = mini_mine.road.get_distance(truck, mini_mine.load_sites[1], enable_event=False)
        assert dist == pytest.approx(10.0)

    def test_charging_to_load_distance(self, mini_mine):
        """ChargingSite -> LoadSite-1 should be 5.0."""
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.charging_site
        dist = mini_mine.road.get_distance(truck, mini_mine.load_sites[1], enable_event=False)
        assert dist == pytest.approx(5.0)

    def test_asymmetric_distance(self, mini_mine):
        """l2d and d2l should give different distances for same site pair."""
        truck = mini_mine.trucks[0]

        # LoadSite-0 -> DumpSite-0 = l2d[0][0] = 5.0
        truck.current_location = mini_mine.load_sites[0]
        l2d_dist = mini_mine.road.get_distance(truck, mini_mine.dump_sites[0], enable_event=False)

        # DumpSite-0 -> LoadSite-0 = d2l[0][0] = 6.0
        truck.current_location = mini_mine.dump_sites[0]
        d2l_dist = mini_mine.road.get_distance(truck, mini_mine.load_sites[0], enable_event=False)

        assert l2d_dist != d2l_dist, (
            "l2d and d2l distances should differ for asymmetric road network"
        )

    def test_same_type_raises(self, mini_mine):
        """Distance query between same site types should raise."""
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.load_sites[0]
        with pytest.raises(AssertionError):
            mini_mine.road.get_distance(truck, mini_mine.load_sites[1], enable_event=False)

    def test_all_distances_non_negative(self, mini_mine):
        """All distance matrix values should be >= 0."""
        assert np.all(mini_mine.road.l2d_road_matrix >= 0)
        assert np.all(mini_mine.road.d2l_road_matrix >= 0)
        assert all(d >= 0 for d in mini_mine.road.charging_to_load)


class TestRoadTruckOnRoad:
    """Test truck-on-road detection."""

    def test_no_trucks_on_empty_road(self, mini_mine):
        """No moving trucks means empty result."""
        result = mini_mine.road.truck_on_road(
            mini_mine.load_sites[0], mini_mine.dump_sites[0]
        )
        assert result == []

    def test_detects_moving_truck(self, mini_mine):
        """A truck with status='moving' and matching start/end should be detected."""
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.load_sites[0]
        truck.target_location = mini_mine.dump_sites[0]
        truck.status = "moving"

        result = mini_mine.road.truck_on_road(
            mini_mine.load_sites[0], mini_mine.dump_sites[0]
        )
        assert len(result) == 1
        assert result[0].name == "Truck-1"

    def test_ignores_non_moving_truck(self, mini_mine):
        """A truck that is not 'moving' should not be detected."""
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.load_sites[0]
        truck.target_location = mini_mine.dump_sites[0]
        truck.status = "loading"

        result = mini_mine.road.truck_on_road(
            mini_mine.load_sites[0], mini_mine.dump_sites[0]
        )
        assert result == []

    def test_ignores_truck_on_different_road(self, mini_mine):
        """A truck on a different route should not be detected."""
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.load_sites[0]
        truck.target_location = mini_mine.dump_sites[1]
        truck.status = "moving"

        result = mini_mine.road.truck_on_road(
            mini_mine.load_sites[0], mini_mine.dump_sites[0]
        )
        assert result == []
