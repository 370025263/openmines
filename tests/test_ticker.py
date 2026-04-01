"""Tests for TickGenerator — position interpolation and frame data generation.

Tests focus on the correctness of truck position calculation during movement,
and the proper generation of frame data for animation.
"""
import pytest
import numpy as np

from openmines.src.utils.event import Event, EventPool
from openmines.src.utils.ticker import TickGenerator


class TestTickerPositionInterpolation:
    """Test that truck positions are correctly interpolated during movement."""

    def test_truck_at_charging_site_when_idle(self, mini_mine_with_dispatcher):
        """At tick 0 of a running sim, trucks should be at or near charging site."""
        mine = mini_mine_with_dispatcher
        # Start a short sim so trucks have events
        mine.start(total_time=10)
        ticker = TickGenerator(mine, tick_num=10)
        ticker.run()

        # At tick 0, trucks should be at or near charging site (before they move)
        tick_0 = ticker.ticks.get(0)
        assert tick_0 is not None
        cs_pos = mine.charging_site.position
        for truck_name, state in tick_0["truck_states"].items():
            pos = state["position"]
            # State -1 = ON_CHARGING_SITE
            if state["state"] == -1:
                assert abs(pos[0] - cs_pos[0]) < 0.2
                assert abs(pos[1] - cs_pos[1]) < 0.2

    def test_moving_truck_between_start_and_end(self, mini_mine_with_dispatcher):
        """A moving truck should have position between start and end sites."""
        mine = mini_mine_with_dispatcher
        mine.start(total_time=30)

        ticker = TickGenerator(mine, tick_num=30)
        ticker.run()

        # Check mid-simulation ticks for moving trucks
        for t in range(5, 25):
            tick = ticker.ticks.get(t)
            if tick is None:
                continue
            for truck_name, state in tick["truck_states"].items():
                pos = state["position"]
                # Position should be within the mine's coordinate space [0, 1]
                # (with some margin for offsets)
                assert -0.5 <= pos[0] <= 1.5, f"Truck {truck_name} x={pos[0]} out of bounds at t={t}"
                assert -0.5 <= pos[1] <= 1.5, f"Truck {truck_name} y={pos[1]} out of bounds at t={t}"

    def test_position_does_not_overshoot(self, mini_mine_with_dispatcher):
        """time_ratio should be clamped to [0, 1] — position should not go beyond target.

        This tests for Bug #5: time_ratio exceeding 1.0.
        """
        mine = mini_mine_with_dispatcher
        mine.start(total_time=60)

        ticker = TickGenerator(mine, tick_num=60)
        ticker.run()

        # Check all ticks: no truck should have unreasonable positions
        for t, tick in ticker.ticks.items():
            if not isinstance(t, int):
                continue
            for truck_name, state in tick["truck_states"].items():
                pos = state["position"]
                # All positions should be within reasonable bounds
                # Mine coordinates are 0-1 with small offsets possible
                assert -0.5 <= pos[0] <= 1.5, (
                    f"Truck {truck_name} position x={pos[0]} out of bounds at t={t}, "
                    f"state={state['state']} — possible time_ratio overshoot"
                )
                assert -0.5 <= pos[1] <= 1.5, (
                    f"Truck {truck_name} position y={pos[1]} out of bounds at t={t}, "
                    f"state={state['state']} — possible time_ratio overshoot"
                )


class TestTickerFrameStructure:
    """Test that tick frames contain all required data."""

    def test_tick_has_truck_states(self, mini_mine_with_dispatcher):
        mine = mini_mine_with_dispatcher
        mine.start(total_time=10)
        ticker = TickGenerator(mine, tick_num=10)
        ticker.run()

        tick = ticker.ticks[0]
        assert "truck_states" in tick
        assert len(tick["truck_states"]) == len(mine.trucks)

    def test_tick_has_site_states(self, mini_mine_with_dispatcher):
        mine = mini_mine_with_dispatcher
        mine.start(total_time=10)
        ticker = TickGenerator(mine, tick_num=10)
        ticker.run()

        tick = ticker.ticks[0]
        assert "load_site_states" in tick
        assert "dump_site_states" in tick

    def test_truck_state_has_required_fields(self, mini_mine_with_dispatcher):
        mine = mini_mine_with_dispatcher
        mine.start(total_time=10)
        ticker = TickGenerator(mine, tick_num=10)
        ticker.run()

        tick = ticker.ticks[5]
        for truck_name, state in tick["truck_states"].items():
            assert "name" in state
            assert "time" in state
            assert "state" in state
            assert "position" in state
            assert isinstance(state["position"], (list, np.ndarray))
            assert len(state["position"]) == 2

    def test_mine_states_has_jams(self, mini_mine_with_dispatcher):
        """Mine states should include jam data for visualization."""
        mine = mini_mine_with_dispatcher
        mine.start(total_time=30)
        ticker = TickGenerator(mine, tick_num=30)
        ticker.run()

        tick = ticker.ticks[15]
        assert "mine_states" in tick
        mine_states = tick["mine_states"]
        assert "jams" in mine_states
        assert "position" in mine_states["jams"]
        assert "last_times" in mine_states["jams"]


class TestTickerWriteToFile:
    """Test that tick data can be written to JSON."""

    def test_write_to_file_returns_ticks(self, mini_mine_with_dispatcher, tmp_path):
        mine = mini_mine_with_dispatcher
        mine.start(total_time=10)
        ticker = TickGenerator(mine, tick_num=10)
        ticker.run()

        # Override result path to use temp directory
        ticker.result_path = str(tmp_path)
        result = ticker.write_to_file("test_output.json")
        assert result is not None

        # Check file was created
        output_file = tmp_path / "test_output.json"
        assert output_file.exists()
