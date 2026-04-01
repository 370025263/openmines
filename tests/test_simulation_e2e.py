"""End-to-end smoke tests — run full simulations with real config files.

These tests verify the entire simulation pipeline works correctly,
from config loading through simulation execution to output generation.
"""
import pytest
import json
import pathlib

from openmines.src.cli.run import run_dispatch_sim, load_config
from openmines.src.dispatch_algorithms.naive_dispatcher import NaiveDispatcher
from openmines.src.dispatch_algorithms.nearest_dispatcher import NearestDispatcher
from openmines.src.dispatch_algorithms.random_dispatcher import RandomDispatcher


# Path to the short config file
CONFIG_PATH = pathlib.Path(__file__).parent.parent / "openmines" / "src" / "conf" / "north_pit_mine_short.json"


class TestEndToEndSimulation:
    """Run full simulation with real config files."""

    @pytest.fixture(autouse=True)
    def check_config_exists(self):
        if not CONFIG_PATH.exists():
            pytest.skip(f"Config file not found: {CONFIG_PATH}")

    def test_naive_dispatcher_completes(self):
        """NaiveDispatcher should complete simulation without errors."""
        ticks = run_dispatch_sim(NaiveDispatcher(), str(CONFIG_PATH))
        assert isinstance(ticks, dict)
        assert len(ticks) > 0

        # Check mine_states exist
        last_mine_state = None
        for tick_data in ticks.values():
            if isinstance(tick_data, dict) and "mine_states" in tick_data:
                last_mine_state = tick_data["mine_states"]
        assert last_mine_state is not None

    def test_nearest_dispatcher_completes(self):
        """NearestDispatcher should complete simulation without errors."""
        ticks = run_dispatch_sim(NearestDispatcher(), str(CONFIG_PATH))
        assert isinstance(ticks, dict)

        last_mine_state = None
        for tick_data in ticks.values():
            if isinstance(tick_data, dict) and "mine_states" in tick_data:
                last_mine_state = tick_data["mine_states"]
        assert last_mine_state is not None

    def test_simulation_with_sufficient_time_produces_tons(self):
        """With enough sim time, trucks should complete at least one cycle.

        The short config has trucks with capacity=30 and shovels with 0.5 tons/cycle,
        so loading takes (30/0.5)*1 = 60 minutes. With travel time, a full cycle
        needs at least ~90 min. We use a longer config override.
        """
        import copy
        config = load_config(str(CONFIG_PATH))
        config_extended = copy.deepcopy(config)
        config_extended["sim_time"] = 480  # 8 hours

        ticks = run_dispatch_sim(NaiveDispatcher(), config_extended)
        last_mine_state = None
        for tick_data in ticks.values():
            if isinstance(tick_data, dict) and "mine_states" in tick_data:
                last_mine_state = tick_data["mine_states"]
        assert last_mine_state is not None
        assert last_mine_state["produced_tons"] > 0, (
            "With 8 hours of simulation, at least one truck should complete a cycle"
        )

    def test_tick_data_has_summary(self):
        """Tick output should include a summary section."""
        ticks = run_dispatch_sim(NaiveDispatcher(), str(CONFIG_PATH))
        assert "summary" in ticks

    def test_all_trucks_have_events(self):
        """Every truck should have generated at least one event during simulation."""
        ticks = run_dispatch_sim(NaiveDispatcher(), str(CONFIG_PATH))

        # Check truck states exist in tick data
        found_trucks = set()
        for tick_data in ticks.values():
            if isinstance(tick_data, dict) and "truck_states" in tick_data:
                for truck_name in tick_data["truck_states"]:
                    found_trucks.add(truck_name)

        config = load_config(str(CONFIG_PATH))
        expected_truck_count = sum(t["count"] for t in config["charging_site"]["trucks"])
        assert len(found_trucks) == expected_truck_count

    def test_load_config_returns_dict(self):
        """load_config should return a valid config dictionary."""
        config = load_config(str(CONFIG_PATH))
        assert isinstance(config, dict)
        assert "mine" in config
        assert "charging_site" in config
        assert "load_sites" in config
        assert "dump_sites" in config
        assert "road" in config


class TestMineStartRlRewardBug:
    """Test for Bug #3: start_rl sparse mode calling dense reward function.

    This is a code-level test verifying the correct reward function is referenced.
    """

    def test_sparse_reward_code_path(self):
        """Verify that the sparse reward branch exists and is distinct from dense.

        We check the source code since running RL requires multiprocessing setup.
        """
        import inspect
        from openmines.src.mine import Mine

        source = inspect.getsource(Mine.start_rl)

        # The sparse branch should call _get_reward_sparse, not _get_reward_dense
        # Find the sparse branch
        lines = source.split('\n')
        found_sparse_branch = False
        sparse_calls_dense = False
        for i, line in enumerate(lines):
            if 'reward_mode == "sparse"' in line:
                found_sparse_branch = True
                # Check the next line for which function is called
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if '_get_reward_dense' in next_line:
                        sparse_calls_dense = True

        assert found_sparse_branch, "No sparse reward branch found in start_rl"
        if sparse_calls_dense:
            pytest.fail(
                "BUG: start_rl sparse mode calls _get_reward_dense instead of "
                "_get_reward_sparse (mine.py line ~459)"
            )
