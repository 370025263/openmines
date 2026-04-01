"""Tests for Dispatchers — contract tests ensuring all dispatchers obey the interface.

Every dispatcher must:
1. Return valid integer indices within the correct range
2. Track call counts and timing
3. Not crash when given valid inputs

We parametrize over all "simple" dispatchers (excluding RL, LLM, PPO, TabuOptimizer
which require special setup).
"""
import pytest
import time

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.dispatch_algorithms.naive_dispatcher import NaiveDispatcher
from openmines.src.dispatch_algorithms.random_dispatcher import RandomDispatcher
from openmines.src.dispatch_algorithms.nearest_dispatcher import NearestDispatcher
from openmines.src.dispatch_algorithms.shortest_trip_dispatcher import ShortestTripDispatcher
from openmines.src.dispatch_algorithms.fixed_group_dispatcher import FixedGroupDispatcher


# Dispatchers that can be tested without special dependencies
# (SQDispatcher excluded: imports openai unnecessarily)
SIMPLE_DISPATCHERS = [
    NaiveDispatcher,
    RandomDispatcher,
    NearestDispatcher,
    ShortestTripDispatcher,
    FixedGroupDispatcher,
]


@pytest.fixture(params=SIMPLE_DISPATCHERS, ids=lambda d: d.__name__)
def dispatcher_and_mine(request, mini_mine):
    """Parametrized fixture: yields (dispatcher, mine) for each dispatcher type."""
    dispatcher_cls = request.param
    dispatcher = dispatcher_cls()
    mini_mine.add_dispatcher(dispatcher)
    # Re-set truck dispatcher references
    for truck in mini_mine.trucks:
        truck.dispatcher = dispatcher
    return dispatcher, mini_mine


class TestDispatcherContract:
    """All dispatchers must satisfy these contracts."""

    def test_init_order_returns_valid_index(self, dispatcher_and_mine):
        """give_init_order must return an int in [0, num_load_sites)."""
        dispatcher, mine = dispatcher_and_mine
        truck = mine.trucks[0]
        truck.current_location = mine.charging_site

        result = dispatcher.give_init_order(truck=truck, mine=mine)

        assert isinstance(result, (int, np.integer)) or (isinstance(result, float) and result == int(result)), \
            f"Expected int, got {type(result)}"
        result = int(result)
        assert 0 <= result < len(mine.load_sites), \
            f"init_order {result} out of range [0, {len(mine.load_sites)})"

    def test_haul_order_returns_valid_index(self, dispatcher_and_mine):
        """give_haul_order must return an int in [0, num_dump_sites)."""
        dispatcher, mine = dispatcher_and_mine
        truck = mine.trucks[0]
        truck.current_location = mine.load_sites[0]

        result = dispatcher.give_haul_order(truck=truck, mine=mine)

        result = int(result)
        assert 0 <= result < len(mine.dump_sites), \
            f"haul_order {result} out of range [0, {len(mine.dump_sites)})"

    def test_back_order_returns_valid_index(self, dispatcher_and_mine):
        """give_back_order must return an int in [0, num_load_sites)."""
        dispatcher, mine = dispatcher_and_mine
        truck = mine.trucks[0]
        truck.current_location = mine.dump_sites[0]

        result = dispatcher.give_back_order(truck=truck, mine=mine)

        result = int(result)
        assert 0 <= result < len(mine.load_sites), \
            f"back_order {result} out of range [0, {len(mine.load_sites)})"

    def test_order_count_increments(self, dispatcher_and_mine):
        """Each order call should increment the corresponding counter."""
        dispatcher, mine = dispatcher_and_mine
        truck = mine.trucks[0]

        truck.current_location = mine.charging_site
        dispatcher.give_init_order(truck=truck, mine=mine)
        assert dispatcher.init_order_count == 1

        truck.current_location = mine.load_sites[0]
        dispatcher.give_haul_order(truck=truck, mine=mine)
        assert dispatcher.haul_order_count == 1

        truck.current_location = mine.dump_sites[0]
        dispatcher.give_back_order(truck=truck, mine=mine)
        assert dispatcher.back_order_count == 1

        assert dispatcher.total_order_count == 3

    def test_order_time_tracked(self, dispatcher_and_mine):
        """total_order_time should be >= 0 after calls."""
        dispatcher, mine = dispatcher_and_mine
        truck = mine.trucks[0]
        truck.current_location = mine.charging_site
        dispatcher.give_init_order(truck=truck, mine=mine)

        assert dispatcher.total_order_time >= 0


class TestNaiveDispatcher:
    """NaiveDispatcher always returns 0."""

    def test_always_returns_zero(self, mini_mine):
        from openmines.src.dispatch_algorithms.naive_dispatcher import NaiveDispatcher
        d = NaiveDispatcher()
        mini_mine.add_dispatcher(d)

        truck = mini_mine.trucks[0]

        truck.current_location = mini_mine.charging_site
        assert int(d.give_init_order(truck=truck, mine=mini_mine)) == 0

        truck.current_location = mini_mine.load_sites[0]
        assert int(d.give_haul_order(truck=truck, mine=mini_mine)) == 0

        truck.current_location = mini_mine.dump_sites[0]
        assert int(d.give_back_order(truck=truck, mine=mini_mine)) == 0


class TestNearestDispatcher:
    """NearestDispatcher should return the closest site."""

    def test_init_returns_nearest_load_site(self, mini_mine):
        """ChargingSite -> LoadSites: distances are [3.0, 5.0], so nearest is index 0."""
        from openmines.src.dispatch_algorithms.nearest_dispatcher import NearestDispatcher
        d = NearestDispatcher()
        mini_mine.add_dispatcher(d)

        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.charging_site
        result = int(d.give_init_order(truck=truck, mine=mini_mine))
        assert result == 0, "Nearest load site from charging should be index 0 (distance=3.0)"

    def test_haul_returns_nearest_dump_site(self, mini_mine):
        """LoadSite-1 -> DumpSites: l2d[1] = [7.0, 3.0, 9.0], so nearest is index 1."""
        from openmines.src.dispatch_algorithms.nearest_dispatcher import NearestDispatcher
        d = NearestDispatcher()
        mini_mine.add_dispatcher(d)

        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.load_sites[1]
        result = int(d.give_haul_order(truck=truck, mine=mini_mine))
        assert result == 1, "Nearest dump site from LoadSite-1 should be index 1 (distance=3.0)"


class TestBaseDispatcherAbstract:
    """BaseDispatcher should raise NotImplementedError for unimplemented methods."""

    def test_give_init_order_raises(self, mini_mine):
        d = BaseDispatcher()
        mini_mine.add_dispatcher(d)
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.charging_site
        with pytest.raises(NotImplementedError):
            d.give_init_order(truck=truck, mine=mini_mine)

    def test_give_haul_order_raises(self, mini_mine):
        d = BaseDispatcher()
        mini_mine.add_dispatcher(d)
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.load_sites[0]
        with pytest.raises(NotImplementedError):
            d.give_haul_order(truck=truck, mine=mini_mine)

    def test_give_back_order_raises(self, mini_mine):
        d = BaseDispatcher()
        mini_mine.add_dispatcher(d)
        truck = mini_mine.trucks[0]
        truck.current_location = mini_mine.dump_sites[0]
        with pytest.raises(NotImplementedError):
            d.give_back_order(truck=truck, mine=mini_mine)


# Need numpy for integer type checking
import numpy as np
