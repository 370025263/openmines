"""Tests for EventPool — storage, query, and retrieval functionality."""
import pytest
from openmines.src.utils.event import Event, EventPool


class TestEventBasics:
    """Test Event object behavior."""

    def test_event_stores_attributes(self):
        e = Event(1.0, "test", "description", info={"key": "val"})
        assert e.time_stamp == 1.0
        assert e.event_type == "test"
        assert e.desc == "description"
        assert e.info == {"key": "val"}

    def test_event_ordering(self):
        e1 = Event(1.0, "a", "first")
        e2 = Event(2.0, "b", "second")
        assert e1 < e2
        assert not e2 < e1

    def test_event_str_repr(self):
        e = Event(1.0, "test", "desc")
        assert "test" in str(e)
        assert "1.0" in repr(e)


class TestEventPoolAddAndQuery:
    """Test adding events and querying by type."""

    def test_add_and_get_by_type(self, event_pool):
        event_pool.add_event(Event(1.0, "haul", "truck haul"))
        event_pool.add_event(Event(2.0, "haul", "truck haul 2"))
        event_pool.add_event(Event(3.0, "init", "truck init"))

        haul_events = event_pool.get_even_by_type("haul")
        assert len(haul_events) == 2
        init_events = event_pool.get_even_by_type("init")
        assert len(init_events) == 1

    def test_get_by_type_returns_sorted(self, event_pool):
        event_pool.add_event(Event(5.0, "move", "late move"))
        event_pool.add_event(Event(1.0, "move", "early move"))
        event_pool.add_event(Event(3.0, "move", "mid move"))

        moves = event_pool.get_even_by_type("move")
        timestamps = [e.time_stamp for e in moves]
        assert timestamps == sorted(timestamps)

    def test_get_nonexistent_type_returns_empty(self, event_pool):
        result = event_pool.get_even_by_type("nonexistent")
        assert result == []

    def test_get_by_desc(self, event_pool):
        event_pool.add_event(Event(1.0, "a", "truck moved to LoadSite"))
        event_pool.add_event(Event(2.0, "b", "truck moved to DumpSite"))
        event_pool.add_event(Event(3.0, "c", "shovel started"))

        result = event_pool.get_even_by_desc("truck moved")
        assert len(result) == 2

    def test_empty_pool_queries_dont_crash(self):
        pool = EventPool()
        assert pool.get_even_by_type("any") == []
        assert pool.get_even_by_desc("any") == []
        assert pool.get_event_by_time(10.0) == []
        assert pool.get_event_by_time_range(0, 10) == []


class TestEventPoolTimeQueries:
    """Test time-based event retrieval."""

    def test_get_by_time_backward(self, event_pool):
        for t in [1.0, 3.0, 5.0, 7.0, 9.0]:
            event_pool.add_event(Event(t, "test", f"event at {t}"))

        result = event_pool.get_event_by_time(5.5, mode="backward")
        timestamps = [e.time_stamp for e in result]
        assert all(t <= 5.5 for t in timestamps)
        assert len(result) == 3  # 1.0, 3.0, 5.0

    def test_get_by_time_future(self, event_pool):
        for t in [1.0, 3.0, 5.0, 7.0, 9.0]:
            event_pool.add_event(Event(t, "test", f"event at {t}"))

        result = event_pool.get_event_by_time(5.0, mode="future")
        timestamps = [e.time_stamp for e in result]
        assert all(t > 5.0 for t in timestamps)
        assert len(result) == 2  # 7.0, 9.0

    def test_get_by_time_range(self, event_pool):
        for t in [1.0, 3.0, 5.0, 7.0, 9.0]:
            event_pool.add_event(Event(t, "test", f"event at {t}"))

        result = event_pool.get_event_by_time_range(3.0, 7.0)
        timestamps = [e.time_stamp for e in result]
        assert all(3.0 <= t <= 7.0 for t in timestamps)
        assert len(result) == 3  # 3.0, 5.0, 7.0

    def test_get_by_time_backward_returns_sorted(self, event_pool):
        # Add in non-sorted order
        event_pool.add_event(Event(5.0, "a", ""))
        event_pool.add_event(Event(1.0, "b", ""))
        event_pool.add_event(Event(3.0, "c", ""))

        result = event_pool.get_event_by_time(10.0)
        timestamps = [e.time_stamp for e in result]
        assert timestamps == sorted(timestamps)


class TestEventPoolDuplicateTimestamps:
    """Test handling of events with same timestamp."""

    def test_duplicate_timestamps_both_stored(self, event_pool):
        event_pool.add_event(Event(1.0, "typeA", "first"))
        event_pool.add_event(Event(1.0, "typeB", "second"))

        # Both should be queryable
        a_events = event_pool.get_even_by_type("typeA")
        b_events = event_pool.get_even_by_type("typeB")
        assert len(a_events) == 1
        assert len(b_events) == 1


class TestEventPoolLastEvent:
    """Test get_last_event and update_last_info."""

    def test_get_last_event_strict(self, event_pool):
        event_pool.add_event(Event(1.0, "haul", "first"))
        event_pool.add_event(Event(2.0, "haul", "second"))

        last = event_pool.get_last_event("haul", strict=True)
        assert last.time_stamp == 2.0

    def test_get_last_event_non_strict(self, event_pool):
        event_pool.add_event(Event(1.0, "haul", "first"))
        event_pool.add_event(Event(2.0, "init", "second"))

        last = event_pool.get_last_event("haul", strict=False)
        assert last.time_stamp == 1.0

    def test_update_last_info_non_strict(self, event_pool):
        event_pool.add_event(Event(1.0, "wait", "", info={"wait_time": 0}))
        event_pool.add_event(Event(2.0, "load", "", info={"load_time": 10}))

        event_pool.update_last_info("wait", {"wait_time": 99}, strict=False)
        wait_events = event_pool.get_even_by_type("wait")
        assert wait_events[0].info["wait_time"] == 99

    def test_clear_empties_pool(self, event_pool):
        event_pool.add_event(Event(1.0, "test", ""))
        event_pool.clear()
        assert event_pool.get_even_by_type("test") == []
        assert len(event_pool.event_set) == 0
