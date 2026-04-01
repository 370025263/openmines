"""
SmartDispatcher: A full-cycle-aware dynamic dispatch algorithm.

Key ideas beyond existing algorithms:
1. Full round-trip lookahead: when choosing a load site, also considers the best
   dump site reachable from there (and vice versa), optimizing total cycle time.
2. Capacity-weighted queue estimation: weights incoming trucks by their capacity
   relative to site productivity, not just count.
3. Road event awareness: penalizes routes going through jammed/repaired roads.
4. Multi-shovel parallelism: accounts for the number of available shovels/dumpers
   to estimate true service throughput.
5. Adaptive load balancing: combines time-optimal greedy with queue-balancing
   to prevent starvation of high-productivity sites.
"""
from __future__ import annotations
import numpy as np

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite
from openmines.src.dump_site import DumpSite


class SmartDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "SmartDispatcher"

    def _get_road_jam_penalty(self, mine, start, end) -> float:
        """Estimate extra travel time (minutes) from active jam events on a road."""
        penalty = 0.0
        for event_key, event in mine.random_event_pool.event_set.items():
            if event.event_type == "RoadEvent:jam":
                info = event.info
                if (info.get("start_location") == start.name and
                        info.get("end_location") == end.name):
                    penalty += info.get("jam_delay", 0.5)
            elif event.event_type == "RoadEvent:repair":
                info = event.info
                if (info.get("start_location") == start.name and
                        info.get("end_location") == end.name):
                    penalty += info.get("punish_distance", 0.0) * 60.0 / 25.0  # extra km → minutes
        return penalty

    def _count_incoming_capacity(self, mine, target, target_type) -> float:
        """Sum up truck_capacity of all trucks heading to a target location."""
        total = 0.0
        for t in mine.trucks:
            if t.target_location is not None and t.status == "moving":
                if isinstance(t.target_location, target_type):
                    if t.target_location.name == target.name:
                        total += t.truck_capacity
        return total

    def _estimate_load_time(self, truck, load_site, mine, from_location, is_charging=False):
        """
        Estimate total time from current location to load_site, including:
        - Travel time
        - Queue wait (capacity-weighted)
        - Loading service time
        """
        speed = truck.truck_speed  # km/h
        now = mine.env.now

        # Travel distance
        if is_charging:
            ls_idx = mine.load_sites.index(load_site)
            dist = mine.road.charging_to_load[ls_idx]
        elif isinstance(from_location, DumpSite):
            ds_idx = mine.dump_sites.index(from_location)
            ls_idx = mine.load_sites.index(load_site)
            dist = mine.road.d2l_road_matrix[ls_idx, ds_idx]
        else:
            ls_idx = mine.load_sites.index(load_site)
            dist = 0

        travel_time = 60.0 * dist / speed  # minutes

        # Road jam penalty
        if is_charging:
            jam_penalty = self._get_road_jam_penalty(mine, mine.charging_site, load_site)
        elif isinstance(from_location, DumpSite):
            jam_penalty = self._get_road_jam_penalty(mine, from_location, load_site)
        else:
            jam_penalty = 0

        travel_time += jam_penalty

        # Queue estimation: existing queue wait + incoming truck capacity / productivity
        queue_wait = load_site.estimated_queue_wait_time
        incoming_cap = self._count_incoming_capacity(mine, load_site, LoadSite)
        productivity = max(load_site.load_site_productivity, 0.01)
        num_shovels = max(len(load_site.shovel_list), 1)
        incoming_wait = incoming_cap / productivity

        # Service time for this truck
        service_time = truck.truck_capacity / (productivity / num_shovels)

        # Effective wait: max(travel_time, queue_wait + incoming_wait) - travel_time gives wait-on-arrival
        arrival_time = now + travel_time
        queue_finish_time = now + queue_wait + incoming_wait
        effective_wait = max(0, queue_finish_time - arrival_time)

        return travel_time + effective_wait + service_time

    def _estimate_dump_time(self, truck, dump_site, mine, from_load_site):
        """
        Estimate total time from a load site to dump_site, including:
        - Travel time
        - Queue wait
        - Unloading time
        """
        speed = truck.truck_speed
        now = mine.env.now

        ls_idx = mine.load_sites.index(from_load_site)
        ds_idx = mine.dump_sites.index(dump_site)
        dist = mine.road.l2d_road_matrix[ls_idx, ds_idx]
        travel_time = 60.0 * dist / speed

        # Road jam penalty
        jam_penalty = self._get_road_jam_penalty(mine, from_load_site, dump_site)
        travel_time += jam_penalty

        # Queue estimation
        queue_wait = dump_site.estimated_queue_wait_time
        incoming_count = sum(
            1 for t in mine.trucks
            if t.target_location is not None and t.status == "moving"
            and isinstance(t.target_location, DumpSite)
            and t.target_location.name == dump_site.name
        )
        num_dumpers = max(len(dump_site.dumper_list), 1)
        dump_time = dump_site.dumper_list[0].dump_time if dump_site.dumper_list else 1.0
        incoming_wait = (incoming_count / num_dumpers) * dump_time

        arrival_time = now + travel_time
        queue_finish_time = now + queue_wait + incoming_wait
        effective_wait = max(0, queue_finish_time - arrival_time)

        return travel_time + effective_wait + dump_time

    def _best_dump_time_for_load_site(self, truck, load_site, mine):
        """Find the minimum dump time from a load site across all dump sites."""
        best = float('inf')
        for ds in mine.dump_sites:
            t = self._estimate_dump_time(truck, ds, mine, load_site)
            if t < best:
                best = t
        return best

    def _best_load_time_for_dump_site(self, truck, dump_site, mine):
        """Find the minimum load time from a dump site across all load sites."""
        best = float('inf')
        for ls in mine.load_sites:
            t = self._estimate_load_time(truck, ls, mine, dump_site)
            if t < best:
                best = t
        return best

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        """Choose load site minimizing (travel + wait + load + best_dump_time)."""
        best_idx = 0
        best_score = float('inf')

        for i, ls in enumerate(mine.load_sites):
            load_time = self._estimate_load_time(truck, ls, mine, None, is_charging=True)
            dump_time = self._best_dump_time_for_load_site(truck, ls, mine)
            score = load_time + dump_time
            if score < best_score:
                best_score = score
                best_idx = i

        return best_idx

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        """Choose dump site minimizing (travel + wait + dump + best_return_load_time)."""
        current_location = truck.current_location
        best_idx = 0
        best_score = float('inf')

        for i, ds in enumerate(mine.dump_sites):
            dump_time = self._estimate_dump_time(truck, ds, mine, current_location)
            return_time = self._best_load_time_for_dump_site(truck, ds, mine)
            score = dump_time + return_time
            if score < best_score:
                best_score = score
                best_idx = i

        return best_idx

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        """Choose load site minimizing (travel + wait + load + best_dump_time)."""
        current_location = truck.current_location
        best_idx = 0
        best_score = float('inf')

        for i, ls in enumerate(mine.load_sites):
            load_time = self._estimate_load_time(truck, ls, mine, current_location)
            dump_time = self._best_dump_time_for_load_site(truck, ls, mine)
            score = load_time + dump_time
            if score < best_score:
                best_score = score
                best_idx = i

        return best_idx
