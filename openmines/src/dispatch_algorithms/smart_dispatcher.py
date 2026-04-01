"""
SmartDispatcher v4: Arrival-time-aware + coordinated dispatch.

Two key improvements over previous versions:
1. Only counts trucks arriving BEFORE this truck in queue estimation
2. Coordinated dispatch: when a truck picks a site, a "virtual load" is added
   to that site's queue estimate for subsequent trucks within the same time
   window, preventing multiple trucks from making identical greedy choices
   (the herd effect).
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
        # Coordination state: tracks recent dispatch decisions
        self._recent_ls_load = {}  # ls_idx → (time, accumulated_cap)
        self._recent_ds_load = {}  # ds_idx → (time, accumulated_count)
        self._coord_window = 1.0   # coordination window in sim minutes

    def _get_coord_load_ls(self, ls_idx, now):
        """Get accumulated virtual load at a load site from recent dispatches."""
        entry = self._recent_ls_load.get(ls_idx)
        if entry and (now - entry[0]) < self._coord_window:
            return entry[1]
        return 0.0

    def _get_coord_load_ds(self, ds_idx, now):
        """Get accumulated virtual truck count at dump site from recent dispatches."""
        entry = self._recent_ds_load.get(ds_idx)
        if entry and (now - entry[0]) < self._coord_window:
            return entry[1]
        return 0

    def _record_ls_dispatch(self, ls_idx, truck_cap, now):
        """Record that a truck was dispatched to this load site."""
        entry = self._recent_ls_load.get(ls_idx)
        if entry and (now - entry[0]) < self._coord_window:
            self._recent_ls_load[ls_idx] = (entry[0], entry[1] + truck_cap)
        else:
            self._recent_ls_load[ls_idx] = (now, truck_cap)

    def _record_ds_dispatch(self, ds_idx, now):
        """Record that a truck was dispatched to this dump site."""
        entry = self._recent_ds_load.get(ds_idx)
        if entry and (now - entry[0]) < self._coord_window:
            self._recent_ds_load[ds_idx] = (entry[0], entry[1] + 1)
        else:
            self._recent_ds_load[ds_idx] = (now, 1)

    def _trucks_arriving_before(self, mine, target, target_type, my_arrival_time):
        """Count incoming trucks whose ETA is before ours."""
        count = 0
        total_cap = 0.0
        now = mine.env.now
        for t in mine.trucks:
            if (t.status == "moving" and t.target_location is not None
                    and isinstance(t.target_location, target_type)
                    and t.target_location.name == target.name):
                their_eta = now
                try:
                    for etype in ["haul", "unhaul", "init"]:
                        evt = t.event_pool.get_last_event(type=etype, strict=False)
                        if evt and evt.info.get("est_end_time"):
                            their_eta = evt.info["est_end_time"]
                            break
                except (KeyError, IndexError, AssertionError):
                    their_eta = now
                if their_eta <= my_arrival_time:
                    count += 1
                    total_cap += t.truck_capacity
        return count, total_cap

    def _load_site_score(self, truck, mine, ls_idx, travel_dist):
        ls = mine.load_sites[ls_idx]
        speed = truck.truck_speed
        my_travel = 60.0 * travel_dist / speed
        my_arrival = mine.env.now + my_travel
        now = mine.env.now

        active_shovels = [s for s in ls.shovel_list if not s.repair]
        if not active_shovels:
            return float('inf')
        n_active = len(active_shovels)
        productivity = sum(s.shovel_tons / s.shovel_cycle_time for s in active_shovels)
        service_time = truck.truck_capacity / (productivity / n_active)

        # Queue decays as we travel
        queue_at_arrival = max(0, ls.estimated_queue_wait_time - my_travel)

        # Only trucks arriving before us matter
        _, incoming_cap = self._trucks_arriving_before(mine, ls, LoadSite, my_arrival)
        incoming_wait = incoming_cap / productivity

        # Coordination: add virtual load from trucks dispatched in this window
        coord_cap = self._get_coord_load_ls(ls_idx, now)
        coord_wait = coord_cap / productivity

        return my_travel + queue_at_arrival + incoming_wait + coord_wait + service_time

    def _dump_site_score(self, truck, mine, ls_idx, ds_idx):
        ds = mine.dump_sites[ds_idx]
        dist = mine.road.l2d_road_matrix[ls_idx, ds_idx]
        speed = truck.truck_speed
        my_travel = 60.0 * dist / speed
        my_arrival = mine.env.now + my_travel
        now = mine.env.now

        n_dumpers = len(ds.dumper_list)
        dump_time = ds.dumper_list[0].dump_time if ds.dumper_list else 1.0

        queue_at_arrival = max(0, ds.estimated_queue_wait_time - my_travel)

        count_before, _ = self._trucks_arriving_before(mine, ds, DumpSite, my_arrival)
        incoming_wait = (count_before / max(n_dumpers, 1)) * dump_time

        # Coordination: virtual load from recent dispatches
        coord_count = self._get_coord_load_ds(ds_idx, now)
        coord_wait = (coord_count / max(n_dumpers, 1)) * dump_time

        return my_travel + queue_at_arrival + incoming_wait + coord_wait + dump_time

    def _best_return_time(self, truck, mine, ds_idx):
        best = float('inf')
        for j in range(len(mine.load_sites)):
            dist = mine.road.d2l_road_matrix[j, ds_idx]
            score = self._load_site_score(truck, mine, j, dist)
            if score < best:
                best = score
        return best

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        best_idx = 0
        best_score = float('inf')
        for j in range(len(mine.load_sites)):
            dist = mine.road.charging_to_load[j]
            score = self._load_site_score(truck, mine, j, dist)
            if score < best_score:
                best_score = score
                best_idx = j
        self._record_ls_dispatch(best_idx, truck.truck_capacity, mine.env.now)
        return best_idx

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        ls_idx = mine.load_sites.index(truck.current_location)
        best_idx = 0
        best_score = float('inf')
        for k in range(len(mine.dump_sites)):
            dump_score = self._dump_site_score(truck, mine, ls_idx, k)
            return_score = self._best_return_time(truck, mine, k)
            score = dump_score + 0.5 * return_score
            if score < best_score:
                best_score = score
                best_idx = k
        self._record_ds_dispatch(best_idx, mine.env.now)
        return best_idx

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        ds_idx = mine.dump_sites.index(truck.current_location)
        best_idx = 0
        best_score = float('inf')
        for j in range(len(mine.load_sites)):
            dist = mine.road.d2l_road_matrix[j, ds_idx]
            score = self._load_site_score(truck, mine, j, dist)
            if score < best_score:
                best_score = score
                best_idx = j
        self._record_ls_dispatch(best_idx, truck.truck_capacity, mine.env.now)
        return best_idx
