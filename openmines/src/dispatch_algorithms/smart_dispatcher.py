"""
SmartDispatcher: Enhanced dynamic dispatch based on ShortestTrip.

Improvements over ShortestTripDispatcher:
1. Dump site selection considers return trip quality (not just dump time)
2. Separates "trucks on road" from "trucks in queue" to avoid double-counting
3. Uses individual shovel/dumper availability instead of site-level averages
4. Accounts for truck capacity differences in queue estimation
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

    def _road_trucks_to_load(self, mine, ls_idx):
        """Trucks currently on road heading to this load site."""
        ls = mine.load_sites[ls_idx]
        total_cap = 0.0
        for t in mine.trucks:
            if (t.status == "moving" and t.target_location is not None
                    and isinstance(t.target_location, LoadSite)
                    and t.target_location.name == ls.name):
                total_cap += t.truck_capacity
        return total_cap

    def _road_trucks_to_dump(self, mine, ds_idx):
        """Count trucks currently on road heading to this dump site."""
        ds = mine.dump_sites[ds_idx]
        count = 0
        for t in mine.trucks:
            if (t.status == "moving" and t.target_location is not None
                    and isinstance(t.target_location, DumpSite)
                    and t.target_location.name == ds.name):
                count += 1
        return count

    def _available_shovels(self, load_site):
        """Count shovels not under maintenance."""
        return sum(1 for s in load_site.shovel_list if not s.repair)

    def _load_site_score(self, truck, mine, ls_idx, travel_dist):
        """
        Score a load site: lower = better.
        Returns estimated time from departure to finish loading.
        """
        ls = mine.load_sites[ls_idx]
        speed = truck.truck_speed
        travel_time = 60.0 * travel_dist / speed

        # Productivity: only count non-broken shovels
        active_shovels = self._available_shovels(ls)
        if active_shovels == 0:
            return float('inf')  # site is down

        productivity = sum(
            s.shovel_tons / s.shovel_cycle_time
            for s in ls.shovel_list if not s.repair
        )
        productivity = max(productivity, 0.01)

        # Service time for this truck at one shovel
        service_time = truck.truck_capacity / (productivity / active_shovels)

        # Queue estimation from the site's parking lot (already computed by framework)
        queue_wait = ls.estimated_queue_wait_time

        # Additional wait from trucks on road (not yet in queue)
        road_cap = self._road_trucks_to_load(mine, ls_idx)
        road_wait = road_cap / productivity

        # Total estimated time
        arrival = travel_time
        queue_done = queue_wait + road_wait
        wait_on_arrival = max(0, queue_done - arrival)

        return travel_time + wait_on_arrival + service_time

    def _dump_site_score(self, truck, mine, from_ls_idx, ds_idx):
        """
        Score a dump site: lower = better.
        Returns estimated time from departure to finish unloading.
        """
        ds = mine.dump_sites[ds_idx]
        speed = truck.truck_speed
        dist = mine.road.l2d_road_matrix[from_ls_idx, ds_idx]
        travel_time = 60.0 * dist / speed

        num_dumpers = len(ds.dumper_list)
        dump_time = ds.dumper_list[0].dump_time if ds.dumper_list else 1.0

        queue_wait = ds.estimated_queue_wait_time
        road_count = self._road_trucks_to_dump(mine, ds_idx)
        road_wait = (road_count / max(num_dumpers, 1)) * dump_time

        arrival = travel_time
        queue_done = queue_wait + road_wait
        wait_on_arrival = max(0, queue_done - arrival)

        return travel_time + wait_on_arrival + dump_time

    def _best_return_load_time(self, truck, mine, ds_idx):
        """Estimate the best load site reachable from this dump site."""
        best = float('inf')
        for ls_idx in range(len(mine.load_sites)):
            dist = mine.road.d2l_road_matrix[ls_idx, ds_idx]
            score = self._load_site_score(truck, mine, ls_idx, dist)
            if score < best:
                best = score
        return best

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        """Choose load site with minimum (travel + queue + service) time."""
        best_idx = 0
        best_score = float('inf')

        for ls_idx in range(len(mine.load_sites)):
            dist = mine.road.charging_to_load[ls_idx]
            score = self._load_site_score(truck, mine, ls_idx, dist)
            if score < best_score:
                best_score = score
                best_idx = ls_idx

        return best_idx

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        Choose dump site minimizing (dump_time + alpha * return_time).
        The return_time term prevents sending trucks to dump sites with
        terrible return routes, even if the dump itself is fast.
        """
        current_location = truck.current_location
        ls_idx = mine.load_sites.index(current_location)

        best_idx = 0
        best_score = float('inf')

        for ds_idx in range(len(mine.dump_sites)):
            dump_score = self._dump_site_score(truck, mine, ls_idx, ds_idx)
            return_score = self._best_return_load_time(truck, mine, ds_idx)
            # Weight: dump matters more (immediate), return is discounted
            score = dump_score + 0.5 * return_score
            if score < best_score:
                best_score = score
                best_idx = ds_idx

        return best_idx

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        """Choose load site with minimum (travel + queue + service) time."""
        current_location = truck.current_location
        ds_idx = mine.dump_sites.index(current_location)

        best_idx = 0
        best_score = float('inf')

        for ls_idx in range(len(mine.load_sites)):
            dist = mine.road.d2l_road_matrix[ls_idx, ds_idx]
            score = self._load_site_score(truck, mine, ls_idx, dist)
            if score < best_score:
                best_score = score
                best_idx = ls_idx

        return best_idx
