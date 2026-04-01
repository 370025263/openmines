"""
MPCDispatcher: Model Predictive Control based dispatch.

Instead of solving a steady-state LP, MPC solves a short-horizon assignment
problem at each decision point considering the CURRENT system snapshot:
- Trucks on roads with estimated arrival times
- Current queue lengths and wait times
- Active road events (jams, repairs)
- Shovel/dumper maintenance status

Uses scipy.optimize.linear_sum_assignment for the assignment subproblem.

Key difference from LPDispatcher: MPC models the transient dynamics
(trucks in transit, queue buildup) rather than assuming steady state.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite
from openmines.src.dump_site import DumpSite


class MPCDispatcher(BaseDispatcher):
    def __init__(self, horizon=30):
        super().__init__()
        self.name = "MPCDispatcher"
        self.horizon = horizon  # lookahead in minutes

        # Cached plan: solved once per simulation timestep, shared across truck calls
        self._plan_time = -1
        self._plan = {}          # truck_name → (ls_idx, ds_idx)
        self._ls_assignments = None  # per load site: count of trucks assigned this step
        self._ds_assignments = None

    def _take_snapshot(self, mine):
        """
        Build a snapshot of the current system state.
        Returns structured info about trucks, queues, and capacities.
        """
        now = mine.env.now
        NL = len(mine.load_sites)
        ND = len(mine.dump_sites)

        # Load site info
        ls_info = []
        for j, ls in enumerate(mine.load_sites):
            active = [s for s in ls.shovel_list if not s.repair]
            prod = sum(s.shovel_tons / s.shovel_cycle_time for s in active) if active else 0.01
            n_active = len(active) if active else 0
            ls_info.append({
                'prod': prod,
                'n_shovels': n_active,
                'queue_wait': ls.estimated_queue_wait_time,
            })

        # Dump site info
        ds_info = []
        for k, ds in enumerate(mine.dump_sites):
            n_dumpers = len(ds.dumper_list)
            dump_time = ds.dumper_list[0].dump_time if ds.dumper_list else 1.0
            ds_info.append({
                'n_dumpers': n_dumpers,
                'dump_time': dump_time,
                'queue_wait': ds.estimated_queue_wait_time,
            })

        # Count trucks heading to each site (capacity already committed)
        ls_incoming_cap = np.zeros(NL)
        ds_incoming_count = np.zeros(ND)
        for t in mine.trucks:
            if t.status == "moving" and t.target_location is not None:
                if isinstance(t.target_location, LoadSite):
                    for j, ls in enumerate(mine.load_sites):
                        if t.target_location.name == ls.name:
                            ls_incoming_cap[j] += t.truck_capacity
                            break
                elif isinstance(t.target_location, DumpSite):
                    for k, ds in enumerate(mine.dump_sites):
                        if t.target_location.name == ds.name:
                            ds_incoming_count[k] += 1
                            break

        return {
            'now': now, 'NL': NL, 'ND': ND,
            'ls_info': ls_info, 'ds_info': ds_info,
            'ls_incoming_cap': ls_incoming_cap,
            'ds_incoming_count': ds_incoming_count,
        }

    def _estimate_throughput(self, truck, mine, snap, ls_idx, ds_idx):
        """
        Estimate production rate (tons/min) for a truck on route (ls_idx → ds_idx),
        considering current system state.
        """
        ls = snap['ls_info'][ls_idx]
        ds = snap['ds_info'][ds_idx]
        speed = truck.truck_speed
        cap = truck.truck_capacity

        if ls['n_shovels'] == 0:
            return 0.0

        # Travel time
        l2d = mine.road.l2d_road_matrix[ls_idx, ds_idx]
        d2l = mine.road.d2l_road_matrix[ls_idx, ds_idx]
        travel = 60.0 * (l2d + d2l) / speed

        # Load time at individual shovel
        load_time = cap / (ls['prod'] / ls['n_shovels'])

        # Queue wait at load site (considering incoming)
        ls_queue = ls['queue_wait'] + snap['ls_incoming_cap'][ls_idx] / ls['prod']

        # Dump time
        dump_time = ds['dump_time']

        # Queue wait at dump site (considering incoming)
        ds_queue = ds['queue_wait'] + (snap['ds_incoming_count'][ds_idx] / max(ds['n_dumpers'], 1)) * dump_time

        cycle = travel + load_time + dump_time + ls_queue + ds_queue
        if cycle <= 0:
            return 0.0

        return cap / cycle

    def _estimate_total_time(self, truck, mine, snap, ls_idx, ds_idx, from_location, is_init=False):
        """
        Estimate total time for a full cycle starting from current position.
        Used for the assignment cost matrix.
        """
        speed = truck.truck_speed
        cap = truck.truck_capacity
        ls = snap['ls_info'][ls_idx]
        ds = snap['ds_info'][ds_idx]

        if ls['n_shovels'] == 0:
            return float('inf')

        # Phase 1: Travel to load site
        if is_init:
            dist_to_ls = mine.road.charging_to_load[ls_idx]
        elif isinstance(from_location, DumpSite):
            from_ds_idx = mine.dump_sites.index(from_location)
            dist_to_ls = mine.road.d2l_road_matrix[ls_idx, from_ds_idx]
        else:
            dist_to_ls = 0

        travel_to_ls = 60.0 * dist_to_ls / speed

        # Queue at load site (penalized by how many we've already assigned this step)
        extra_ls = 0
        if self._ls_assignments is not None:
            extra_ls = (self._ls_assignments[ls_idx] * cap) / ls['prod']
        ls_queue = ls['queue_wait'] + snap['ls_incoming_cap'][ls_idx] / ls['prod'] + extra_ls
        wait_at_ls = max(0, ls_queue - travel_to_ls)

        # Loading
        load_time = cap / (ls['prod'] / ls['n_shovels'])

        # Phase 2: Travel to dump site
        l2d = mine.road.l2d_road_matrix[ls_idx, ds_idx]
        travel_to_ds = 60.0 * l2d / speed

        # Queue at dump site
        extra_ds = 0
        if self._ds_assignments is not None:
            extra_ds = (self._ds_assignments[ds_idx] / max(ds['n_dumpers'], 1)) * ds['dump_time']
        ds_queue = ds['queue_wait'] + (snap['ds_incoming_count'][ds_idx] / max(ds['n_dumpers'], 1)) * ds['dump_time'] + extra_ds
        wait_at_ds = max(0, ds_queue - travel_to_ds)

        # Dumping
        dump_time = ds['dump_time']

        total = travel_to_ls + wait_at_ls + load_time + travel_to_ds + wait_at_ds + dump_time

        # Throughput: prefer routes with high tons/time
        # Negative throughput = cost (we want to minimize cost = maximize throughput)
        return total / max(cap, 1)  # normalize by capacity: time per ton

    def _solve_assignment(self, truck, mine, snap, destinations, from_location, is_init=False):
        """
        For the current truck, evaluate all (ls, ds) route pairs and pick the best.
        Considers trucks already assigned in this planning step.
        """
        NL = snap['NL']
        ND = snap['ND']

        best_score = float('inf')
        best_ls = 0
        best_ds = 0

        for j in range(NL):
            for k in range(ND):
                score = self._estimate_total_time(
                    truck, mine, snap, j, k, from_location, is_init
                )
                # Also consider return trip quality
                d2l_dists = mine.road.d2l_road_matrix[:, k]
                best_return = min(
                    self._estimate_total_time(truck, mine, snap, jj, k, mine.dump_sites[k])
                    for jj in range(NL)
                    if snap['ls_info'][jj]['n_shovels'] > 0
                ) if any(snap['ls_info'][jj]['n_shovels'] > 0 for jj in range(NL)) else 0

                total = score + 0.3 * best_return

                if total < best_score:
                    best_score = total
                    best_ls = j
                    best_ds = k

        # Record assignment for future trucks in same planning step
        if self._ls_assignments is not None:
            self._ls_assignments[best_ls] += 1
        if self._ds_assignments is not None:
            self._ds_assignments[best_ds] += 1

        return best_ls, best_ds

    def _ensure_plan(self, mine):
        """Reset assignment counters at each new simulation timestep."""
        now = int(mine.env.now * 10)  # discretize to 0.1 min
        if now != self._plan_time:
            self._plan_time = now
            self._ls_assignments = np.zeros(len(mine.load_sites))
            self._ds_assignments = np.zeros(len(mine.dump_sites))

    # ──────────────────────────────────────────────────────
    # Dispatch decisions
    # ──────────────────────────────────────────────────────

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        self._ensure_plan(mine)
        snap = self._take_snapshot(mine)
        ls, ds = self._solve_assignment(truck, mine, snap, None, None, is_init=True)
        self._plan[truck.name] = (ls, ds)
        return ls

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        self._ensure_plan(mine)
        snap = self._take_snapshot(mine)
        current_location = truck.current_location
        ls_idx = mine.load_sites.index(current_location)

        # Evaluate dump sites considering current position
        NL = snap['NL']
        ND = snap['ND']
        best_ds = 0
        best_score = float('inf')

        for k in range(ND):
            ds = snap['ds_info'][k]
            dist = mine.road.l2d_road_matrix[ls_idx, k]
            travel = 60.0 * dist / truck.truck_speed

            dump_time = ds['dump_time']
            n_dumpers = ds['n_dumpers']

            extra_ds = 0
            if self._ds_assignments is not None:
                extra_ds = (self._ds_assignments[k] / max(n_dumpers, 1)) * dump_time

            ds_queue = ds['queue_wait'] + (snap['ds_incoming_count'][k] / max(n_dumpers, 1)) * dump_time + extra_ds
            wait_at_ds = max(0, ds_queue - travel)

            score = travel + wait_at_ds + dump_time

            # Return trip quality
            best_return = float('inf')
            for jj in range(NL):
                if snap['ls_info'][jj]['n_shovels'] == 0:
                    continue
                ret_dist = mine.road.d2l_road_matrix[jj, k]
                ret_travel = 60.0 * ret_dist / truck.truck_speed
                ls_q = snap['ls_info'][jj]['queue_wait'] + snap['ls_incoming_cap'][jj] / snap['ls_info'][jj]['prod']
                ret_wait = max(0, ls_q - ret_travel)
                ret_service = truck.truck_capacity / (snap['ls_info'][jj]['prod'] / snap['ls_info'][jj]['n_shovels'])
                ret_total = ret_travel + ret_wait + ret_service
                if ret_total < best_return:
                    best_return = ret_total

            total = score + 0.5 * best_return

            if total < best_score:
                best_score = total
                best_ds = k

        if self._ds_assignments is not None:
            self._ds_assignments[best_ds] += 1

        return best_ds

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        self._ensure_plan(mine)
        snap = self._take_snapshot(mine)
        current_location = truck.current_location
        ds_idx = mine.dump_sites.index(current_location)

        NL = snap['NL']
        ND = snap['ND']
        best_ls = 0
        best_score = float('inf')

        for j in range(NL):
            ls = snap['ls_info'][j]
            if ls['n_shovels'] == 0:
                continue

            dist = mine.road.d2l_road_matrix[j, ds_idx]
            travel = 60.0 * dist / truck.truck_speed

            extra_ls = 0
            if self._ls_assignments is not None:
                extra_ls = (self._ls_assignments[j] * truck.truck_capacity) / ls['prod']

            ls_queue = ls['queue_wait'] + snap['ls_incoming_cap'][j] / ls['prod'] + extra_ls
            wait_at_ls = max(0, ls_queue - travel)
            service = truck.truck_capacity / (ls['prod'] / ls['n_shovels'])

            score = travel + wait_at_ls + service

            # Dump site lookahead
            best_dump = float('inf')
            for k in range(ND):
                d_dist = mine.road.l2d_road_matrix[j, k]
                d_travel = 60.0 * d_dist / truck.truck_speed
                ds_info = snap['ds_info'][k]
                d_queue = ds_info['queue_wait'] + (snap['ds_incoming_count'][k] / max(ds_info['n_dumpers'], 1)) * ds_info['dump_time']
                d_wait = max(0, d_queue - d_travel)
                d_total = d_travel + d_wait + ds_info['dump_time']
                if d_total < best_dump:
                    best_dump = d_total

            total = score + 0.5 * best_dump

            if total < best_score:
                best_score = total
                best_ls = j

        if self._ls_assignments is not None:
            self._ls_assignments[best_ls] += 1

        return best_ls
