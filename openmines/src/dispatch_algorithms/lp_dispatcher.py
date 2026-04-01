"""
LPDispatcher: LP-optimal allocation + dynamic queue-aware online dispatch.

Two phases:
1. OFFLINE: Solve LP to find optimal truck→route allocation maximizing throughput
2. ONLINE: Use SmartDispatcher-style dynamic estimation for real-time decisions,
   biased by LP-optimal targets to reduce oscillation

Re-solves LP periodically to adapt to breakdowns and road events.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite
from openmines.src.dump_site import DumpSite


class LPDispatcher(BaseDispatcher):
    def __init__(self, reoptimize_interval=15, lp_bias=0.30):
        super().__init__()
        self.name = "LPDispatcher"
        self.reoptimize_interval = reoptimize_interval
        self.lp_bias = lp_bias  # discount factor for LP-preferred choices

        # LP state
        self._solved = False
        self._last_solve_time = -999
        self._truck_route = {}   # truck_name → (ls_idx, ds_idx)
        self._ls_target = {}     # ls_idx → target fraction of trucks
        self._num_ls = 0
        self._num_ds = 0

    # ──────────────────────────────────────────────────────
    # LP Solver
    # ──────────────────────────────────────────────────────

    def _cycle_time(self, cap, speed, ls, ls_idx, ds, ds_idx, mine):
        """Full round-trip cycle time in minutes."""
        l2d = mine.road.l2d_road_matrix[ls_idx, ds_idx]
        d2l = mine.road.d2l_road_matrix[ls_idx, ds_idx]
        travel = 60.0 * (l2d + d2l) / speed

        active = [s for s in ls.shovel_list if not s.repair]
        if not active:
            return float('inf')
        prod = sum(s.shovel_tons / s.shovel_cycle_time for s in active)
        load_time = cap / (prod / len(active))

        unload = ds.dumper_list[0].dump_time if ds.dumper_list else 1.0
        return travel + load_time + unload

    def _solve_lp(self, mine):
        trucks = mine.trucks
        NL = len(mine.load_sites)
        ND = len(mine.dump_sites)
        NT = len(trucks)
        NR = NL * ND
        NV = NT * NR
        self._num_ls = NL
        self._num_ds = ND

        # Precompute throughput[i,r] and trip_rate[i,r]
        throughput = np.zeros(NV)
        trip_rate = np.zeros(NV)

        for i, truck in enumerate(trucks):
            for j in range(NL):
                for k in range(ND):
                    r = j * ND + k
                    idx = i * NR + r
                    ct = self._cycle_time(
                        truck.truck_capacity, truck.truck_speed,
                        mine.load_sites[j], j, mine.dump_sites[k], k, mine
                    )
                    if ct > 0 and ct != float('inf'):
                        throughput[idx] = truck.truck_capacity / ct
                        trip_rate[idx] = 1.0 / ct

        # Objective: minimize -throughput
        c = -throughput

        # Equality: each truck allocated to exactly 1 unit of routes
        A_eq = np.zeros((NT, NV))
        for i in range(NT):
            A_eq[i, i * NR:(i + 1) * NR] = 1.0
        b_eq = np.ones(NT)

        # Inequality: load site capacity
        A_ub_list = []
        b_ub_list = []

        for j in range(NL):
            ls = mine.load_sites[j]
            active = [s for s in ls.shovel_list if not s.repair]
            prod = sum(s.shovel_tons / s.shovel_cycle_time for s in active) if active else 0.01

            row = np.zeros(NV)
            for i in range(NT):
                for k in range(ND):
                    row[i * NR + j * ND + k] = throughput[i * NR + j * ND + k]
            A_ub_list.append(row)
            b_ub_list.append(prod)

        # Inequality: dump site capacity
        for k in range(ND):
            ds = mine.dump_sites[k]
            n_dumpers = len(ds.dumper_list)
            unload = ds.dumper_list[0].dump_time if ds.dumper_list else 1.0
            cap_trucks_per_min = n_dumpers / unload

            row = np.zeros(NV)
            for i in range(NT):
                for j in range(NL):
                    row[i * NR + j * ND + k] = trip_rate[i * NR + j * ND + k]
            A_ub_list.append(row)
            b_ub_list.append(cap_trucks_per_min)

        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        bounds = [(0, 1)] * NV

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')

        if result.success:
            x = result.x
        else:
            x = np.ones(NV) / NR

        # Extract per-truck primary assignment
        self._truck_route = {}
        ls_truck_count = np.zeros(NL)

        for i, truck in enumerate(trucks):
            allocs = x[i * NR:(i + 1) * NR]
            best_r = np.argmax(allocs)
            j = best_r // ND
            k = best_r % ND
            self._truck_route[truck.name] = (j, k)
            ls_truck_count[j] += 1

        # Target load site fractions (for bias)
        total = ls_truck_count.sum()
        self._ls_target = {j: ls_truck_count[j] / total for j in range(NL)} if total > 0 else {}

        self._solved = True
        self._last_solve_time = mine.env.now

    def _maybe_reoptimize(self, mine):
        now = mine.env.now
        if not self._solved or (now - self._last_solve_time >= self.reoptimize_interval):
            self._solve_lp(mine)

    # ──────────────────────────────────────────────────────
    # Dynamic estimation (SmartDispatcher-style)
    # ──────────────────────────────────────────────────────

    def _load_score(self, truck, mine, ls_idx, dist):
        """Estimate time from departure to finish loading at ls_idx."""
        ls = mine.load_sites[ls_idx]
        speed = truck.truck_speed
        travel = 60.0 * dist / speed

        active = [s for s in ls.shovel_list if not s.repair]
        if not active:
            return float('inf')
        prod = sum(s.shovel_tons / s.shovel_cycle_time for s in active)
        service = truck.truck_capacity / (prod / len(active))

        queue_wait = ls.estimated_queue_wait_time
        incoming_cap = sum(
            t.truck_capacity for t in mine.trucks
            if t.name != truck.name and t.status == "moving"
            and t.target_location is not None
            and isinstance(t.target_location, LoadSite)
            and t.target_location.name == ls.name
        )
        incoming_wait = incoming_cap / prod
        wait_on_arrival = max(0, queue_wait + incoming_wait - travel)

        return travel + wait_on_arrival + service

    def _dump_score(self, truck, mine, ls_idx, ds_idx):
        """Estimate time from load site ls_idx to finish dumping at ds_idx."""
        ds = mine.dump_sites[ds_idx]
        dist = mine.road.l2d_road_matrix[ls_idx, ds_idx]
        travel = 60.0 * dist / truck.truck_speed

        n_dumpers = len(ds.dumper_list)
        dump_time = ds.dumper_list[0].dump_time if ds.dumper_list else 1.0

        queue_wait = ds.estimated_queue_wait_time
        incoming = sum(
            1 for t in mine.trucks
            if t.name != truck.name and t.status == "moving"
            and t.target_location is not None
            and isinstance(t.target_location, DumpSite)
            and t.target_location.name == ds.name
        )
        incoming_wait = (incoming / max(n_dumpers, 1)) * dump_time
        wait_on_arrival = max(0, queue_wait + incoming_wait - travel)

        return travel + wait_on_arrival + dump_time

    def _best_return_time(self, truck, mine, ds_idx):
        """Best load site reachable from dump site ds_idx."""
        best = float('inf')
        for j in range(len(mine.load_sites)):
            dist = mine.road.d2l_road_matrix[j, ds_idx]
            score = self._load_score(truck, mine, j, dist)
            if score < best:
                best = score
        return best

    # ──────────────────────────────────────────────────────
    # Dispatch decisions
    # ──────────────────────────────────────────────────────

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        For init: use LP assignment but stagger via dynamic scoring.
        Early trucks follow LP; as queues build, redirect dynamically.
        """
        self._maybe_reoptimize(mine)
        assigned_ls = self._truck_route.get(truck.name, (0, 0))[0]

        # Score all load sites dynamically
        scores = []
        for j in range(len(mine.load_sites)):
            dist = mine.road.charging_to_load[j]
            s = self._load_score(truck, mine, j, dist)
            # LP bias: discount the assigned load site
            if j == assigned_ls:
                s *= (1.0 - self.lp_bias)
            scores.append(s)

        return int(np.argmin(scores))

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        Dynamic dump selection with return-trip lookahead.
        LP bias applied to the assigned dump site.
        """
        self._maybe_reoptimize(mine)
        ls_idx = mine.load_sites.index(truck.current_location)
        assigned = self._truck_route.get(truck.name, (0, 0))
        assigned_ds = assigned[1] if assigned[0] == ls_idx else -1

        best_idx = 0
        best_score = float('inf')

        for k in range(len(mine.dump_sites)):
            d = self._dump_score(truck, mine, ls_idx, k)
            r = self._best_return_time(truck, mine, k)
            score = d + 0.5 * r
            # LP bias
            if k == assigned_ds:
                score *= (1.0 - self.lp_bias)
            if score < best_score:
                best_score = score
                best_idx = k

        return best_idx

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        Dynamic load site selection with LP bias on assigned site.
        """
        self._maybe_reoptimize(mine)
        ds_idx = mine.dump_sites.index(truck.current_location)
        assigned_ls = self._truck_route.get(truck.name, (0, 0))[0]

        best_idx = 0
        best_score = float('inf')

        for j in range(len(mine.load_sites)):
            dist = mine.road.d2l_road_matrix[j, ds_idx]
            s = self._load_score(truck, mine, j, dist)
            if j == assigned_ls:
                s *= (1.0 - self.lp_bias)
            if s < best_score:
                best_score = s
                best_idx = j

        return best_idx
