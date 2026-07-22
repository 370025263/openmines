"""
DualLPDispatcher: LPDispatcher enhanced with shadow prices and regret tracking.

Improvements over LPDispatcher:
1. Uses LP dual variables (shadow prices) to dynamically value each site's
   marginal capacity — high shadow price = binding constraint = avoid overloading
2. Tracks actual vs LP-target truck allocation per route and adjusts bias
   dynamically (regret-based) instead of using a fixed discount
3. Shadow prices are re-computed on each LP re-solve (every 15 min)
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite
from openmines.src.dump_site import DumpSite


class DualLPDispatcher(BaseDispatcher):
    def __init__(self, reoptimize_interval=15):
        super().__init__()
        self.name = "DualLPDispatcher"
        self.reoptimize_interval = reoptimize_interval

        # LP state
        self._solved = False
        self._last_solve_time = -999
        self._truck_route = {}    # truck_name → (ls_idx, ds_idx)
        self._num_ls = 0
        self._num_ds = 0

        # Shadow prices (dual variables)
        self._ls_shadow = {}      # ls_idx → shadow price (marginal value of capacity)
        self._ds_shadow = {}      # ds_idx → shadow price

        # Regret tracking: actual dispatches vs LP target per route
        self._route_target = {}   # (ls, ds) → target fraction
        self._route_actual = {}   # (ls, ds) → actual dispatch count
        self._total_dispatches = 0

    # ──────────────────────────────────────────────────────
    # LP Solver with Dual Extraction
    # ──────────────────────────────────────────────────────

    def _cycle_time(self, cap, speed, ls, ls_idx, ds, ds_idx, mine):
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

        c = -throughput

        A_eq = np.zeros((NT, NV))
        for i in range(NT):
            A_eq[i, i * NR:(i + 1) * NR] = 1.0
        b_eq = np.ones(NT)

        A_ub_list = []
        b_ub_list = []

        # Load site constraints (first NL rows)
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

        # Dump site constraints (next ND rows)
        for k in range(ND):
            ds = mine.dump_sites[k]
            n_dumpers = len(ds.dumper_list)
            unload = ds.dumper_list[0].dump_time if ds.dumper_list else 1.0
            row = np.zeros(NV)
            for i in range(NT):
                for j in range(NL):
                    row[i * NR + j * ND + k] = trip_rate[i * NR + j * ND + k]
            A_ub_list.append(row)
            b_ub_list.append(n_dumpers / unload)

        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        bounds = [(0, 1)] * NV

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')

        if result.success:
            x = result.x
            # Extract shadow prices (dual variables for inequality constraints)
            duals = result.ineqlin.marginals  # negative = binding, magnitude = value
            for j in range(NL):
                self._ls_shadow[j] = abs(duals[j])  # higher = more congested
            for k in range(ND):
                self._ds_shadow[k] = abs(duals[NL + k])
        else:
            x = np.ones(NV) / NR
            for j in range(NL):
                self._ls_shadow[j] = 0.0
            for k in range(ND):
                self._ds_shadow[k] = 0.0

        # Extract truck assignments and route targets
        self._truck_route = {}
        route_alloc = {}  # (j, k) → total allocation weight

        for i, truck in enumerate(trucks):
            allocs = x[i * NR:(i + 1) * NR]
            best_r = np.argmax(allocs)
            j = best_r // ND
            k = best_r % ND
            self._truck_route[truck.name] = (j, k)

            # Track LP allocation across all routes
            for r in range(NR):
                jj, kk = r // ND, r % ND
                route_alloc[(jj, kk)] = route_alloc.get((jj, kk), 0.0) + allocs[r]

        total_alloc = sum(route_alloc.values())
        if total_alloc > 0:
            self._route_target = {rte: v / total_alloc for rte, v in route_alloc.items()}
        else:
            self._route_target = {}

        # Initialize actual counts if first solve
        if not self._route_actual:
            self._route_actual = {(j, k): 0 for j in range(NL) for k in range(ND)}

        self._solved = True
        self._last_solve_time = mine.env.now

    def _maybe_reoptimize(self, mine):
        now = mine.env.now
        if not self._solved or (now - self._last_solve_time >= self.reoptimize_interval):
            self._solve_lp(mine)

    # ──────────────────────────────────────────────────────
    # Regret-based dynamic bias
    # ──────────────────────────────────────────────────────

    def _regret_bias(self, ls_idx, ds_idx):
        """
        Compute dynamic bias for a route based on regret (deviation from target).
        Under-served routes get a bonus (lower score multiplier).
        Over-served routes get a penalty (higher score multiplier).
        """
        if self._total_dispatches < 5:
            # Not enough data, use fixed bias for LP-assigned routes
            return 0.0

        route = (ls_idx, ds_idx)
        target_frac = self._route_target.get(route, 0.0)
        actual_frac = self._route_actual.get(route, 0) / max(self._total_dispatches, 1)
        deviation = actual_frac - target_frac  # positive = over-served

        # Clamp to [-0.3, 0.3] range for stability
        return max(-0.3, min(0.3, deviation * 3.0))

    def _shadow_penalty(self, ls_idx=None, ds_idx=None):
        """
        Shadow price penalty: sites with high shadow prices are at capacity.
        Returns a multiplier > 1.0 for congested sites.
        """
        penalty = 0.0
        max_ls = max(self._ls_shadow.values()) if self._ls_shadow else 1.0
        max_ds = max(self._ds_shadow.values()) if self._ds_shadow else 1.0

        if ls_idx is not None and max_ls > 0:
            # Normalize shadow price to [0, 1]
            normalized = self._ls_shadow.get(ls_idx, 0.0) / max_ls
            penalty += normalized * 0.15  # up to 15% penalty for fully binding

        if ds_idx is not None and max_ds > 0:
            normalized = self._ds_shadow.get(ds_idx, 0.0) / max_ds
            penalty += normalized * 0.15

        return penalty

    def _record_dispatch(self, ls_idx, ds_idx):
        """Record a dispatch for regret tracking."""
        route = (ls_idx, ds_idx)
        self._route_actual[route] = self._route_actual.get(route, 0) + 1
        self._total_dispatches += 1

    # ──────────────────────────────────────────────────────
    # Dynamic estimation
    # ──────────────────────────────────────────────────────

    def _load_score(self, truck, mine, ls_idx, dist):
        ls = mine.load_sites[ls_idx]
        travel = 60.0 * dist / truck.truck_speed
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
        best = float('inf')
        for j in range(len(mine.load_sites)):
            dist = mine.road.d2l_road_matrix[j, ds_idx]
            s = self._load_score(truck, mine, j, dist)
            if s < best:
                best = s
        return best

    # ──────────────────────────────────────────────────────
    # Dispatch decisions
    # ──────────────────────────────────────────────────────

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        self._maybe_reoptimize(mine)
        assigned_ls, assigned_ds = self._truck_route.get(truck.name, (0, 0))

        best_idx = 0
        best_score = float('inf')

        for j in range(len(mine.load_sites)):
            s = self._load_score(truck, mine, j, mine.road.charging_to_load[j])

            # Shadow price: penalize sites at capacity
            s *= (1.0 + self._shadow_penalty(ls_idx=j))

            # LP assignment bonus (for init, use fixed bias since no regret data yet)
            if j == assigned_ls:
                s *= 0.70  # strong bias at start

            if s < best_score:
                best_score = s
                best_idx = j

        return best_idx

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
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

            # Shadow price penalty for dump site
            score *= (1.0 + self._shadow_penalty(ds_idx=k))

            # Regret-based dynamic bias (replaces fixed lp_bias)
            regret = self._regret_bias(ls_idx, k)
            if k == assigned_ds:
                # LP-preferred: base discount + reduce if over-served
                score *= (0.70 + regret)
            else:
                # Non-LP: base + increase if under-served (regret negative)
                score *= (1.0 + regret)

            if score < best_score:
                best_score = score
                best_idx = k

        self._record_dispatch(ls_idx, best_idx)
        return best_idx

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        self._maybe_reoptimize(mine)
        ds_idx = mine.dump_sites.index(truck.current_location)
        assigned_ls = self._truck_route.get(truck.name, (0, 0))[0]

        best_idx = 0
        best_score = float('inf')

        for j in range(len(mine.load_sites)):
            dist = mine.road.d2l_road_matrix[j, ds_idx]
            s = self._load_score(truck, mine, j, dist)

            # Shadow price
            s *= (1.0 + self._shadow_penalty(ls_idx=j))

            # Regret-based bias
            # For back order, we don't know ds yet, use average regret across dump sites
            avg_regret = np.mean([self._regret_bias(j, k) for k in range(self._num_ds)]) if self._num_ds > 0 else 0.0
            if j == assigned_ls:
                s *= (0.70 + avg_regret)
            else:
                s *= (1.0 + avg_regret)

            if s < best_score:
                best_score = s
                best_idx = j

        return best_idx
