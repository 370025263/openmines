from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite
from openmines.src.dump_site import DumpSite
from gurobipy import Model, GRB, quicksum  # 如果没有 gurobi，可以换成 pulp/ortools
import random

class OptimizeDispatcher(BaseDispatcher):
    def __init__(self, max_local_search_iter=50):
        super().__init__()
        self.name = "OptimizeDispatcher"
        self.max_local_search_iter = max_local_search_iter
        self.solution = {}  # { truck_name: (chosen_load_site_name, chosen_dump_site_name) }

    def compute_solution(self, mine:"Mine"):
        """
        先用 MILP 得到初始解，再做一次局部搜索改进。
        产出 self.solution = {truck_name -> (load_site_name, dump_site_name)}.
        """
        # 若已经有解，就不重复求解
        if self.solution:
            return

        # ============= 1. 预备：数据收集 =============

        trucks = mine.trucks
        load_sites = mine.load_sites
        dump_sites = mine.dump_sites

        # 给每辆卡车、每个装载点-卸载点组合，计算一个粗略的"周转时间cost"
        # 例如： cost_{t,l,d} = T(充->l) + load_time + T(l->d) + unload_time + T(d->l)
        # 注：以下均用最简单的线性时间近似(距离/速度*60), 可以自行改进
        cycle_time = {}
        for t_idx, truck in enumerate(trucks):
            for l_idx, load_site in enumerate(load_sites):
                # 充电站到装载点
                dist_init = mine.road.charging_to_load[l_idx]
                time_init = dist_init / truck.truck_speed * 60
                # 装载时间(简化处理)
                # 取装载点里所有 shovel 的平均值，也可以取最短 shovel_cycle_time
                # 这里只做演示，所以直接取一个近似值:
                load_time = 0.0
                if load_site.shovel_list:
                    avg_shovel_tons = sum(sv.shovel_tons for sv in load_site.shovel_list)/len(load_site.shovel_list)
                    avg_shovel_cycle = sum(sv.shovel_cycle_time for sv in load_site.shovel_list)/len(load_site.shovel_list)
                    load_time = (truck.truck_capacity / avg_shovel_tons) * avg_shovel_cycle
                for d_idx, dump_site in enumerate(dump_sites):
                    # 装载点到卸载点
                    dist_haul = mine.road.l2d_road_matrix[l_idx][d_idx]
                    time_haul = dist_haul / truck.truck_speed * 60
                    # 卸载时间(简化处理，取所有 dumper 平均卸载时间)
                    unload_time = 0.0
                    if dump_site.dumper_list:
                        avg_dumper_cycle = sum(dp.dump_time for dp in dump_site.dumper_list)/len(dump_site.dumper_list)
                        unload_time = avg_dumper_cycle
                    # 卸载点回到装载点
                    dist_unhaul = mine.road.d2l_road_matrix[d_idx][l_idx]  # 同一矩阵, l->d vs d->l
                    time_unhaul = dist_unhaul / truck.truck_speed * 60
                    # 总周转时间
                    total_cycle = time_init + load_time + time_haul + unload_time + time_unhaul
                    cycle_time[(t_idx, l_idx, d_idx)] = total_cycle

        # ============= 2. 构建 MILP =============
        model = Model("MineTruckAssignment")
        model.setParam('OutputFlag', 0)  # 不输出详细日志

        # 决策变量 x_{t,l,d} ∈ {0,1}, 表示车t 是否选择 (load_site l, dump_site d)
        x = {}
        for t_idx in range(len(trucks)):
            for l_idx in range(len(load_sites)):
                for d_idx in range(len(dump_sites)):
                    x[(t_idx, l_idx, d_idx)] = model.addVar(vtype=GRB.BINARY,
                                                            name=f"x_{t_idx}_{l_idx}_{d_idx}")

        # 约束: 每辆车只能绑定到唯一 (l,d)
        for t_idx in range(len(trucks)):
            model.addConstr(quicksum(x[(t_idx, l_idx, d_idx)]
                                     for l_idx in range(len(load_sites))
                                     for d_idx in range(len(dump_sites))) == 1,
                            name=f"truck_{t_idx}_one_route")

        # 目标: 最小化 sum_{t,l,d} ( cycle_time_{t,l,d} * x_{t,l,d} )
        obj = quicksum(cycle_time[(t_idx, l_idx, d_idx)] * x[(t_idx, l_idx, d_idx)]
                       for t_idx in range(len(trucks))
                       for l_idx in range(len(load_sites))
                       for d_idx in range(len(dump_sites)))
        model.setObjective(obj, GRB.MINIMIZE)

        # 求解
        model.optimize()

        # ============= 3. 解析 MILP 解 =============
        assignment = {}  # truck_idx -> (l_idx, d_idx)
        for t_idx in range(len(trucks)):
            for l_idx in range(len(load_sites)):
                for d_idx in range(len(dump_sites)):
                    if x[(t_idx, l_idx, d_idx)].X > 0.5:  # 表示此解里 truck t_idx 选了 (l_idx, d_idx)
                        assignment[t_idx] = (l_idx, d_idx)
                        break

        # ============= 4. 转化为 self.solution 结构 =============
        for t_idx, (l_idx, d_idx) in assignment.items():
            truck_name = trucks[t_idx].name
            load_site_name = load_sites[l_idx].name
            dump_site_name = dump_sites[d_idx].name
            self.solution[truck_name] = (load_site_name, dump_site_name)

        # ============= 5. 简单局部搜索(可选) =============
        # 尝试随机交换两辆卡车的(装载点,卸载点)看能否让总目标再降
        best_obj_val = self._evaluate_solution(self.solution, cycle_time, trucks, load_sites, dump_sites)
        for _ in range(self.max_local_search_iter):
            # 随机抽两辆卡车
            tA, tB = random.sample(range(len(trucks)), 2)
            truckA_name = trucks[tA].name
            truckB_name = trucks[tB].name
            oldA = self.solution[truckA_name]
            oldB = self.solution[truckB_name]
            # 交换(装载点,卸载点)
            self.solution[truckA_name] = oldB
            self.solution[truckB_name] = oldA
            new_obj_val = self._evaluate_solution(self.solution, cycle_time, trucks, load_sites, dump_sites)
            if new_obj_val + 1e-6 < best_obj_val:
                best_obj_val = new_obj_val
            else:
                # 不改进则回退
                self.solution[truckA_name] = oldA
                self.solution[truckB_name] = oldB

        print(f"[OptimizeDispatcher] MILP + LocalSearch done, final objective = {best_obj_val:.2f}")

    def _evaluate_solution(self, solution, cycle_time, trucks, load_sites, dump_sites):
        """辅助函数：给定一个解(对每车指定了 load_site, dump_site)，计算总周转时间之和。"""
        total = 0.0
        # 建立反查: load_site_name -> idx
        l_name_to_idx = {ls.name: i for i, ls in enumerate(load_sites)}
        d_name_to_idx = {ds.name: i for i, ds in enumerate(dump_sites)}
        for t_idx, truck in enumerate(trucks):
            truck_name = truck.name
            if truck_name not in solution:
                continue
            ls_name, ds_name = solution[truck_name]
            l_idx = l_name_to_idx[ls_name]
            d_idx = d_name_to_idx[ds_name]
            total += cycle_time[(t_idx, l_idx, d_idx)]
        return total

    # ========== 以下三个函数，用于给 Truck.run() 提供调度指令 ==========

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        告诉刚启动的卡车：去哪个装载点？
        """
        if not self.solution:
            self.compute_solution(mine)
        load_sites = {ls.name: i for i, ls in enumerate(mine.load_sites)}

        # 根据 solution 中 truck->(load_site_name, dump_site_name)
        # 返回 load_site 的下标
        load_site_name, _ = self.solution[truck.name]
        return load_sites[load_site_name]

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        车辆装载完成后：去哪个卸载点？
        """
        if not self.solution:
            self.compute_solution(mine)
        dump_sites = {ds.name: i for i, ds in enumerate(mine.dump_sites)}

        # 根据 solution 中 truck->(load_site_name, dump_site_name)
        # 返回 dump_site 的下标
        _, dump_site_name = self.solution[truck.name]
        return dump_sites[dump_site_name]

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        """
        卸载完成后：回哪个装载点？
        """
        if not self.solution:
            self.compute_solution(mine)
        load_sites = {ls.name: i for i, ls in enumerate(mine.load_sites)}

        load_site_name, _ = self.solution[truck.name]
        return load_sites[load_site_name]
