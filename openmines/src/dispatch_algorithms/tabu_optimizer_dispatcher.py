import math
import random
from gurobipy import Model, GRB, quicksum

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.mine import Mine

class MaxProductionDispatcher(BaseDispatcher):
    """
    一个示例的"两阶段"调度器：
      - 第一阶段(MIP)：为每台卡车选择(装载区→卸载区)路线，使理论总产量最大化。
      - 第二阶段(Tabu)：在MIP初解的基础上做邻域搜索(交换或更换路线)，进一步提升产量。
      - 动态：detect重大变化后调用 re_optimize()，再次跑以上两阶段，并更新 self.current_solution。
    """
    def __init__(self,
                 max_tabu_iter=50,
                 tabu_tenure=8,
                 reopt_interval=30):
        """
        :param max_tabu_iter:  Tabu Search 最大迭代次数
        :param tabu_tenure:    禁忌期限
        :param reopt_interval: 定期(或基于事件)重调度的触发间隔(可自定义逻辑)
        """
        super().__init__()
        self.name = "MaxProductionDispatcher"
        self.max_tabu_iter = max_tabu_iter
        self.tabu_tenure = tabu_tenure
        self.reopt_interval = reopt_interval

        self.best_solution = {}       # 记录全局最优解 { truck_name: (load_site_name, dump_site_name) }
        self.best_obj_val = 0.0
        self.current_solution = {}    # 当前可行解

        self.tabu_list = {}           # { move_key: remain_tenure }
        self.last_reopt_time = 0      # 上次reoptimize时刻

    def compute_solution(self, mine:"Mine"):
        """
        两阶段：先MIP求一次(静态近似)最优分配，再用Tabu Search做微调。
        得到 self.current_solution
        """
        # -------- 阶段1：MIP 分配 --------
        mip_solution, mip_obj = self._solve_mip(mine)

        # -------- 阶段2：Tabu 微调 --------
        final_sol, final_obj = self._tabu_search(mine, init_solution=mip_solution)

        # 更新 current_solution
        self.current_solution = final_sol
        if final_obj > self.best_obj_val:
            self.best_obj_val = final_obj
            self.best_solution = dict(final_sol)

    # =============== 阶段1：MIP (最大化产量) ===============
    def _solve_mip(self, mine:"Mine"):
        """
        使用一个简单的"单条线路分配"模型：每台卡车只能选 (l_idx, d_idx) 一条线路。
        目标: \sum_{t,l,d} capacity[t] * floor(TotalTime / cycleTime_{t,l,d}) * x_{t,l,d}.
        仅作为静态近似。
        """
        trucks = mine.trucks
        load_sites = mine.load_sites
        dump_sites = mine.dump_sites
        total_time = mine.total_time

        # 计算 truck_cycle_time_{t,l,d}
        #   = ( charging->l + load_time + l->d + unload_time + d->l )
        cycle_time = {}
        production_gain = {}  # 产量(理想状态) = capacity[t] * floor( total_time / cycleTime_{t,l,d} )
        for t_idx, t in enumerate(trucks):
            for l_idx, l_site in enumerate(load_sites):
                # 充电->l
                dist_init = mine.road.charging_to_load[l_idx]
                # 估算 load/unload time (可用更准确方式)
                approx_load_time = 5.0
                approx_unload_time = 3.0
                for d_idx, d_site in enumerate(dump_sites):
                    # l->d
                    dist_haul = mine.road.l2d_road_matrix[l_idx][d_idx]
                    # d->l (回程)
                    dist_back = mine.road.d2l_road_matrix[d_idx][l_idx]
                    
                    # 车速 t.truck_speed (km/h)-> min
                    speed_km_h = t.truck_speed
                    # 里程 / 速度 => 小时 => *60 => 分钟
                    init_time = dist_init / speed_km_h * 60.0
                    haul_time = dist_haul / speed_km_h * 60.0
                    back_time = dist_back / speed_km_h * 60.0
                    
                    cyc_time = init_time + approx_load_time + haul_time + approx_unload_time + back_time
                    cycle_time[(t_idx,l_idx,d_idx)] = cyc_time

                    # 一台车在一条线路全程跑 => 周转次数 ~ floor(total_time / cyc_time)
                    # 产量 = truck_capacity * 次数(简化)
                    max_cycle = math.floor(total_time / cyc_time) if cyc_time>0 else 0
                    production_gain[(t_idx,l_idx,d_idx)] = t.truck_capacity * max_cycle

        # 构建MIP
        model = Model("MaxProdMIP")
        model.setParam("OutputFlag", 0)

        T = len(trucks)
        L = len(load_sites)
        D = len(dump_sites)

        # x_{t,l,d} ∈ {0,1}：卡车 t 是否选(l,d)
        x = {}
        for t_idx in range(T):
            for l_idx in range(L):
                for d_idx in range(D):
                    x[(t_idx,l_idx,d_idx)] = model.addVar(vtype=GRB.BINARY,
                                                          name=f"x_{t_idx}_{l_idx}_{d_idx}")

        # 约束：每台卡车只能选一条线路
        for t_idx in range(T):
            model.addConstr(quicksum(x[(t_idx,l_idx,d_idx)]
                                     for l_idx in range(L)
                                     for d_idx in range(D)) == 1,
                            name=f"assign_{t_idx}")

        # 目标：最大化总产量
        obj_expr = quicksum( production_gain[(t_idx,l_idx,d_idx)] * x[(t_idx,l_idx,d_idx)]
                             for t_idx in range(T)
                             for l_idx in range(L)
                             for d_idx in range(D) )
        model.setObjective(obj_expr, GRB.MAXIMIZE)

        model.optimize()

        # 解析出(卡车->(l_idx, d_idx))
        mip_solution = {}
        total_obj = 0.0
        for t_idx, truck in enumerate(trucks):
            chosen = None
            best_val = 0.0
            for l_idx in range(L):
                for d_idx in range(D):
                    if x[(t_idx,l_idx,d_idx)].X > 0.5:
                        chosen = (l_idx, d_idx)
                        best_val = production_gain[(t_idx,l_idx,d_idx)]
                        break
                if chosen: break
            if chosen:
                ls_name = load_sites[chosen[0]].name
                ds_name = dump_sites[chosen[1]].name
                mip_solution[truck.name] = (ls_name, ds_name)
                total_obj += best_val
        return mip_solution, total_obj

    # =============== 阶段2：Tabu Search 微调 ===============
    def _tabu_search(self, mine:"Mine", init_solution:dict):
        """
        在 MIP 初解基础上，做邻域搜索(交换 or 改线路)，以期在仿真中获得更高产量。
        产量计算 _evaluate_production(...) 仍是一个静态近似，可加队列/堵车等惩罚。
        """
        current_sol = dict(init_solution)
        best_sol = dict(init_solution)
        best_prod = self._evaluate_production(mine, current_sol)

        for iteration in range(self.max_tabu_iter):
            neighbors = self._generate_neighbors(current_sol, mine, size=15)
            candidate_best_sol = None
            candidate_best_val = -1
            candidate_move_key = None

            for (nbr_sol, move_key) in neighbors:
                if move_key in self.tabu_list:
                    # 若没有破禁 => 不考虑
                    if self._evaluate_production(mine, nbr_sol) <= best_prod:
                        continue
                val = self._evaluate_production(mine, nbr_sol)
                if val> candidate_best_val:
                    candidate_best_val = val
                    candidate_best_sol = nbr_sol
                    candidate_move_key = move_key

            if candidate_best_sol is None:
                # 全被禁忌, 或无更好邻域 => 停止
                break

            # 接受该邻域
            current_sol = candidate_best_sol
            # 加入Tabu
            self.tabu_list[candidate_move_key] = self.tabu_tenure

            # 更新全局最优
            if candidate_best_val> best_prod:
                best_sol = dict(candidate_best_sol)
                best_prod = candidate_best_val

            # 衰减
            to_del = []
            for mk in self.tabu_list:
                self.tabu_list[mk] -= 1
                if self.tabu_list[mk] <=0:
                    to_del.append(mk)
            for mk in to_del:
                del self.tabu_list[mk]

        return best_sol, best_prod

    def _generate_neighbors(self, solution:dict, mine:"Mine", size=10):
        """
        生成邻域：随机对若干卡车进行操作：
           - 交换两车路线
           - 或某车换一个随机(装载区,卸载区)
        """
        trucks = mine.trucks
        load_sites = mine.load_sites
        dump_sites = mine.dump_sites

        neighbors = []
        truck_list = list(solution.keys())
        L = len(load_sites)
        D = len(dump_sites)

        for _ in range(size):
            new_sol = dict(solution)
            op = random.choice(["swap","reassign"])
            if op=="swap" and len(truck_list)>=2:
                tA, tB = random.sample(truck_list, 2)
                oldA = new_sol[tA]
                oldB = new_sol[tB]
                new_sol[tA] = oldB
                new_sol[tB] = oldA
                move_key = ("swap", tA, tB)
            else:
                # pick one truck, change route
                tA = random.choice(truck_list)
                old = new_sol[tA]
                l_idx = random.randint(0, L-1)
                d_idx = random.randint(0, D-1)
                new_sol[tA] = (mine.load_sites[l_idx].name,
                               mine.dump_sites[d_idx].name)
                move_key = ("reassign", tA, old, new_sol[tA])

            neighbors.append((new_sol, move_key))
        return neighbors

    def _evaluate_production(self, mine:"Mine", solution:dict):
        """
        用和 MIP 相同的静态近似公式，计算产量。
        你可扩展：把当前排队或拥堵等信息纳入里面，改成更贴近真实仿真。
        """
        trucks = mine.trucks
        load_sites = mine.load_sites
        dump_sites = mine.dump_sites
        total_time = mine.total_time

        # 建立索引
        truck_idx = {t.name:i for i,t in enumerate(trucks)}
        l_name2idx = {ls.name:i for i,ls in enumerate(load_sites)}
        d_name2idx = {ds.name:i for i,ds in enumerate(dump_sites)}

        # 计算
        approx_load_time = 5.0
        approx_unload_time = 3.0
        total_production = 0.0
        
        for t in trucks:
            t_idx = truck_idx[t.name]
            if t.name not in solution:
                continue
            
            (ls_name, ds_name) = solution[t.name]
            l_idx = l_name2idx[ls_name]
            d_idx = d_name2idx[ds_name]
            
            # 充电->l
            dist_init = mine.road.charging_to_load[l_idx]
            # l->d
            dist_haul = mine.road.l2d_road_matrix[l_idx][d_idx]
            # d->l (回程)
            dist_back = mine.road.d2l_road_matrix[d_idx][l_idx]
            
            speed_km_h = t.truck_speed
            init_time = dist_init / speed_km_h * 60.0
            haul_time = dist_haul / speed_km_h * 60.0  
            back_time = dist_back / speed_km_h * 60.0

            cyc_time = init_time + approx_load_time + haul_time + approx_unload_time + back_time
            
            if cyc_time > 0:
                cycles = math.floor(total_time / cyc_time)
            else:
                cycles = 0
            
            total_production += t.truck_capacity * cycles
        
        return total_production

    # =============== 实时/动态 调度 ===============
    def re_optimize(self, mine:"Mine"):
        """
        可在 give_init_order / give_haul_order / give_back_order 里调用，
        当发现时间或环境变化大时 => 重新跑 compute_solution
        """
        if (mine.env.now - self.last_reopt_time) >= self.reopt_interval:
            # 仅做示例：每隔 reopt_interval 分钟调用一次
            self.compute_solution(mine)
            self.last_reopt_time = mine.env.now

    # =============== 提供给Truck.run()的调度指令 ===============
    def give_init_order(self, truck:"Truck", mine:"Mine") -> int:
        self.re_optimize(mine)
        if not self.current_solution:
            self.compute_solution(mine)

        # 找到 truck 对应 (load_site_name)
        load_sites_map = {ls.name:i for i,ls in enumerate(mine.load_sites)}
        ld = self.current_solution.get(truck.name, None)
        if ld is None:
            return 0
        ls_name, ds_name = ld
        return load_sites_map[ls_name]

    def give_haul_order(self, truck:"Truck", mine:"Mine") -> int:
        self.re_optimize(mine)
        if not self.current_solution:
            self.compute_solution(mine)

        dump_sites_map = {ds.name:i for i,ds in enumerate(mine.dump_sites)}
        ld = self.current_solution.get(truck.name, None)
        if ld is None:
            return 0
        ls_name, ds_name = ld
        return dump_sites_map[ds_name]

    def give_back_order(self, truck:"Truck", mine:"Mine") -> int:
        self.re_optimize(mine)
        if not self.current_solution:
            self.compute_solution(mine)

        load_sites_map = {ls.name:i for i,ls in enumerate(mine.load_sites)}
        ld = self.current_solution.get(truck.name, None)
        if ld is None:
            return 0
        ls_name, ds_name = ld
        return load_sites_map[ls_name]
