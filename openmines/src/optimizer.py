import os
import re
import time
import importlib
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from openmines.src.utils.analyzer import LogAnalyzer
from rich.markdown import Markdown
import logging

PLANNER_MODEL = "Pro/deepseek-ai/DeepSeek-R1"
CODER_MODEL = "Pro/deepseek-ai/DeepSeek-V3"
ANALYZER_MODEL = "Qwen/Qwen2.5-72B-Instruct-128K"


@dataclass
class StrategyRecord:
    name: str
    code: str
    metrics: dict
    analysis_report: str
    debug_count: int = 0
    error_history: List[str] = None

class APIError(Exception):
    """自定义API异常"""
    pass

class StrategyOptimizer:
    MAX_CONSECUTIVE_FAILURES = 30  # 新增最大连续失败次数
    
    def __init__(self, api_key, log_dir="optimization_logs"):
        self.api_key = api_key
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.history: List[StrategyRecord] = []
        self.console = Console()
        self.current_iter = 0
        self.max_iter = 10
        self.consecutive_failures = 0  # 新增失败计数器
        
        # 初始化API客户端
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn")
            self.api_version = "new"
        except ImportError:
            import openai
            openai.api_key = api_key
            openai.api_base = "https://api.siliconflow.cn"
            self.client = openai
            self.api_version = "old"

        self._init_logging()

    def _init_logging(self):
        """初始化错误日志"""
        self.error_log = self.log_dir / "error.log"
        logging.basicConfig(
            filename=self.error_log,
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _call_ai(self, prompt, model="Pro/deepseek-ai/DeepSeek-V3", max_retries=3):
        """统一调用AI接口（带流式输出）"""
        messages = [{"role": "system", "content": "你是一个专业的矿山调度算法工程师"}, 
                   {"role": "user", "content": prompt}]
        
        last_error = None
        for attempt in range(max_retries):
            try:
                if self.api_version == "new":
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.3,
                        stream=True
                    )
                    
                    full_response = ""
                    with self.console.status("[bold green]生成中...[/bold green]") as status:
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                # 实时显示带语法高亮的代码
                                if "```python" in full_response:
                                    self.console.print(Markdown(full_response))
                                else:
                                    self.console.print(content, end="", markup=False)
                    self.console.print("\n")
                    self.consecutive_failures = 0  # 成功时重置计数器
                    return full_response
                    
                else:
                    # 旧版API无流式支持
                    response = self.client.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=0.3
                    )
                    content = response.choices[0].message.content
                    self.console.print(f"[bold green]生成结果：[/bold green]\n{content}")
                    self.consecutive_failures = 0  # 成功时重置计数器
                    return content
                    
            except Exception as e:
                self.logger.error(f"API调用失败: {str(e)}")
                last_error = e
                self.consecutive_failures += 1
                error_msg = f"API调用失败({attempt+1}/{max_retries}): {str(e)}"
                self.console.print(f"[red]{error_msg}[/red]")
                time.sleep(2 ** attempt)  # 指数退避
                
                # 检查连续失败次数
                if self.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    raise SystemExit(f"[bold red]连续{self.MAX_CONSECUTIVE_FAILURES}次失败，系统终止[/bold red]")
        
        raise RuntimeError(f"API调用最终失败: {str(last_error)}")

    def planner_prompt(self, history: List[StrategyRecord], example_code: str) -> str:
        """生成策略规划提示词"""
        prompt = """你是一个矿山调度策略规划专家。请根据历史策略表现设计新的调度策略用于openmines仿真环境，需包含以下部分：
                        1. 初始化策略：车辆初始分配规则
                        2. 运输策略：满载时的路径选择规则 
                        3. 返程策略：空载时的路径选择规则
                    如果你发现你给出的修改导致产量降低，请你反思原因并调整策略，控制不要跟原始策略相差太大。
                    openmines环境介绍:
                        矿山Mine对象包含装载点LoadSite、卸载点DumpSite、充电站ChargingSite和卡车Truck对象，道路Road对象。
                        LoadSite包含多个Shovel对象，DumpSite包含多个Dumper对象。
                        卡车开始时从充电站出发前往装载点装载矿石(init订单)，然后运输到卸载点卸载（haul订单），然后再前往装载点(back订单）。
                        调度算法类需要实现give_init_order, give_haul_order, give_back_order三个方法。
                        道路Road对象包含道路矩阵l2d_road_matrix、d2l_road_matrix和充电站到装载点的道路矩阵charging_to_load_road_matrix。
                        l2d_road_matrix[i][j]表示从装载点i到卸载区j的距离，
                        d2l_road_matrix[i][j]表示从卸载区i到装载点j的距离，
                        charging_to_load_road_matrix[i]表示从充电站到装载点i的距离。
                        调度算法需要根据Mine对象(包含上述所有对象）和truck对象决策出init, haul, back订单（动作空间为装载区、卸载区、装载区）。
                    对象信息接口介绍：
                        Mine对象(openmines.src.mine.Mine)
                            包含load_sites:List[LoadSite],
                            dump_sites:List[DumpSite],
                            charging_site:ChargingSite,
                            road:Road,
                            dispatcher:BaseDispatcher
                            trucks:List[Truck], # 所有卡车对象
                            produced_tons, # 矿山产量
                            service_count # 已有订单总数
                        LoadSite对象(openmines.src.load_site.LoadSite)
                            包含shovel_list:List[Shovel],
                            parkinglot:ParkingLot,
                            service_ability_ratio, # 服务能力比例（个别铲车暂时停机导致）
                            estimated_queue_wait_time, # 预估最早可服务的铲车等待时间
                            avg_queue_wait_time, # 平均铲车等待时间
                            load_site_productivity # 装载点生产力（吨/min）
                            status:Dict[str, str] # 装载点历史状态
                                例如：
                                self.status[int(env.now)] = {
                                    "produced_tons": self.produced_tons,
                                    "service_count": self.service_count,
                                }
                        Shovel对象(openmines.src.shovel.Shovel)
                            包含load_site:LoadSite, # 所属装载点    
                            shovel_tons, # 铲车铲斗容量
                            shovel_cycle_time, # 铲车铲斗周期时间min
                            service_count, # 已服务次数
                            last_service_time, # 上次服务开始时间
                            last_service_done_time, # 上次服务结束时间
                            est_waiting_time, # 估计等待时间
                            last_breakdown_time, # 上次故障时间
                            status:Dict[str, str] # 铲车历史状态
                                例如：
                                self.status[int(env.now)] = {
                                    "repair": self.repair,
                                    "produced_tons": self.produced_tons,
                                    "service_count": self.service_count,
                                }
                        DumpSite对象(openmines.src.dump_site.DumpSite)
                            包含dumper_list:List[Dumper],
                            truck_visits, # 卸载点总车次数
                            produce_tons, # 卸载点产量
                            service_count, # 服务次数
                        Dumper对象(openmines.src.dump_site.Dumper)
                            包含dump_site:DumpSite, # 所属卸载点
                            dump_time:float, # 卸载时间
                            dumper_tons:float, # 已卸载吨数
                            service_count, # 服务次数
                        Road对象(openmines.src.road.Road)
                            包含l2d_road_matrix:np.ndarray, # 装载点到卸载点的距离矩阵
                            d2l_road_matrix:np.ndarray, # 卸载点到装载点的距离矩阵
                            charging_to_load_road_matrix:List[float], # 充电站到装载点的距离列表
                            road_repairs:Dict[Tuple[Union[LoadSite, DumpSite], Union[LoadSite, DumpSite]], Tuple[bool, float]], # 道路维修状态字典,key为(起点,终点)元组,value为(是否在维修,预期修复完成时间)
                            truck_on_road(self, start:Union[LoadSite, DumpSite], end:Union[LoadSite, DumpSite])->List["Truck"] # 获取在道路上的卡车列表
                            
原始策略代码:
{example_code}           

历史策略表现（最新5条）：
"""
        for record in history[-2:]:
            prompt += f"\n策略名称：{record.name}\n"
            prompt += f"运行指标：{record.metrics}\n"
            prompt += f"关键问题：{record.analysis_report[-800:]}...\n"
            prompt += f"调试次数：{record.debug_count}\n"
            prompt += f"策略代码：{record.code}...\n"

        prompt += """
请按以下格式输出：
策略名称：<体现策略核心的英文名称>
策略描述：
1. 初始化策略：...
2. 运输策略：...
3. 返程策略：...
创新点：<指出本策略相比前代的改进>"""
        return prompt

    def coder_prompt(self, strategy_desc: str, example_code: str, error: str = None) -> str:
        """生成代码编写提示词"""
        prompt = f"""你是一个Python开发专家，请根据策略描述编写调度算法类代码，以实现最大产量。
        要求：
            1. 继承BaseDispatcher
            2. 实现give_init_order, give_haul_order, give_back_order三个方法
            3. 使用类型注解
            4. 包含必要的异常处理

 openmines环境介绍:
        矿山Mine对象包含装载点LoadSite、卸载点DumpSite、充电站ChargingSite和卡车Truck对象，道路Road对象。
        LoadSite包含多个Shovel对象，DumpSite包含多个Dumper对象。
        卡车开始时从充电站出发前往装载点装载矿石(init订单)，然后运输到卸载点卸载（haul订单），然后再前往装载点(back订单）。
        调度算法类需要实现give_init_order, give_haul_order, give_back_order三个方法。
        道路Road对象包含道路矩阵l2d_road_matrix、d2l_road_matrix和充电站到装载点的道路矩阵charging_to_load_road_matrix。
        l2d_road_matrix[i][j]表示从装载点i到卸载区j的距离，
        d2l_road_matrix[i][j]表示从卸载区i到装载点j的距离，
        charging_to_load_road_matrix[i]表示从充电站到装载点i的距离。
        调度算法需要根据Mine对象(包含上述所有对象）和truck对象决策出init, haul, back订单（动作空间为装载区、卸载区、装载区）。
Mine对象(openmines.src.mine.Mine)
    包含load_sites:List[LoadSite],
    dump_sites:List[DumpSite],
    charging_site:ChargingSite,
    road:Road,
    dispatcher:BaseDispatcher
    trucks:List[Truck], # 所有卡车对象
    produced_tons, # 矿山产量
    service_count # 已有订单总数
LoadSite对象(openmines.src.load_site.LoadSite)
    包含shovel_list:List[Shovel],
    parkinglot:ParkingLot,
    service_ability_ratio, # 服务能力比例（个别铲车暂时停机导致）
    estimated_queue_wait_time, # 预估最早可服务的铲车等待时间
    avg_queue_wait_time, # 平均铲车等待时间
    load_site_productivity # 装载点生产力（吨/min）
    status:Dict[str, str] # 装载点历史状态
        例如：
        self.status[int(env.now)] = {{
            "produced_tons": self.produced_tons,
            "service_count": self.service_count,
        }}
Shovel对象(openmines.src.shovel.Shovel)
    包含load_site:LoadSite, # 所属装载点    
    shovel_tons, # 铲车铲斗容量
    shovel_cycle_time, # 铲车铲斗周期时间min
    service_count, # 已服务次数
    last_service_time, # 上次服务开始时间
    last_service_done_time, # 上次服务结束时间
    est_waiting_time, # 估计等待时间
    last_breakdown_time, # 上次故障时间
    status:Dict[str, str] # 铲车历史状态
        例如：
        self.status[int(env.now)] = {{
            "repair": self.repair,
            "produced_tons": self.produced_tons,
            "service_count": self.service_count,
        }}
DumpSite对象(openmines.src.dump_site.DumpSite)
    包含dumper_list:List[Dumper],
    truck_visits, # 卸载点总车次数
    produce_tons, # 卸载点产量
    service_count, # 服务次数
Dumper对象(openmines.src.dump_site.Dumper)
    包含dump_site:DumpSite, # 所属卸载点
    dump_time:float, # 卸载时间
    dumper_tons:float, # 已卸载吨数
    service_count, # 服务次数
Road对象(openmines.src.road.Road)
    包含l2d_road_matrix:np.ndarray, # 装载点到卸载点的距离矩阵
    d2l_road_matrix:np.ndarray, # 卸载点到装载点的距离矩阵
    charging_to_load_road_matrix:List[float], # 充电站到装载点的距离列表
    road_repairs:Dict[Tuple[Union[LoadSite, DumpSite], Union[LoadSite, DumpSite]], Tuple[bool, float]], # 道路维修状态字典,key为(起点,终点)元组,value为(是否在维修,预期修复完成时间)
    truck_on_road(self, start:Union[LoadSite, DumpSite], end:Union[LoadSite, DumpSite])->List["Truck"] # 获取在道路上的卡车列表
    
参考示例(产量12000tons)：
{example_code}

当前策略描述：
{strategy_desc}"""

        if error:
            prompt += f"\n\n需要修复的错误：\n{error}"

        prompt += "\n\n请直接输出完整的Python类代码，不要包含任何解释。"
        return prompt

    def generate_strategy(self):
        """生成新策略并动态注册到包中"""
        # 规划策略
        example_code = Path("openmines/src/dispatch_algorithms/fixed_group_dispatch.py").read_text()
        plan_prompt = self.planner_prompt(self.history,example_code)
        strategy_text = self._call_ai(plan_prompt, model=PLANNER_MODEL)
        
        # 提取策略名称并规范化
        name_match = re.search(r"策略名称：(.+)", strategy_text)
        strategy_name = name_match.group(1) if name_match else f"Strategy_{len(self.history)+1}"
        # 清洗括号和空格
        strategy_name = re.sub(r'\s*\([^)]*\)', '', strategy_name).strip()
        module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', strategy_name).lower()
        
        # 生成代码
        code_prompt = self.coder_prompt(strategy_text, example_code)
        generated_code = self._call_ai(code_prompt, model=CODER_MODEL)
        
        # 将代码写入dispatch_algorithms目录
        dispatch_dir = Path("openmines/src/dispatch_algorithms")
        code_path = dispatch_dir / f"{module_name}.py"
        
        # 添加文件头确保可导入
        # 清洗代码中的markdown标记
        generated_code = generated_code.replace("```python", "").replace("```", "").strip()
        full_code = f"{generated_code}"
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(full_code)
        
        return strategy_name, generated_code, strategy_text

    def test_strategy(self, strategy_name: str) -> dict:
        """测试策略并返回指标"""
        from openmines.src.cli.run import run_dispatch_sim
        import traceback
        
        # 初始化默认结果
        result = {"error": "未知错误"}
        
        try:
            # 动态加载策略模块
            module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', strategy_name).lower()
            spec = importlib.util.spec_from_file_location(
                name=f"openmines.src.dispatch_algorithms.{module_name}",
                location=f"openmines/src/dispatch_algorithms/{module_name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 获取调度器类
            dispatcher_class = getattr(module, strategy_name)
            dispatcher = dispatcher_class()
            
            # 运行仿真测试
            config_path = "openmines/src/conf/north_pit_mine.json"
            if not Path(config_path).exists():
                return {"error": f"配置文件不存在: {config_path}"}
            
            ticks = run_dispatch_sim(dispatcher, config_path)
            
            # 解析运行结果
            if not isinstance(ticks, dict) or "summary" not in ticks:
                return {"error": "无效的仿真结果格式"}
            
            result = ticks["summary"]
            result["status"] = "success"
            
        except SyntaxError as e:
            error_msg = f"语法错误: 第{e.lineno}行, {e.msg}\n示例代码:\n{e.text}"
            result = {"error": f"代码错误: {error_msg}", "status": "compile_error"}
        except AttributeError as e:
            error_msg = f"类名不匹配: 需要 {strategy_name}\n{traceback.format_exc()}"
            result = {"error": f"类定义错误: {error_msg}", "status": "class_mismatch"}
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result = {"error": f"运行时错误: {error_msg}", "status": "runtime_error"}
        finally:
            # 清理临时文件
            temp_file = Path(f"openmines/src/dispatch_algorithms/{module_name}.py")
            if temp_file.exists():
                temp_file.unlink()
            
        # 确保返回类型正确
        return result if isinstance(result, dict) else {"error": "未知错误"}

    def analyze_strategy(self, log_path: str, strategy_name: str,model_name:str = "Qwen/Qwen2.5-Coder-32B-Instruct") -> str:
        """分析策略表现"""
        analyzer = LogAnalyzer(self.api_key,model_name=model_name)
        return analyzer.analyze_logs(log_path)

    def optimization_loop(self):
        try:
            with Progress(transient=True) as progress:  # 使用transient模式
                task = progress.add_task("[cyan]优化进度...", total=self.max_iter)
                
                while self.current_iter < self.max_iter:
                    # 生成阶段
                    progress.update(task, description="生成策略...")
                    strategy_name, code, desc = self.generate_strategy()
                    
                    # 测试阶段
                    progress.update(task, description="测试策略...")
                    test_result = self.test_strategy(strategy_name)
                    
                    # 分析阶段前暂停进度条
                    progress.stop()
                    self.console.print("\n"*2)  # 添加换行分隔
                    with self.console.status("[bold green]分析结果中...[/bold green]", spinner="dots"):
                        log_path = Path.cwd() / "logs"
                        analysis_report = self.analyze_strategy(str(log_path), strategy_name,model_name=ANALYZER_MODEL)
                    
                    # 恢复进度条
                    progress.start()
                    
                    # 记录历史
                    record = StrategyRecord(
                        name=strategy_name,
                        code=code,
                        metrics=test_result,
                        analysis_report=analysis_report,
                        debug_count=0,
                        error_history=[test_result.get("error")]
                    )
                    self.history.append(record)
                    
                    # 更新显示
                    self.display_status()
                    self.current_iter += 1
                    progress.update(task, advance=1)
        except Exception as e:
            self.console.print(f"[bold red]未处理异常: {str(e)}[/bold red]")
            sys.exit(1)

    def display_status(self):
        """用Rich表格展示优化状态"""
        table = Table(title=f"优化进度 {self.current_iter}/{self.max_iter}")
        table.add_column("策略名称")
        table.add_column("产量", justify="right")
        table.add_column("调试次数", justify="right")
        table.add_column("关键指标")
        
        for record in self.history[-5:]:
            prod = record.metrics.get("produced_tons", "N/A")
            table.add_row(
                record.name,
                f"{prod:.1f}吨" if isinstance(prod, float) else prod,
                str(record.debug_count),
                record.analysis_report[:50] + "..."
            )
        self.console.print(table)

if __name__ == "__main__":
    api_key = "sk-whknzqhqufnsnfrjtrofqmlyuxhaobawmdtfhpuvyctaoblr"
    optimizer = StrategyOptimizer(api_key)
    optimizer.optimization_loop() 