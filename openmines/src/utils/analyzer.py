import re
import json
import time
import os
import inspect
import argparse
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from importlib import import_module
import pkgutil

class LogAnalyzer:
    def __init__(self, api_key, api_base="https://api.siliconflow.cn", model_name="deepseek-chat"):
        self.model_name = model_name
        self.time_intervals = [
            (0, 80, "0-80分钟"),
            (80, 160, "80-160分钟"),
            (160, 240, "160-240分钟")
        ]
        self.summaries = []
        self.console = Console()
        
        # 初始化API客户端（支持新旧两种方式）
        try:
            # 尝试新版API
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=api_base)
            self.api_version = "new"
        except ImportError:
            # 回退到旧版API
            import openai
            openai.api_key = api_key
            openai.api_base = "https://api.siliconflow.cn"
            self.client = openai
            self.api_version = "old"

    def extract_time(self, log_line):
        """使用正则表达式提取时间信息"""
        time_match = re.search(r"Time:<([\d.]+)>", log_line)
        return float(time_match.group(1)) if time_match else None

    def categorize_logs(self, log_path):
        """将日志按时间区间分类"""
        categorized = {interval[2]: [] for interval in self.time_intervals}
        
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                log_time = self.extract_time(line)
                if log_time is not None:
                    for start, end, name in self.time_intervals:
                        if start <= log_time < end:
                            categorized[name].append(line.strip())
                            break
        return categorized

    def get_summary(self, prompt, max_retries=3):
        """调用API获取总结"""
        for _ in range(max_retries):
            try:
                if self.api_version == "new":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        stream=False  # 关闭流式输出
                    )
                    return response.choices[0].message.content
                else:
                    response = self.client.ChatCompletion.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        stream=False  # 关闭流式输出
                    )
                    return response.choices[0].message.content
                
            except Exception as e:
                self.console.print(f"[bold red]API调用失败，正在重试... 错误信息：{str(e)}[/bold red]")
                time.sleep(2)
        return "总结失败"

    def get_dispatcher_code(self, algo_name):
        """通过模块遍历获取调度算法源代码"""
        try:
            # 遍历所有dispatch_algorithms模块
            dispatchers_package = 'openmines.src.dispatch_algorithms'
            package = import_module(dispatchers_package)
            
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                module = import_module(f"{dispatchers_package}.{module_name}")
                # 查找匹配的类
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and name == algo_name:
                        return inspect.getsource(obj)
            return f"未找到{algo_name}类的实现"
        except Exception as e:
            return f"获取调度算法代码失败: {str(e)}"

    def analyze_logs(self, log_path):
        """主分析函数"""
        self.console.print("[bold blue]开始分析日志文件...[/bold blue]")
        import pathlib
        
        # 处理日志路径
        path = pathlib.Path(log_path)
        if path.is_dir():
            log_files = list(path.glob("*.log"))
            if not log_files:
                self.console.print(f"[bold red]错误: 目录 {log_path} 中没有找到日志文件[/bold red]")
                return ""
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            log_path = str(latest_log)
            self.console.print(f"[bold blue]分析最新的日志文件: {latest_log.name}[/bold blue]")
        
        dispatcher_sections = self.identify_dispatcher_sections(log_path)
        
        # 打印段落信息
        self.console.print("\n[bold cyan]识别到以下调度算法段落：[/bold cyan]")
        for section in dispatcher_sections:
            self.console.print(f"• {section['name']}: 行号 {section['start_line']}-{section['end_line']}")

        # 选择要分析的段落
        if hasattr(self, 'dispatcher_name') and self.dispatcher_name:
            selected_sections = [s for s in dispatcher_sections if s['name'] == self.dispatcher_name]
            if not selected_sections:
                self.console.print(f"[bold red]错误: 未找到指定的调度算法 {self.dispatcher_name}[/bold red]")
                return ""
            self.console.print(f"\n[bold green]正在分析指定算法: {self.dispatcher_name}[/bold green]")
        else:
            if len(dispatcher_sections) > 1:
                latest_section = max(dispatcher_sections, key=lambda x: x['end_line'])
                selected_sections = [latest_section]
                self.console.print(f"[bold yellow]⚠ 检测到多个调度算法段落，自动选择最新的算法: {latest_section['name']}[/bold yellow]")
            else:
                selected_sections = dispatcher_sections

        # 分析选中的段落
        final_report = ""
        for section in selected_sections:
            self.console.print(f"\n[bold magenta]分析 {section['name']} 段落中...[/bold magenta]")
            report = self.analyze_section(log_path, section)
            final_report += f"## {section['name']}分析报告\n\n{report}\n\n"
        
        return final_report

    def identify_dispatcher_sections(self, log_path):
        """识别日志中的调度算法段落"""
        sections = []
        current_section = None
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 检测段落开始
                start_match = re.search(r'simulation started with dispatcher (\w+)', line)
                if start_match:
                    if current_section:  # 处理未正常结束的段落
                        current_section['end_line'] = line_num - 1
                        sections.append(current_section)
                    current_section = {
                        'name': start_match.group(1),
                        'start_line': line_num,
                        'end_line': None
                    }
                
                # 检测段落结束
                end_match = re.search(r'simulation finished with dispatcher (\w+)', line)
                if end_match and current_section:
                    if end_match.group(1) == current_section['name']:
                        current_section['end_line'] = line_num
                        sections.append(current_section)
                        current_section = None
        
        return sections

    def analyze_section(self, log_path, section):
        """分析单个调度算法段落"""
        # 提取指定行号的日志内容
        section_logs = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if section['start_line'] <= line_num <= section['end_line']:
                    section_logs.append(line.strip())
        
        # 临时日志文件
        temp_log = f"temp_{section['name']}.log"
        try:
            with open(temp_log, 'w', encoding='utf-8') as f:
                f.write("\n".join(section_logs))
            
            # 使用原有分析逻辑
            self.console.print(f"\n[bold magenta]正在分析 {section['name']} 段落（{len(section_logs)}行）...[/bold magenta]")
            return self._analyze_single_log(temp_log, section['name'])
        finally:
            # 清理临时文件
            if os.path.exists(temp_log):
                os.remove(temp_log)

    def _analyze_single_log(self, log_path, algo_name):
        """原有分析逻辑的封装"""
        # 获取调度算法代码
        dispatcher_code = self.get_dispatcher_code(algo_name)
        
        # 分时间段总结
        categorized = self.categorize_logs(log_path)
        
        self.summaries = []  # 清空之前的总结
        
        for interval_name, logs in categorized.items():
            if not logs:
                continue
            
            self.console.print(f"\n[bold cyan]正在分析 {interval_name} 时间段...[/bold cyan]")
            
            # 构建提示词
            prompt = f"""
            你现在是一个矿山卡车调度分析专家，你将会从专业角度分析日志数据，并尽量的使用数据详实的语言进行总结，方便调度员更好的进行策略改进。
            调度员只能写必要的代码，不能添加硬件设施，因此你的建议应该集中在派车策略上。
            你应该重点分析基于数据的现状和问题，而不是给出方案和建议。

            当前使用的调度算法代码如下：
            ```python
            {dispatcher_code}
            ```

            请分析以下矿山卡车调度日志片段（时间范围：{interval_name}），总结包含以下内容：
                1. 卡车在时段里的去向
                2. 装载点、卸载点的负载情况
                3. 异常事件（交通堵塞、设备故障）的数量和分布
                4. 交通情况
                5. 系统中可能影响效率的因素

            日志示例格式：
            [Truck: Time:<时间> Truck:<名称> Start moving to <目的地>, 距离: <数字>km, 速度: <数字>]

            请用中文返回分析报告，文本精炼直击现状且数据丰度高："""

            # 添加示例日志（控制长度）
            prompt += "\n\n相关日志片段：\n" + "\n".join(logs[:20])  # 取前20条作为示例
            
            # 获取并存储总结
            summary = self.get_summary(prompt)
            self.summaries.append({
                "interval": interval_name,
                "summary": summary
            })
            self.console.print(f"[bold green]✓ 完成{interval_name}的分析[/bold green]")
            time.sleep(1)  # 避免速率限制

        # 阶段2：综合总结
        self.console.print("\n[bold magenta]生成最终综合报告...[/bold magenta]")
        
        combined_summary = "\n\n".join(
            [f"## {s['interval']}\n{s['summary']}" for s in self.summaries]
        )
        
        final_prompt = f"""
        你现在是一个矿山卡车调度分析专家，你将会从专业角度分析日志数据，并尽量的使用数据详实的语言进行总结，方便调度员更好的进行策略改进。
        调度员只能写必要的代码，不能添加硬件设施，因此你的建议应该集中在派车策略上。
            基于以下分时段总结，请综合分析整个矿山运营情况：
                    1. 整体调度策略的有效性
                    2. 系统瓶颈和潜在风险
                    3. 策略上的优化建议
                    4. 关键数据指标趋势分析

            {combined_summary}
            该策略代码：
            ```python
            {dispatcher_code}
            ```
            请用中文返回分析报告，文本精炼直击现状且数据丰度高："""
        
        final_report = self.get_summary(final_prompt)
        return final_report

if __name__ == "__main__":
    console = Console()
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='矿山卡车调度日志分析工具')
    parser.add_argument('log_path', help='日志文件路径')
    parser.add_argument('--api-key', default="sk-whknzqhqufnsnfrjtrofqmlyuxhaobawmdtfhpuvyctaoblr",
                        help='API密钥')
    parser.add_argument('--model', default="deepseek-ai/DeepSeek-V3",
                        help='模型名称')
    
    args = parser.parse_args()
    
    try:
        analyzer = LogAnalyzer(
            api_key=args.api_key,
            model_name=args.model
        )
        
        console.print("[bold blue]开始分析矿山卡车调度日志...[/bold blue]")
        
        analysis_result = analyzer.analyze_logs(args.log_path)
        
        # 保存结果
        output_file = "log_analysis_report.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# 矿山卡车调度分析报告\n\n")
            f.write(analysis_result)
        
        console.print(f"[bold green]✓ 分析完成，结果已保存至{output_file}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]错误: {str(e)}[/bold red]")