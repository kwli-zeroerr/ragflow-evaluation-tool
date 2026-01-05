"""
架构性能对比测试工具
对比 SQLite+本地存储+ES 与 PostgreSQL+MinIO+ES 两种架构的性能差异
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import matplotlib.pyplot as plt
import matplotlib
from evaluation_tool import RagFlowClient, RetrievalConfig, TestCase, EvaluationRunner

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
performance_test_dir = output_dir / "performance_test"
performance_test_dir.mkdir(parents=True, exist_ok=True)

logs_dir = output_dir / "logs"
logs_dir.mkdir(exist_ok=True)

# 配置日志
file_handler = logging.FileHandler(
    logs_dir / f'architecture_performance_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    encoding='utf-8'
)
stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)


@dataclass
class ArchitectureConfig:
    """架构配置"""
    name: str
    api_url: str
    api_key: str
    description: str  # 架构描述，如 "SQLite+本地存储+ES"
    knowledge_id: Optional[str] = None  # OpenWebUI 知识库ID（可选）
    use_ragflow_format: bool = False  # 是否使用 RAGflow 格式
    use_ragflow_index: bool = False  # 是否使用 RAGFlow ES 索引
    ragflow_timeout: int = 30  # OpenWebUI 接口超时时间


@dataclass
class PerformanceMetrics:
    """性能指标"""
    latency_mean: float  # 平均延迟
    latency_median: float  # 中位数延迟
    latency_p95: float  # P95延迟
    latency_p99: float  # P99延迟
    latency_min: float  # 最小延迟
    latency_max: float  # 最大延迟
    throughput: float  # 吞吐量（请求/秒）
    success_rate: float  # 成功率
    error_count: int  # 错误数量
    total_requests: int  # 总请求数
    total_time: float  # 总耗时


class ArchitecturePerformanceTester:
    """架构性能测试器"""
    
    def __init__(self, architecture_a: ArchitectureConfig, architecture_b: ArchitectureConfig):
        self.arch_a = architecture_a
        self.arch_b = architecture_b
        
        # 根据配置创建客户端（支持 OpenWebUI）
        if architecture_a.knowledge_id:
            self.client_a = RagFlowClient(
                architecture_a.api_url,
                architecture_a.api_key,
                knowledge_id=architecture_a.knowledge_id,
                use_ragflow_format=architecture_a.use_ragflow_format,
                use_ragflow_index=architecture_a.use_ragflow_index,
                ragflow_timeout=architecture_a.ragflow_timeout
            )
        else:
            self.client_a = RagFlowClient(architecture_a.api_url, architecture_a.api_key)
        
        if architecture_b.knowledge_id:
            self.client_b = RagFlowClient(
                architecture_b.api_url,
                architecture_b.api_key,
                knowledge_id=architecture_b.knowledge_id,
                use_ragflow_format=architecture_b.use_ragflow_format,
                use_ragflow_index=architecture_b.use_ragflow_index,
                ragflow_timeout=architecture_b.ragflow_timeout
            )
        else:
            self.client_b = RagFlowClient(architecture_b.api_url, architecture_b.api_key)
        
        self.results_lock = Lock()
    
    def run_single_request_test(self, question: str, config: RetrievalConfig, 
                                client: RagFlowClient, arch_name: str) -> Dict[str, Any]:
        """运行单次请求测试"""
        start_time = time.time()
        try:
            response = client.search(question, "", config)
            latency = time.time() - start_time
            
            success = "error" not in response
            return {
                "latency": latency,
                "success": success,
                "error": response.get("error") if not success else None,
                "chunks_count": len(response.get("data", {}).get("chunks", [])) if success else 0
            }
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"[{arch_name}] 请求失败: {str(e)}")
            return {
                "latency": latency,
                "success": False,
                "error": str(e),
                "chunks_count": 0
            }
    
    def run_latency_test(self, test_cases: List[TestCase], config: RetrievalConfig,
                        warmup_rounds: int = 3) -> Tuple[Dict[str, PerformanceMetrics], Dict[str, PerformanceMetrics]]:
        """
        运行延迟测试
        返回: (metrics_a, metrics_b)
        """
        logger.info("=" * 80)
        logger.info("开始延迟测试")
        logger.info("=" * 80)
        
        # 预热阶段
        if warmup_rounds > 0:
            logger.info(f"预热阶段: {warmup_rounds} 轮请求（两个架构各预热{warmup_rounds}次）")
            for i in range(warmup_rounds):
                if test_cases:
                    test_case = test_cases[i % len(test_cases)]
                    self.run_single_request_test(test_case.question, config, self.client_a, self.arch_a.name)
                    self.run_single_request_test(test_case.question, config, self.client_b, self.arch_b.name)
            logger.info("预热完成\n")
        
        # 正式测试 - 先测试架构A，再测试架构B
        logger.info("=" * 80)
        logger.info(f"阶段1: 测试架构A ({self.arch_a.description})")
        logger.info("=" * 80)
        latencies_a = []
        errors_a = 0
        start_time_a = time.time()
        
        for idx, test_case in enumerate(test_cases, 1):
            if idx % 50 == 0 or idx == 1:
                logger.info(f"[架构A] 进度: {idx}/{len(test_cases)}")
            
            result_a = self.run_single_request_test(test_case.question, config, self.client_a, self.arch_a.name)
            if result_a["success"]:
                latencies_a.append(result_a["latency"])
            else:
                errors_a += 1
                logger.warning(f"[架构A] 请求失败 ({idx}/{len(test_cases)}): {result_a.get('error')}")
            
            # 避免请求过快
            time.sleep(0.05)
        
        total_time_a = time.time() - start_time_a
        metrics_a = self._calculate_metrics(latencies_a, errors_a, len(test_cases), total_time_a)
        logger.info(f"[架构A] 测试完成: 成功 {len(latencies_a)}/{len(test_cases)}, 平均延迟 {metrics_a.latency_mean:.3f}s\n")
        
        # 短暂休息，避免服务器压力
        time.sleep(2)
        
        logger.info("=" * 80)
        logger.info(f"阶段2: 测试架构B ({self.arch_b.description})")
        logger.info("=" * 80)
        latencies_b = []
        errors_b = 0
        start_time_b = time.time()
        
        for idx, test_case in enumerate(test_cases, 1):
            if idx % 50 == 0 or idx == 1:
                logger.info(f"[架构B] 进度: {idx}/{len(test_cases)}")
            
            result_b = self.run_single_request_test(test_case.question, config, self.client_b, self.arch_b.name)
            if result_b["success"]:
                latencies_b.append(result_b["latency"])
            else:
                errors_b += 1
                logger.warning(f"[架构B] 请求失败 ({idx}/{len(test_cases)}): {result_b.get('error')}")
            
            # 避免请求过快
            time.sleep(0.05)
        
        total_time_b = time.time() - start_time_b
        metrics_b = self._calculate_metrics(latencies_b, errors_b, len(test_cases), total_time_b)
        logger.info(f"[架构B] 测试完成: 成功 {len(latencies_b)}/{len(test_cases)}, 平均延迟 {metrics_b.latency_mean:.3f}s\n")
        
        logger.info("=" * 80)
        logger.info("延迟测试完成")
        logger.info("=" * 80)
        
        return metrics_a, metrics_b
    
    def run_concurrency_test(self, test_cases: List[TestCase], config: RetrievalConfig,
                            concurrency_levels: List[int] = [1, 5, 10, 20, 50]) -> Dict[str, Dict[int, PerformanceMetrics]]:
        """
        运行并发测试
        测试不同并发级别下的性能
        返回: {"arch_a": {concurrency: metrics}, "arch_b": {concurrency: metrics}}
        """
        logger.info("=" * 80)
        logger.info("开始并发测试")
        logger.info("=" * 80)
        logger.info("提示: 按 Ctrl+C 可以安全中断测试，已完成的结果会被保存")
        
        results = {"arch_a": {}, "arch_b": {}}
        
        try:
            for concurrency in concurrency_levels:
                logger.info("\n" + "=" * 80)
                logger.info(f"测试并发级别: {concurrency}")
                logger.info("=" * 80)
                
                # 测试架构A
                logger.info(f"\n[架构A] 开始测试并发级别 {concurrency}")
                logger.info(f"架构: {self.arch_a.description}")
                metrics_a = self._run_concurrent_requests(
                    test_cases, config, self.client_a, self.arch_a.name, concurrency
                )
                results["arch_a"][concurrency] = metrics_a
                logger.info(f"[架构A] 完成 - 平均延迟: {metrics_a.latency_mean:.3f}s, 吞吐量: {metrics_a.throughput:.2f} req/s, 成功率: {metrics_a.success_rate*100:.1f}%")
                
                # 短暂休息，避免服务器压力过大
                time.sleep(2)
                
                # 测试架构B
                logger.info(f"\n[架构B] 开始测试并发级别 {concurrency}")
                logger.info(f"架构: {self.arch_b.description}")
                metrics_b = self._run_concurrent_requests(
                    test_cases, config, self.client_b, self.arch_b.name, concurrency
                )
                results["arch_b"][concurrency] = metrics_b
                logger.info(f"[架构B] 完成 - 平均延迟: {metrics_b.latency_mean:.3f}s, 吞吐量: {metrics_b.throughput:.2f} req/s, 成功率: {metrics_b.success_rate*100:.1f}%")
                
                # 对比结果
                logger.info(f"\n并发 {concurrency} 对比结果:")
                logger.info(f"  架构A - 平均延迟: {metrics_a.latency_mean:.3f}s, 吞吐量: {metrics_a.throughput:.2f} req/s")
                logger.info(f"  架构B - 平均延迟: {metrics_b.latency_mean:.3f}s, 吞吐量: {metrics_b.throughput:.2f} req/s")
                if metrics_a.throughput > 0:
                    improvement = ((metrics_b.throughput - metrics_a.throughput) / metrics_a.throughput) * 100
                    logger.info(f"  吞吐量差异: {improvement:+.1f}% (B相对A)")
                
                    # 短暂休息，避免服务器压力过大
                    time.sleep(2)
        except KeyboardInterrupt:
            logger.warning("\n" + "=" * 80)
            logger.warning("收到中断信号 (Ctrl+C)，正在停止测试...")
            logger.warning("=" * 80)
            logger.info(f"已完成的并发级别: {list(results['arch_a'].keys())}")
            logger.info("正在保存已完成的结果...")
            raise  # 重新抛出异常，让主程序处理
        
        return results
    
    def _run_concurrent_requests(self, test_cases: List[TestCase], config: RetrievalConfig,
                                 client: RagFlowClient, arch_name: str, concurrency: int) -> PerformanceMetrics:
        """运行并发请求"""
        # 如果测试用例不够，循环使用
        questions = [tc.question for tc in test_cases]
        if len(questions) < concurrency:
            questions = questions * (concurrency // len(questions) + 1)
        questions = questions[:concurrency * 10]  # 每个线程10个请求
        
        latencies = []
        errors = 0
        completed = 0
        start_time = time.time()
        
        def run_request(question: str):
            nonlocal errors, completed
            result = self.run_single_request_test(question, config, client, arch_name)
            if result["success"]:
                with self.results_lock:
                    latencies.append(result["latency"])
                    completed += 1
            else:
                with self.results_lock:
                    errors += 1
                    completed += 1
        
        # 并发执行
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(run_request, q) for q in questions]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"[{arch_name}] 并发请求异常: {str(e)}")
                    with self.results_lock:
                        errors += 1
                        completed += 1
        
        total_time = time.time() - start_time
        return self._calculate_metrics(latencies, errors, len(questions), total_time)
    
    def _calculate_metrics(self, latencies: List[float], errors: int, 
                          total_requests: int, total_time: float) -> PerformanceMetrics:
        """计算性能指标"""
        if not latencies:
            return PerformanceMetrics(
                latency_mean=0.0,
                latency_median=0.0,
                latency_p95=0.0,
                latency_p99=0.0,
                latency_min=0.0,
                latency_max=0.0,
                throughput=0.0,
                success_rate=0.0,
                error_count=errors,
                total_requests=total_requests,
                total_time=total_time
            )
        
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        
        return PerformanceMetrics(
            latency_mean=np.mean(latencies),
            latency_median=np.median(latencies),
            latency_p95=latencies_sorted[int(n * 0.95)] if n > 0 else 0.0,
            latency_p99=latencies_sorted[int(n * 0.99)] if n > 0 else 0.0,
            latency_min=min(latencies),
            latency_max=max(latencies),
            throughput=len(latencies) / total_time if total_time > 0 else 0.0,
            success_rate=len(latencies) / total_requests if total_requests > 0 else 0.0,
            error_count=errors,
            total_requests=total_requests,
            total_time=total_time
        )
    
    def generate_comparison_report(self, latency_metrics: Optional[Tuple[PerformanceMetrics, PerformanceMetrics]],
                                  concurrency_results: Dict[str, Dict[int, PerformanceMetrics]],
                                  previous_results: Optional[pd.DataFrame] = None):
        """生成对比报告，可以合并之前的测试结果"""
        report_data = []
        
        # 延迟测试结果（如果进行了延迟测试）
        if latency_metrics is not None:
            metrics_a, metrics_b = latency_metrics
            
            # 延迟指标对比
            report_data.append({
                "指标类型": "延迟测试",
                "指标名称": "平均延迟 (秒)",
                "架构A": metrics_a.latency_mean,
                "架构B": metrics_b.latency_mean,
                "差异": metrics_b.latency_mean - metrics_a.latency_mean,
                "差异百分比": ((metrics_b.latency_mean - metrics_a.latency_mean) / metrics_a.latency_mean * 100) if metrics_a.latency_mean > 0 else 0
            })
            
            report_data.append({
                "指标类型": "延迟测试",
                "指标名称": "中位数延迟 (秒)",
                "架构A": metrics_a.latency_median,
                "架构B": metrics_b.latency_median,
                "差异": metrics_b.latency_median - metrics_a.latency_median,
                "差异百分比": ((metrics_b.latency_median - metrics_a.latency_median) / metrics_a.latency_median * 100) if metrics_a.latency_median > 0 else 0
            })
            
            report_data.append({
                "指标类型": "延迟测试",
                "指标名称": "P95延迟 (秒)",
                "架构A": metrics_a.latency_p95,
                "架构B": metrics_b.latency_p95,
                "差异": metrics_b.latency_p95 - metrics_a.latency_p95,
                "差异百分比": ((metrics_b.latency_p95 - metrics_a.latency_p95) / metrics_a.latency_p95 * 100) if metrics_a.latency_p95 > 0 else 0
            })
            
            report_data.append({
                "指标类型": "延迟测试",
                "指标名称": "P99延迟 (秒)",
                "架构A": metrics_a.latency_p99,
                "架构B": metrics_b.latency_p99,
                "差异": metrics_b.latency_p99 - metrics_a.latency_p99,
                "差异百分比": ((metrics_b.latency_p99 - metrics_a.latency_p99) / metrics_a.latency_p99 * 100) if metrics_a.latency_p99 > 0 else 0
            })
            
            report_data.append({
                "指标类型": "延迟测试",
                "指标名称": "最小延迟 (秒)",
                "架构A": metrics_a.latency_min,
                "架构B": metrics_b.latency_min,
                "差异": metrics_b.latency_min - metrics_a.latency_min,
                "差异百分比": ((metrics_b.latency_min - metrics_a.latency_min) / metrics_a.latency_min * 100) if metrics_a.latency_min > 0 else 0
            })
            
            report_data.append({
                "指标类型": "延迟测试",
                "指标名称": "最大延迟 (秒)",
                "架构A": metrics_a.latency_max,
                "架构B": metrics_b.latency_max,
                "差异": metrics_b.latency_max - metrics_a.latency_max,
                "差异百分比": ((metrics_b.latency_max - metrics_a.latency_max) / metrics_a.latency_max * 100) if metrics_a.latency_max > 0 else 0
            })
            
            report_data.append({
                "指标类型": "吞吐量",
                "指标名称": "吞吐量 (请求/秒)",
                "架构A": metrics_a.throughput,
                "架构B": metrics_b.throughput,
                "差异": metrics_b.throughput - metrics_a.throughput,
                "差异百分比": ((metrics_b.throughput - metrics_a.throughput) / metrics_a.throughput * 100) if metrics_a.throughput > 0 else 0
            })
            
            report_data.append({
                "指标类型": "可靠性",
                "指标名称": "成功率 (%)",
                "架构A": metrics_a.success_rate * 100,
                "架构B": metrics_b.success_rate * 100,
                "差异": (metrics_b.success_rate - metrics_a.success_rate) * 100,
                "差异百分比": ((metrics_b.success_rate - metrics_a.success_rate) / metrics_a.success_rate * 100) if metrics_a.success_rate > 0 else 0
            })
        
        # 并发测试结果（合并之前的测试结果）
        all_concurrency_levels = set()
        
        # 添加当前测试的并发级别
        if concurrency_results:
            arch_a_concurrency = concurrency_results.get("arch_a", {})
            arch_b_concurrency = concurrency_results.get("arch_b", {})
            all_concurrency_levels.update(arch_a_concurrency.keys())
            all_concurrency_levels.update(arch_b_concurrency.keys())
        
        # 添加之前测试的并发级别（1, 5, 10）
        if previous_results is not None:
            import re
            for _, row in previous_results.iterrows():
                if '并发测试' in str(row.get('指标类型', '')) and '并发数=' in str(row.get('指标类型', '')):
                    match = re.search(r'并发数=(\d+)', str(row.get('指标类型', '')))
                    if match:
                        conc_level = int(match.group(1))
                        if conc_level in [1, 5, 10]:  # 只合并之前的1, 5, 10
                            all_concurrency_levels.add(conc_level)
        
        # 按并发级别排序，生成报告
        for concurrency in sorted(all_concurrency_levels):
            # 优先使用当前测试结果
            if concurrency_results and concurrency in arch_a_concurrency and concurrency in arch_b_concurrency:
                metrics_a_conc = arch_a_concurrency[concurrency]
                metrics_b_conc = arch_b_concurrency[concurrency]
                
                report_data.append({
                    "指标类型": f"并发测试 (并发数={concurrency})",
                    "指标名称": "平均延迟 (秒)",
                    "架构A": metrics_a_conc.latency_mean,
                    "架构B": metrics_b_conc.latency_mean,
                    "差异": metrics_b_conc.latency_mean - metrics_a_conc.latency_mean,
                    "差异百分比": ((metrics_b_conc.latency_mean - metrics_a_conc.latency_mean) / metrics_a_conc.latency_mean * 100) if metrics_a_conc.latency_mean > 0 else 0
                })
                
                report_data.append({
                    "指标类型": f"并发测试 (并发数={concurrency})",
                    "指标名称": "吞吐量 (请求/秒)",
                    "架构A": metrics_a_conc.throughput,
                    "架构B": metrics_b_conc.throughput,
                    "差异": metrics_b_conc.throughput - metrics_a_conc.throughput,
                    "差异百分比": ((metrics_b_conc.throughput - metrics_a_conc.throughput) / metrics_a_conc.throughput * 100) if metrics_a_conc.throughput > 0 else 0
                })
            # 如果没有当前测试结果，从之前的测试结果中提取
            elif previous_results is not None:
                import re
                prev_latency_a = prev_latency_b = None
                prev_throughput_a = prev_throughput_b = None
                
                for _, row in previous_results.iterrows():
                    if f'并发数={concurrency}' in str(row.get('指标类型', '')):
                        metric_name = row.get('指标名称', '')
                        if '平均延迟' in metric_name:
                            prev_latency_a = row.get('架构A', 0)
                            prev_latency_b = row.get('架构B', 0)
                        elif '吞吐量' in metric_name:
                            prev_throughput_a = row.get('架构A', 0)
                            prev_throughput_b = row.get('架构B', 0)
                
                if prev_latency_a is not None:
                    report_data.append({
                        "指标类型": f"并发测试 (并发数={concurrency})",
                        "指标名称": "平均延迟 (秒)",
                        "架构A": prev_latency_a,
                        "架构B": prev_latency_b,
                        "差异": prev_latency_b - prev_latency_a,
                        "差异百分比": ((prev_latency_b - prev_latency_a) / prev_latency_a * 100) if prev_latency_a > 0 else 0
                    })
                    
                    report_data.append({
                        "指标类型": f"并发测试 (并发数={concurrency})",
                        "指标名称": "吞吐量 (请求/秒)",
                        "架构A": prev_throughput_a,
                        "架构B": prev_throughput_b,
                        "差异": prev_throughput_b - prev_throughput_a,
                        "差异百分比": ((prev_throughput_b - prev_throughput_a) / prev_throughput_a * 100) if prev_throughput_a > 0 else 0
                    })
        
        # 保存为CSV（添加时间戳，避免覆盖之前的测试结果）
        df = pd.DataFrame(report_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = performance_test_dir / f"architecture_comparison_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"对比报告已保存: {csv_path}")
        
        # 生成可视化图表（使用相同的时间戳，确保文件关联）
        if latency_metrics is not None:
            metrics_a, metrics_b = latency_metrics
            self._plot_comparison(metrics_a, metrics_b, concurrency_results, timestamp, previous_results)
        else:
            # 如果没有延迟测试数据，只生成并发测试图表
            logger.info("跳过延迟测试图表生成（延迟测试未执行）")
            self._plot_concurrency_only(concurrency_results, timestamp, previous_results)
        
        return df
    
    def _plot_comparison(self, metrics_a: PerformanceMetrics, metrics_b: PerformanceMetrics,
                        concurrency_results: Dict[str, Dict[int, PerformanceMetrics]], 
                        timestamp: str = None, previous_results: Optional[pd.DataFrame] = None):
        """生成可视化对比图表"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 图1: 延迟指标对比
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('架构性能对比 - 延迟指标', fontsize=16, fontweight='bold')
        
        # 延迟指标柱状图
        latency_metrics = ['平均延迟', '中位数', 'P95', 'P99', '最小', '最大']
        values_a = [metrics_a.latency_mean, metrics_a.latency_median, metrics_a.latency_p95,
                   metrics_a.latency_p99, metrics_a.latency_min, metrics_a.latency_max]
        values_b = [metrics_b.latency_mean, metrics_b.latency_median, metrics_b.latency_p95,
                   metrics_b.latency_p99, metrics_b.latency_min, metrics_b.latency_max]
        
        x = np.arange(len(latency_metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, values_a, width, label=self.arch_a.name, color='#2E86AB')
        axes[0, 0].bar(x + width/2, values_b, width, label=self.arch_b.name, color='#A23B72')
        axes[0, 0].set_xlabel('延迟指标')
        axes[0, 0].set_ylabel('延迟 (秒)')
        axes[0, 0].set_title('延迟指标对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(latency_metrics, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 吞吐量对比
        axes[0, 1].bar([self.arch_a.name, self.arch_b.name], 
                      [metrics_a.throughput, metrics_b.throughput],
                      color=['#2E86AB', '#A23B72'])
        axes[0, 1].set_ylabel('吞吐量 (请求/秒)')
        axes[0, 1].set_title('吞吐量对比')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 成功率对比
        axes[1, 0].bar([self.arch_a.name, self.arch_b.name],
                      [metrics_a.success_rate * 100, metrics_b.success_rate * 100],
                      color=['#2E86AB', '#A23B72'])
        axes[1, 0].set_ylabel('成功率 (%)')
        axes[1, 0].set_title('成功率对比')
        axes[1, 0].set_ylim([0, 105])
        axes[1, 0].grid(True, alpha=0.3)
        
        # 并发测试结果（如果有）
        if concurrency_results:
            arch_a_conc = concurrency_results.get("arch_a", {})
            arch_b_conc = concurrency_results.get("arch_b", {})
            
            concurrencies = sorted(set(list(arch_a_conc.keys()) + list(arch_b_conc.keys())))
            throughputs_a = [arch_a_conc.get(c, PerformanceMetrics(0,0,0,0,0,0,0,0,0,0,0)).throughput for c in concurrencies]
            throughputs_b = [arch_b_conc.get(c, PerformanceMetrics(0,0,0,0,0,0,0,0,0,0,0)).throughput for c in concurrencies]
            
            axes[1, 1].plot(concurrencies, throughputs_a, marker='o', label=self.arch_a.name, color='#2E86AB', linewidth=2)
            axes[1, 1].plot(concurrencies, throughputs_b, marker='s', label=self.arch_b.name, color='#A23B72', linewidth=2)
            axes[1, 1].set_xlabel('并发数')
            axes[1, 1].set_ylabel('吞吐量 (请求/秒)')
            axes[1, 1].set_title('并发性能对比')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, '无并发测试数据', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('并发性能对比')
        
        plt.tight_layout()
        output_path = performance_test_dir / f"architecture_comparison_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"对比图表已保存: {output_path}")
        
        # 图2: 并发延迟对比（如果有并发测试数据）
        if concurrency_results:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            arch_a_conc = concurrency_results.get("arch_a", {})
            arch_b_conc = concurrency_results.get("arch_b", {})
            
            concurrencies = sorted(set(list(arch_a_conc.keys()) + list(arch_b_conc.keys())))
            latencies_a = [arch_a_conc.get(c, PerformanceMetrics(0,0,0,0,0,0,0,0,0,0,0)).latency_mean for c in concurrencies]
            latencies_b = [arch_b_conc.get(c, PerformanceMetrics(0,0,0,0,0,0,0,0,0,0,0)).latency_mean for c in concurrencies]
            
            ax.plot(concurrencies, latencies_a, marker='o', label=f'{self.arch_a.name} - 平均延迟', 
                   color='#2E86AB', linewidth=2, markersize=8)
            ax.plot(concurrencies, latencies_b, marker='s', label=f'{self.arch_b.name} - 平均延迟',
                   color='#A23B72', linewidth=2, markersize=8)
            ax.set_xlabel('并发数', fontsize=12, fontweight='bold')
            ax.set_ylabel('平均延迟 (秒)', fontsize=12, fontweight='bold')
            ax.set_title('不同并发级别下的延迟对比', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = performance_test_dir / f"concurrency_latency_comparison_{timestamp}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"并发延迟对比图已保存: {output_path}")
    
    def _plot_concurrency_only(self, concurrency_results: Dict[str, Dict[int, PerformanceMetrics]], 
                               timestamp: str = None, previous_results: Optional[pd.DataFrame] = None):
        """只生成并发测试图表（当延迟测试被注释时使用）"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not concurrency_results:
            logger.warning("没有并发测试数据，跳过图表生成")
            return
        
        # 并发延迟对比图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        arch_a_conc = concurrency_results.get("arch_a", {})
        arch_b_conc = concurrency_results.get("arch_b", {})
        
        concurrencies = sorted(set(list(arch_a_conc.keys()) + list(arch_b_conc.keys())))
        latencies_a = [arch_a_conc.get(c, PerformanceMetrics(0,0,0,0,0,0,0,0,0,0,0)).latency_mean for c in concurrencies]
        latencies_b = [arch_b_conc.get(c, PerformanceMetrics(0,0,0,0,0,0,0,0,0,0,0)).latency_mean for c in concurrencies]
        throughputs_a = [arch_a_conc.get(c, PerformanceMetrics(0,0,0,0,0,0,0,0,0,0,0)).throughput for c in concurrencies]
        throughputs_b = [arch_b_conc.get(c, PerformanceMetrics(0,0,0,0,0,0,0,0,0,0,0)).throughput for c in concurrencies]
        
        # 创建双y轴图表
        ax1 = ax
        ax2 = ax1.twinx()
        
        # 延迟线
        line1 = ax1.plot(concurrencies, latencies_a, marker='o', label=f'{self.arch_a.name} - 延迟', 
                       color='#2E86AB', linewidth=2, markersize=8)
        line2 = ax1.plot(concurrencies, latencies_b, marker='s', label=f'{self.arch_b.name} - 延迟',
                       color='#A23B72', linewidth=2, markersize=8)
        
        # 吞吐量线
        line3 = ax2.plot(concurrencies, throughputs_a, marker='^', label=f'{self.arch_a.name} - 吞吐量', 
                        color='#2E86AB', linewidth=2, markersize=8, linestyle='--')
        line4 = ax2.plot(concurrencies, throughputs_b, marker='v', label=f'{self.arch_b.name} - 吞吐量',
                        color='#A23B72', linewidth=2, markersize=8, linestyle='--')
        
        ax1.set_xlabel('并发数', fontsize=12, fontweight='bold')
        ax1.set_ylabel('平均延迟 (秒)', fontsize=12, fontweight='bold', color='#2E86AB')
        ax2.set_ylabel('吞吐量 (请求/秒)', fontsize=12, fontweight='bold', color='#A23B72')
        ax1.set_title('并发性能对比（延迟和吞吐量）', fontsize=14, fontweight='bold')
        
        # 合并图例
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = performance_test_dir / f"concurrency_performance_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"并发性能对比图已保存: {output_path}")


if __name__ == "__main__":
    # 从配置文件加载
    try:
        import config
        # 架构A：生产环境（可能是 OpenWebUI）
        prod_url = getattr(config, 'PROD_API_URL', '')
        prod_key = getattr(config, 'PROD_API_KEY', '')
        prod_knowledge_id = getattr(config, 'PROD_KNOWLEDGE_ID', None)
        use_ragflow_format = getattr(config, 'USE_RAGFLOW_FORMAT', True)
        use_ragflow_index = getattr(config, 'USE_RAGFLOW_INDEX', True)
        ragflow_timeout = getattr(config, 'RAGFLOW_TIMEOUT', 30)
        
        # 架构B：测试环境（标准 RagFlow API）
        test_url = getattr(config, 'TEST_API_URL', '')
        test_key = getattr(config, 'TEST_API_KEY', '')
    except ImportError:
        logger.error("找不到 config.py 配置文件")
        sys.exit(1)
    
    # 配置两种架构
    architecture_a = ArchitectureConfig(
        name="架构A (生产环境)",
        api_url=prod_url,
        api_key=prod_key,
        description="生产环境" + (" (OpenWebUI RAGFlow)" if prod_knowledge_id else " (标准 RagFlow API)"),
        knowledge_id=prod_knowledge_id,
        use_ragflow_format=use_ragflow_format if prod_knowledge_id else False,
        use_ragflow_index=use_ragflow_index if prod_knowledge_id else False,
        ragflow_timeout=ragflow_timeout if prod_knowledge_id else 30
    )
    
    architecture_b = ArchitectureConfig(
        name="架构B (测试环境)",
        api_url=test_url,
        api_key=test_key,
        description="测试环境 (标准 RagFlow API)"
    )
    
    logger.info("=" * 80)
    logger.info("架构性能对比测试")
    logger.info("=" * 80)
    logger.info(f"架构A: {architecture_a.description} ({architecture_a.api_url})")
    logger.info(f"架构B: {architecture_b.description} ({architecture_b.api_url})")
    logger.info("=" * 80)
    
    # 创建测试器
    tester = ArchitecturePerformanceTester(architecture_a, architecture_b)
    
    # 加载测试用例
    from pathlib import Path
    input_dir = Path("input")
    test_set_path = input_dir / "test.xlsx"
    
    if not test_set_path.exists():
        logger.error(f"测试数据文件不存在: {test_set_path}")
        sys.exit(1)
    
    temp_runner = EvaluationRunner(tester.client_a, RetrievalConfig(dataset_ids=[]))
    test_cases = temp_runner.load_test_set(str(test_set_path))
    
    # 限制测试用例数量为500个
    if len(test_cases) > 500:
        test_cases = test_cases[:500]
        logger.info(f"测试用例已限制为前500个（共{len(test_cases)}个）")
    else:
        logger.info(f"使用全部测试用例（共{len(test_cases)}个）")
    
    # 检索配置
    try:
        retrieval_config_dict = getattr(config, 'RETRIEVAL_CONFIG', {})
        retrieval_config = RetrievalConfig(
            dataset_ids=[],
            document_ids=[],
            top_k=retrieval_config_dict.get('top_k', 1024),
            similarity_threshold=retrieval_config_dict.get('similarity_threshold', 0.2),
            vector_similarity_weight=retrieval_config_dict.get('vector_similarity_weight', 0.3),
            rerank_id=retrieval_config_dict.get('rerank_id', ''),
            highlight=retrieval_config_dict.get('highlight', False),
            page=retrieval_config_dict.get('page', 1),
            page_size=retrieval_config_dict.get('page_size', 30)
        )
    except:
        retrieval_config = RetrievalConfig(
            dataset_ids=[],
            document_ids=[],
            top_k=1024,
            similarity_threshold=0.2,
            vector_similarity_weight=0.3
        )
    
    # 运行延迟测试（已注释，暂时不做）
    # logger.info("\n" + "=" * 80)
    # logger.info("阶段1: 延迟测试")
    # logger.info("=" * 80)
    # latency_metrics = tester.run_latency_test(test_cases, retrieval_config, warmup_rounds=2)
    latency_metrics = None  # 延迟测试暂时不做
    
    # 运行并发测试（测试更多并发级别，查看极限并发能力）
    logger.info("\n" + "=" * 80)
    logger.info("阶段2: 并发测试（极限并发能力测试）")
    logger.info("=" * 80)
    # 只测试新的并发级别，1,5,10之前已经测试过
    concurrency_levels = [15, 20, 30, 50]  # 只测试新的并发级别，与之前的1,5,10结果合并
    logger.info(f"将测试以下并发级别: {concurrency_levels}")
    logger.info("注意: 并发级别1, 5, 10之前已测试，本次只测试新级别，结果可与之前合并")
    logger.info("提示: 如果服务器CPU过高，可以按 Ctrl+C 安全停止，已完成的结果会被保存")
    # 并发测试使用前100个用例（如果用例数少于100则使用全部）
    concurrency_test_cases = test_cases[:min(100, len(test_cases))]
    logger.info(f"并发测试使用 {len(concurrency_test_cases)} 个测试用例")
    
    concurrency_results = None
    try:
        concurrency_results = tester.run_concurrency_test(
            concurrency_test_cases,
            retrieval_config,
            concurrency_levels=concurrency_levels
        )
    except KeyboardInterrupt:
        logger.warning("\n" + "=" * 80)
        logger.warning("测试被用户中断")
        logger.warning("=" * 80)
        # 获取已完成的结果（如果有）
        if hasattr(tester, '_last_concurrency_results'):
            concurrency_results = tester._last_concurrency_results
            logger.info("使用已完成的部分测试结果")
        else:
            concurrency_results = {"arch_a": {}, "arch_b": {}}
            logger.warning("没有已完成的结果可保存")
    
    # 生成对比报告（可以合并之前的测试结果）
    if concurrency_results and (concurrency_results.get("arch_a") or concurrency_results.get("arch_b")):
        logger.info("\n" + "=" * 80)
        logger.info("生成对比报告")
        logger.info("=" * 80)
        
        # 尝试合并之前的测试结果（如果有）
        previous_results = None
        previous_csv_files = list(performance_test_dir.glob("architecture_comparison_*.csv"))
        if previous_csv_files:
            # 找到最新的之前的测试结果文件
            previous_csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            previous_file = previous_csv_files[0]
            logger.info(f"发现之前的测试结果文件: {previous_file.name}")
            logger.info("将尝试合并之前的并发测试结果（并发级别1, 5, 10）")
            try:
                previous_df = pd.read_csv(previous_file, encoding='utf-8-sig')
                previous_results = previous_df
            except Exception as e:
                logger.warning(f"读取之前的测试结果失败: {e}")
        
        comparison_df = tester.generate_comparison_report(latency_metrics, concurrency_results, previous_results)
        
        # 打印摘要
        print("\n" + "=" * 80)
        print("架构性能对比测试完成（部分结果）" if concurrency_results and len(concurrency_results.get("arch_a", {})) < len(concurrency_levels) else "架构性能对比测试完成")
        print("=" * 80)
        print("\n关键指标对比:")
        print(comparison_df.to_string(index=False))
        print(f"\n详细报告已保存到: {performance_test_dir.absolute()}")
        print("注意: 所有测试结果文件都包含时间戳，不会覆盖之前的测试结果")
    else:
        logger.warning("没有足够的测试结果生成报告")
        print("\n测试已停止，没有生成报告")

