"""
超参数测试工具
用于批量测试不同的API参数配置，并生成可视化报告
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event
import logging

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入evaluation_tool中的类
from evaluation_tool import (
    RagFlowClient, RetrievalConfig, EvaluationRunner, TestCase
)

# 创建输出目录
output_dir = Path("output")
hyperparameter_output_dir = output_dir / "hyperparameter_test"
hyperparameter_output_dir.mkdir(parents=True, exist_ok=True)

# 配置简化的日志
class SimpleLogger:
    """简化的日志记录器，只显示当前运行到哪个问题"""
    def __init__(self):
        self.current_question_lock = Lock()
        self.current_question = 0
        self.total_questions = 0
        self.current_param = None
        self.total_params = 0
        self.param_index = 0
    
    def set_total_params(self, total: int):
        self.total_params = total
    
    def set_total_questions(self, total: int):
        self.total_questions = total
    
    def set_current_param(self, param_name: str, param_index: int):
        self.current_param = param_name
        self.param_index = param_index
    
    def update_question(self, question_index: int):
        with self.current_question_lock:
            self.current_question = question_index
            if self.current_param:
                print(f"\r参数 {self.param_index}/{self.total_params} ({self.current_param}): "
                      f"问题 {question_index}/{self.total_questions}", end='', flush=True)
            else:
                print(f"\r问题 {question_index}/{self.total_questions}", end='', flush=True)
    
    def finish(self):
        print()  # 换行

simple_logger = SimpleLogger()

# 配置标准日志（用于错误信息）
logging.basicConfig(
    level=logging.WARNING,  # 只显示警告和错误
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterTester:
    """超参数测试器"""
    
    def __init__(self, client: RagFlowClient, test_cases: List[TestCase], 
                 base_config: Optional[Dict[str, Any]] = None):
        """
        初始化超参数测试器
        
        参数:
            client: RagFlow客户端
            test_cases: 测试用例列表
            base_config: 基础配置字典，用于设置默认参数
        """
        self.client = client
        self.test_cases = test_cases
        self.base_config = base_config or {}
        self.results = []
        self.results_lock = Lock()
        self.stop_event = Event()  # 停止事件标志
    
    def test_single_parameter(self, param_name: str, param_value: Any, 
                              param_index: int, total_params: int) -> Optional[Dict[str, Any]]:
        """
        测试单个参数值
        
        参数:
            param_name: 参数名称（如 'vector_similarity_weight'）
            param_value: 参数值
            param_index: 当前参数索引
            total_params: 总参数数量
        
        返回:
            包含该参数值测试结果的字典，如果被停止则返回None
        """
        # 为日志添加参数索引前缀，便于在并行执行时追踪
        param_prefix = f"[Param {param_index}/{total_params}]"
        
        # 检查停止信号
        if self.stop_event.is_set():
            print(f"\n{param_prefix} ({param_name}={param_value}): 已收到停止信号，跳过测试")
            return None
        
        print(f"\n{param_prefix} {'='*80}")
        print(f"{param_prefix} 开始测试参数: {param_name}={param_value}")
        print(f"{param_prefix} {'='*80}")
        
        # 创建该参数值的输出目录
        param_output_dir = hyperparameter_output_dir / f"{param_name}_{param_value}"
        param_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建配置
        config_dict = self.base_config.copy()
        config_dict[param_name] = param_value
        
        # 创建RetrievalConfig
        retrieval_config = RetrievalConfig(
            dataset_ids=config_dict.get('dataset_ids', []),
            document_ids=config_dict.get('document_ids', []),
            top_k=config_dict.get('top_k', 1024),
            similarity_threshold=config_dict.get('similarity_threshold', 0.2),
            vector_similarity_weight=config_dict.get('vector_similarity_weight'),
            rerank_id=config_dict.get('rerank_id', ''),
            highlight=config_dict.get('highlight', False),
            page=config_dict.get('page', 1),
            page_size=config_dict.get('page_size', 30)
        )
        
        # 创建自定义的EvaluationRunner，禁用API响应保存
        runner = CustomEvaluationRunner(
            self.client, retrieval_config, output_base_dir=param_output_dir
        )
        
        # 设置日志
        simple_logger.set_current_param(f"{param_name}={param_value}", param_index)
        simple_logger.set_total_questions(len(self.test_cases))
        
        # 运行测试（使用多线程）
        max_workers = self.base_config.get('max_workers', 5)
        delay = self.base_config.get('delay_between_requests', 0.5)
        
        try:
            results_df = runner.run_batch_evaluation(
                self.test_cases,
                max_workers=max_workers,
                delay_between_requests=delay
            )
        except KeyboardInterrupt:
            # 如果收到中断信号，检查是否应该停止
            if self.stop_event.is_set():
                print(f"\n{param_prefix} ({param_name}={param_value}): 测试被中断")
                return None
            # 否则继续（让 evaluation_tool 处理）
            raise
        
        # 再次检查停止信号
        if self.stop_event.is_set():
            print(f"\n{param_prefix} ({param_name}={param_value}): 测试过程中收到停止信号")
            return None
        
        print(f"{param_prefix} 测试完成，正在计算指标...")
        
        # 计算平均指标
        metrics = {
            'parameter_name': param_name,
            'parameter_value': param_value,
            'accuracy': results_df['accuracy'].mean() if 'accuracy' in results_df.columns else 0.0,
            'recall': results_df['recall'].mean() if 'recall' in results_df.columns else 0.0,
            'recall@3': results_df['recall@3'].mean() if 'recall@3' in results_df.columns else 0.0,
            'recall@5': results_df['recall@5'].mean() if 'recall@5' in results_df.columns else 0.0,
            'recall@10': results_df['recall@10'].mean() if 'recall@10' in results_df.columns else 0.0,
            'latency_avg': results_df['latency'].mean() if 'latency' in results_df.columns else 0.0,
            'latency_total': results_df['latency'].sum() if 'latency' in results_df.columns else 0.0,
        }
        
        # 保存该参数的结果
        csv_path = param_output_dir / "results.csv"
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"{param_prefix} 结果已保存: {csv_path}")
        print(f"{param_prefix} 指标: accuracy={metrics['accuracy']:.4f}, recall={metrics['recall']:.4f}, recall@10={metrics['recall@10']:.4f}")
        print(f"{param_prefix} {'='*80}\n")
        
        return metrics
    
    def test_parameter_values(self, param_name: str, param_values: List[Any]) -> pd.DataFrame:
        """
        测试多个参数值（参数值之间按顺序执行，每个参数值内部并行处理测试用例）
        
        参数:
            param_name: 参数名称
            param_values: 参数值列表
        
        返回:
            包含所有测试结果的DataFrame
        
        注意:
            - 参数值之间按顺序执行（不并行）
            - 每个参数值内部使用 base_config 中的 max_workers 进行并行处理测试用例
        """
        total_params = len(param_values)
        simple_logger.set_total_params(total_params)
        
        results = []
        
        # 获取每个参数值内部的并发数（用于处理测试用例）
        internal_max_workers = self.base_config.get('max_workers', 5)
        print(f"[进度] 开始测试 {param_name}，共 {total_params} 个值")
        print(f"[进度] 参数值之间按顺序执行，每个参数值内部使用 {internal_max_workers} 个worker并行处理测试用例")
        print(f"提示: 按 Ctrl+C 可以中断执行，已完成的结果会被保存")
        
        try:
            # 顺序测试每个参数值（参数值之间不并行）
            for idx, value in enumerate(param_values, 1):
                if self.stop_event.is_set():
                    print("\n[进度] 收到停止信号，停止测试")
                    break
                
                try:
                    result = self.test_single_parameter(param_name, value, idx, total_params)
                    if result is not None:  # 忽略被停止的任务
                        results.append(result)
                        print(f"[进度] 已完成 {idx}/{total_params} 个参数值的测试")
                except KeyboardInterrupt:
                    print("\n[进度] 收到中断信号 (Ctrl+C)，正在停止...")
                    self.stop_event.set()
                    break
                except Exception as e:
                    logger.error(f"[Param {idx}/{total_params}] 测试参数值 {value} 失败: {str(e)}")
        except KeyboardInterrupt:
            print("\n[进度] 收到中断信号 (Ctrl+C)，正在停止...")
            self.stop_event.set()
            # 等待当前任务完成
            print("[进度] 等待当前任务完成...")
            time.sleep(2)
            print("[进度] 执行被用户中断，已完成部分结果")
        
        simple_logger.finish()
        
        # 按参数值排序（过滤掉None值）
        results = [r for r in results if r is not None]
        if results:
            results.sort(key=lambda x: x['parameter_value'])
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()  # 返回空的DataFrame
    
    def generate_visualization(self, results_df: pd.DataFrame, param_name: str):
        """
        生成可视化图表（忽略延迟，突出显示不同参数值的差异）
        
        参数:
            results_df: 测试结果DataFrame
            param_name: 参数名称
        """
        if results_df.empty:
            logger.warning("没有结果数据，无法生成图表")
            return
        
        # 准备数据
        param_values = results_df['parameter_value'].values
        metrics = ['accuracy', 'recall', 'recall@3', 'recall@5', 'recall@10']
        
        # 创建图表 - 2行3列，只显示5个指标（不包含延迟）
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'超参数测试结果对比（忽略延迟）: {param_name}', fontsize=16, fontweight='bold')
        
        # 绘制各个指标
        axes_flat = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes_flat[idx]
            values = results_df[metric].values
            
            # 绘制折线图，使用更粗的线条和更大的标记
            ax.plot(param_values, values, marker='o', linewidth=3, markersize=12, 
                   color='#2E86AB', markerfacecolor='#A23B72', markeredgewidth=2.5,
                   markeredgecolor='white', label=metric.upper())
            
            # 填充区域，突出显示趋势
            ax.fill_between(param_values, values, alpha=0.3, color='#2E86AB')
            
            # 网格
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # 标签和标题
            ax.set_xlabel(f'{param_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
            ax.set_title(f'{metric.upper()} 随参数值变化', fontsize=13, fontweight='bold')
            
            # 设置y轴范围，让差异更明显
            value_range = values.max() - values.min()
            ax.set_ylim([max(0, values.min() - value_range * 0.1), 
                        min(1.05, values.max() + value_range * 0.1)])
            
            # 添加数值标注，使用更明显的字体
            for i, (x, y) in enumerate(zip(param_values, values)):
                ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                           xytext=(0, 15), ha='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # 标记最佳值
            best_idx = values.argmax()
            best_x = param_values[best_idx]
            best_y = values[best_idx]
            ax.plot(best_x, best_y, marker='*', markersize=25, color='gold', 
                   markeredgecolor='red', markeredgewidth=3, zorder=5)
            ax.annotate(f'最佳: {best_x}\n{best_y:.3f}', 
                       xy=(best_x, best_y),
                       xytext=(15, 25), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=2),
                       fontsize=10, fontweight='bold')
        
        # 第6个子图：显示差异对比（最大值-最小值）
        ax_diff = axes_flat[5]
        differences = []
        diff_labels = []
        for metric in metrics:
            values = results_df[metric].values
            diff = values.max() - values.min()
            differences.append(diff)
            diff_labels.append(metric.upper())
        
        bars = ax_diff.barh(diff_labels, differences, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
        ax_diff.set_xlabel('差异范围 (最大值 - 最小值)', fontsize=12, fontweight='bold')
        ax_diff.set_title('各指标在不同参数值下的差异范围', fontsize=13, fontweight='bold')
        ax_diff.grid(True, alpha=0.3, linestyle='--', axis='x')
        
        # 添加数值标注
        for i, (bar, diff) in enumerate(zip(bars, differences)):
            ax_diff.text(diff, i, f' {diff:.4f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        output_path = hyperparameter_output_dir / f'{param_name}_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"详细指标图表已保存: {output_path}")
        
        # 创建综合对比图
        self._create_comparison_chart(results_df, param_name)
    
    def _create_comparison_chart(self, results_df: pd.DataFrame, param_name: str):
        """创建综合对比图（突出显示不同参数值的差异）"""
        param_values = results_df['parameter_value'].values
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 准备数据
        metrics = ['accuracy', 'recall', 'recall@3', 'recall@5', 'recall@10']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        line_styles = ['-', '--', '-.', ':', '-']
        
        # 绘制多条线，使用不同的线型和颜色
        for metric, color, linestyle in zip(metrics, colors, line_styles):
            values = results_df[metric].values
            ax.plot(param_values, values, marker='o', linewidth=3.5, markersize=14,
                   label=metric.upper(), color=color, linestyle=linestyle, alpha=0.9,
                   markeredgecolor='white', markeredgewidth=2.5)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel(f'{param_name}', fontsize=15, fontweight='bold')
        ax.set_ylabel('Score', fontsize=15, fontweight='bold')
        ax.set_title(f'超参数测试综合对比（忽略延迟）: {param_name}', fontsize=17, fontweight='bold')
        ax.legend(loc='lower left', fontsize=12, framealpha=0.95, ncol=2, 
                 columnspacing=1, handlelength=3)
        
        # 设置y轴范围，让差异更明显
        all_values = []
        for metric in metrics:
            all_values.extend(results_df[metric].values)
        min_val = min(all_values)
        max_val = max(all_values)
        value_range = max_val - min_val
        ax.set_ylim([max(0, min_val - value_range * 0.1), 
                     min(1.05, max_val + value_range * 0.1)])
        
        # 添加数值标注（只标注关键点：最大值和最小值）
        for metric in metrics:
            values = results_df[metric].values
            max_idx = values.argmax()
            min_idx = values.argmin()
            
            # 标注最大值
            ax.annotate(f'{values[max_idx]:.3f}',
                       xy=(param_values[max_idx], values[max_idx]),
                       xytext=(0, 20), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9),
                       arrowprops=dict(arrowstyle='->', lw=1.5))
        
        # 添加垂直线标记每个参数值，让差异更明显
        for x in param_values:
            ax.axvline(x=x, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        
        output_path = hyperparameter_output_dir / f'{param_name}_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"综合对比图已保存: {output_path}")


class CustomEvaluationRunner(EvaluationRunner):
    """自定义的EvaluationRunner，禁用API响应保存，简化日志"""
    
    def _save_api_response(self, test_case: TestCase, response: Dict[str, Any], test_index: int):
        """重写：不保存API响应"""
        pass
    
    def run_single_test(self, test_case: TestCase, test_index: int = 0) -> Dict[str, Any]:
        """重写：简化日志输出，只显示进度"""
        # 更新进度
        simple_logger.update_question(test_index)
        
        # 临时禁用日志输出
        import logging
        old_level = logging.getLogger().level
        # 获取evaluation_tool的logger并设置级别
        eval_logger = logging.getLogger('evaluation_tool')
        old_eval_level = eval_logger.level
        logging.getLogger().setLevel(logging.ERROR)
        eval_logger.setLevel(logging.ERROR)
        
        try:
            # 调用父类方法
            result = super().run_single_test(test_case, test_index)
        finally:
            # 恢复日志级别
            logging.getLogger().setLevel(old_level)
            eval_logger.setLevel(old_eval_level)
        
        return result
    
    def run_batch_evaluation(self, test_cases, max_workers: int = 1, delay_between_requests: float = 0.5):
        """重写：简化批量评测的日志输出"""
        # 临时禁用日志
        import logging
        old_level = logging.getLogger().level
        eval_logger = logging.getLogger('evaluation_tool')
        old_eval_level = eval_logger.level
        logging.getLogger().setLevel(logging.ERROR)
        eval_logger.setLevel(logging.ERROR)
        
        try:
            # 调用父类方法
            result = super().run_batch_evaluation(test_cases, max_workers, delay_between_requests)
        finally:
            # 恢复日志级别
            logging.getLogger().setLevel(old_level)
            eval_logger.setLevel(old_eval_level)
        
        return result


def main():
    """主函数"""
    # 加载配置
    try:
        import config
        env = getattr(config, 'DEFAULT_ENV', 'test').lower()
        
        if env == 'prod':
            api_url = getattr(config, 'PROD_API_URL', '')
            api_key = getattr(config, 'PROD_API_KEY', '')
        else:
            api_url = getattr(config, 'TEST_API_URL', '')
            api_key = getattr(config, 'TEST_API_KEY', '')
        
        if not api_url or not api_key:
            raise ValueError("配置文件中的 API URL 或 API Key 为空")
        
        client = RagFlowClient(api_url=api_url, api_key=api_key)
    except Exception as e:
        print(f"加载配置失败: {str(e)}")
        sys.exit(1)
    
    # 加载测试集
    input_dir = Path("input")
    test_set_path = input_dir / "test.xlsx"
    if not test_set_path.exists():
        print(f"错误: 测试数据文件不存在，请将 test.xlsx 放到 {input_dir} 目录下")
        sys.exit(1)
    
    # 创建临时runner来加载测试集
    temp_runner = EvaluationRunner(client, RetrievalConfig(dataset_ids=[]))
    test_cases = temp_runner.load_test_set(str(test_set_path))
    
    print(f"已加载 {len(test_cases)} 条测试用例")
    
    # 从配置文件读取并发参数
    try:
        max_workers = getattr(config, 'MAX_WORKERS', 5)  # 默认值：5
        delay_between_requests = getattr(config, 'DELAY_BETWEEN_REQUESTS', 0.5)  # 默认值：0.5秒
        print(f"从配置文件读取: MAX_WORKERS={max_workers}, DELAY_BETWEEN_REQUESTS={delay_between_requests}")
    except AttributeError:
        # 如果配置文件中没有这些参数，使用默认值
        max_workers = 5
        delay_between_requests = 0.5
        print(f"使用默认值: MAX_WORKERS={max_workers}, DELAY_BETWEEN_REQUESTS={delay_between_requests}")
    
    # 配置基础参数
    base_config = {
        'dataset_ids': [],  # 留空，自动加载
        'document_ids': [],  # 留空，自动加载
        'top_k': 1024,
        'similarity_threshold': 0.2,
        'rerank_id': '',
        'highlight': False,
        'page': 1,
        'page_size': 30,
        'max_workers': max_workers,  # 从配置文件读取
        'delay_between_requests': delay_between_requests  # 从配置文件读取
    }
    
    # 创建超参数测试器
    tester = HyperparameterTester(client, test_cases, base_config)
    
    # 测试 vector_similarity_weight 参数
    # 从0.1到1.0，选择一些有代表性的值
    vector_similarity_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print(f"\n开始测试 vector_similarity_weight 参数")
    print(f"测试值: {vector_similarity_weights}")
    print(f"输出目录: {hyperparameter_output_dir.absolute()}\n")
    
    start_time = time.time()
    
    # 测试参数值（参数值之间按顺序执行，每个参数值内部并行处理测试用例）
    # 注意：每个参数值内部的测试用例并发数已在 base_config 中配置（max_workers）
    try:
        results_df = tester.test_parameter_values(
            'vector_similarity_weight',
            vector_similarity_weights
        )
    except KeyboardInterrupt:
        print("\n[进度] 收到中断信号，正在保存已完成的结果...")
        # 获取已完成的结果（test_parameter_values 内部已处理中断）
        results_df = pd.DataFrame()  # 初始化为空，让后续代码处理
    
    elapsed_time = time.time() - start_time
    
    # 保存结果（即使为空也要保存）
    results_csv_path = hyperparameter_output_dir / 'hyperparameter_test_results.csv'
    if not results_df.empty:
        results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n[进度] 测试结果已保存: {results_csv_path}")
        print(f"[进度] 已完成 {len(results_df)} 个参数值的测试")
        
        # 生成可视化（仅当有结果时）
        print("\n[进度] 生成可视化图表...")
        try:
            tester.generate_visualization(results_df, 'vector_similarity_weight')
        except Exception as e:
            logger.warning(f"[进度] 生成可视化图表失败: {str(e)}")
        
        # 打印摘要
        print("\n" + "=" * 80)
        if tester.stop_event.is_set():
            print("[摘要] 测试摘要（部分完成）")
        else:
            print("[摘要] 测试摘要")
        print("=" * 80)
        print(f"[摘要] 总耗时: {elapsed_time:.2f} 秒")
        print(f"[摘要] 最佳参数值（按各项指标）:")
        
        for metric in ['accuracy', 'recall', 'recall@3', 'recall@5', 'recall@10']:
            if metric in results_df.columns and len(results_df) > 0:
                best_idx = results_df[metric].idxmax()
                best_value = results_df.loc[best_idx, 'parameter_value']
                best_score = results_df.loc[best_idx, metric]
                print(f"[摘要]   {metric.upper()}: {best_value} (得分: {best_score:.4f})")
        
        print("\n[摘要] 完整结果:")
        print(results_df.to_string(index=False))
    else:
        print("\n[进度] 警告: 没有完成任何测试，无法保存结果")
        if tester.stop_event.is_set():
            print("[进度] 执行被用户中断，没有完成任何参数值的测试")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

