"""
评估指标分析工具 - OpenWebUI插件
用于分析评估结果中的总体指标和分类指标（按theme和type分类）
主要关注准确率和召回率

数据已嵌入，无需额外文件
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# ============================================================================
# 嵌入的评估数据（从CSV转换而来）
# 自动生成，请勿手动修改
# ============================================================================
# EVALUATION_DATA 将在这里插入
EVALUATION_DATA = []  # 占位符，生成脚本会替换为实际数据

class Tools:
    """评估指标分析工具集"""
    
    def __init__(self):
        """初始化工具，从嵌入数据加载评估结果"""
        try:
            self._df = pd.DataFrame(EVALUATION_DATA)
            print(f"成功加载评估数据: {len(self._df)} 条记录")
        except Exception as e:
            raise Exception(f"加载评估数据失败: {str(e)}")
    
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """计算总体指标"""
        if self._df is None or len(self._df) == 0:
            return {}
        
        total_count = len(self._df)
        avg_accuracy = self._df['accuracy'].mean()
        avg_recall = self._df['recall'].mean()
        
        avg_recall_at_3 = self._df['recall@3'].mean() if 'recall@3' in self._df.columns else None
        avg_recall_at_5 = self._df['recall@5'].mean() if 'recall@5' in self._df.columns else None
        avg_recall_at_10 = self._df['recall@10'].mean() if 'recall@10' in self._df.columns else None
        
        top1_theme_match_rate = self._df['top1_theme_match'].mean() if 'top1_theme_match' in self._df.columns else None
        top1_chapter_match_rate = self._df['top1_chapter_match'].mean() if 'top1_chapter_match' in self._df.columns else None
        top1_both_match_rate = self._df['top1_both_match'].mean() if 'top1_both_match' in self._df.columns else None
        
        avg_latency = self._df['latency'].mean() if 'latency' in self._df.columns else None
        
        return {
            "total_count": int(total_count),
            "average_accuracy": float(avg_accuracy),
            "average_recall": float(avg_recall),
            "average_recall_at_3": float(avg_recall_at_3) if avg_recall_at_3 is not None else None,
            "average_recall_at_5": float(avg_recall_at_5) if avg_recall_at_5 is not None else None,
            "average_recall_at_10": float(avg_recall_at_10) if avg_recall_at_10 is not None else None,
            "top1_theme_match_rate": float(top1_theme_match_rate) if top1_theme_match_rate is not None else None,
            "top1_chapter_match_rate": float(top1_chapter_match_rate) if top1_chapter_match_rate is not None else None,
            "top1_both_match_rate": float(top1_both_match_rate) if top1_both_match_rate is not None else None,
            "average_latency": float(avg_latency) if avg_latency is not None else None
        }
    
    def _calculate_metrics_by_theme(self) -> Dict[str, Dict[str, Any]]:
        """按theme分类计算指标"""
        if self._df is None or 'theme' not in self._df.columns:
            return {}
        
        theme_metrics = {}
        for theme in self._df['theme'].unique():
            theme_df = self._df[self._df['theme'] == theme]
            theme_count = len(theme_df)
            
            avg_accuracy = theme_df['accuracy'].mean()
            avg_recall = theme_df['recall'].mean()
            
            avg_recall_at_3 = theme_df['recall@3'].mean() if 'recall@3' in theme_df.columns else None
            avg_recall_at_5 = theme_df['recall@5'].mean() if 'recall@5' in theme_df.columns else None
            avg_recall_at_10 = theme_df['recall@10'].mean() if 'recall@10' in theme_df.columns else None
            
            theme_metrics[theme] = {
                "count": int(theme_count),
                "average_accuracy": float(avg_accuracy),
                "average_recall": float(avg_recall),
                "average_recall_at_3": float(avg_recall_at_3) if avg_recall_at_3 is not None else None,
                "average_recall_at_5": float(avg_recall_at_5) if avg_recall_at_5 is not None else None,
                "average_recall_at_10": float(avg_recall_at_10) if avg_recall_at_10 is not None else None,
            }
        
        return theme_metrics
    
    def _calculate_metrics_by_type(self) -> Dict[str, Dict[str, Any]]:
        """按type分类计算指标"""
        if self._df is None or 'type' not in self._df.columns:
            return {}
        
        type_metrics = {}
        for type_name in self._df['type'].unique():
            type_df = self._df[self._df['type'] == type_name]
            type_count = len(type_df)
            
            avg_accuracy = type_df['accuracy'].mean()
            avg_recall = type_df['recall'].mean()
            
            avg_recall_at_3 = type_df['recall@3'].mean() if 'recall@3' in type_df.columns else None
            avg_recall_at_5 = type_df['recall@5'].mean() if 'recall@5' in type_df.columns else None
            avg_recall_at_10 = type_df['recall@10'].mean() if 'recall@10' in type_df.columns else None
            
            type_metrics[type_name] = {
                "count": int(type_count),
                "average_accuracy": float(avg_accuracy),
                "average_recall": float(avg_recall),
                "average_recall_at_3": float(avg_recall_at_3) if avg_recall_at_3 is not None else None,
                "average_recall_at_5": float(avg_recall_at_5) if avg_recall_at_5 is not None else None,
                "average_recall_at_10": float(avg_recall_at_10) if avg_recall_at_10 is not None else None,
            }
        
        return type_metrics
    
    def get_overall_metrics(self) -> str:
        """
        Get overall evaluation metrics including accuracy and recall.
        Returns summary statistics for all evaluation results.
        """
        if self._df is None or len(self._df) == 0:
            return "错误: 评估数据为空。"
        
        metrics = self._calculate_overall_metrics()
        
        if not metrics:
            return "错误: 无法计算指标，数据可能为空。"
        
        result = "=" * 80 + "\n"
        result += "【总体评估指标】\n"
        result += "=" * 80 + "\n\n"
        result += f"总样本数: {metrics.get('total_count', 0)}\n"
        result += f"平均准确率: {metrics.get('average_accuracy', 0):.4f} ({metrics.get('average_accuracy', 0)*100:.2f}%)\n"
        result += f"平均召回率: {metrics.get('average_recall', 0):.4f} ({metrics.get('average_recall', 0)*100:.2f}%)\n\n"
        
        if metrics.get('average_recall_at_3') is not None:
            result += f"平均Recall@3: {metrics.get('average_recall_at_3', 0):.4f} ({metrics.get('average_recall_at_3', 0)*100:.2f}%)\n"
        if metrics.get('average_recall_at_5') is not None:
            result += f"平均Recall@5: {metrics.get('average_recall_at_5', 0):.4f} ({metrics.get('average_recall_at_5', 0)*100:.2f}%)\n"
        if metrics.get('average_recall_at_10') is not None:
            result += f"平均Recall@10: {metrics.get('average_recall_at_10', 0):.4f} ({metrics.get('average_recall_at_10', 0)*100:.2f}%)\n"
        
        if metrics.get('top1_theme_match_rate') is not None:
            result += f"\nTop1主题匹配率: {metrics.get('top1_theme_match_rate', 0):.4f} ({metrics.get('top1_theme_match_rate', 0)*100:.2f}%)\n"
        if metrics.get('top1_chapter_match_rate') is not None:
            result += f"Top1章节匹配率: {metrics.get('top1_chapter_match_rate', 0):.4f} ({metrics.get('top1_chapter_match_rate', 0)*100:.2f}%)\n"
        if metrics.get('top1_both_match_rate') is not None:
            result += f"Top1完全匹配率: {metrics.get('top1_both_match_rate', 0):.4f} ({metrics.get('top1_both_match_rate', 0)*100:.2f}%)\n"
        
        if metrics.get('average_latency') is not None:
            result += f"\n平均延迟: {metrics.get('average_latency', 0):.4f} 秒\n"
        
        result += "\n" + "=" * 80
        
        return result
    
    def get_metrics_by_theme(
        self,
        theme: Optional[str] = Field(None, description="Filter metrics by specific theme. Leave empty to get all themes.")
    ) -> str:
        """
        Get evaluation metrics grouped by theme. Returns accuracy and recall for each theme category.
        """
        if self._df is None or len(self._df) == 0:
            return "错误: 评估数据为空。"
        
        theme_metrics = self._calculate_metrics_by_theme()
        
        if not theme_metrics:
            return "错误: 无法计算按主题分类的指标，数据可能为空或缺少theme列。"
        
        result = "=" * 80 + "\n"
        result += "【按Theme分类的评估指标】\n"
        result += "=" * 80 + "\n\n"
        
        if theme and theme in theme_metrics:
            # 只返回指定theme的指标
            metrics = theme_metrics[theme]
            result += f"Theme: {theme}\n"
            result += f"  样本数: {metrics.get('count', 0)}\n"
            result += f"  平均准确率: {metrics.get('average_accuracy', 0):.4f} ({metrics.get('average_accuracy', 0)*100:.2f}%)\n"
            result += f"  平均召回率: {metrics.get('average_recall', 0):.4f} ({metrics.get('average_recall', 0)*100:.2f}%)\n"
            if metrics.get('average_recall_at_3') is not None:
                result += f"  平均Recall@3: {metrics.get('average_recall_at_3', 0):.4f} ({metrics.get('average_recall_at_3', 0)*100:.2f}%)\n"
            if metrics.get('average_recall_at_5') is not None:
                result += f"  平均Recall@5: {metrics.get('average_recall_at_5', 0):.4f} ({metrics.get('average_recall_at_5', 0)*100:.2f}%)\n"
            if metrics.get('average_recall_at_10') is not None:
                result += f"  平均Recall@10: {metrics.get('average_recall_at_10', 0):.4f} ({metrics.get('average_recall_at_10', 0)*100:.2f}%)\n"
        else:
            # 返回所有theme的指标
            for theme_name, metrics in sorted(theme_metrics.items()):
                result += f"\nTheme: {theme_name}\n"
                result += f"  样本数: {metrics.get('count', 0)}\n"
                result += f"  平均准确率: {metrics.get('average_accuracy', 0):.4f} ({metrics.get('average_accuracy', 0)*100:.2f}%)\n"
                result += f"  平均召回率: {metrics.get('average_recall', 0):.4f} ({metrics.get('average_recall', 0)*100:.2f}%)\n"
                if metrics.get('average_recall_at_3') is not None:
                    result += f"  平均Recall@3: {metrics.get('average_recall_at_3', 0):.4f} ({metrics.get('average_recall_at_3', 0)*100:.2f}%)\n"
                if metrics.get('average_recall_at_5') is not None:
                    result += f"  平均Recall@5: {metrics.get('average_recall_at_5', 0):.4f} ({metrics.get('average_recall_at_5', 0)*100:.2f}%)\n"
                if metrics.get('average_recall_at_10') is not None:
                    result += f"  平均Recall@10: {metrics.get('average_recall_at_10', 0):.4f} ({metrics.get('average_recall_at_10', 0)*100:.2f}%)\n"
        
        result += "\n" + "=" * 80
        
        return result
    
    def get_metrics_by_type(
        self,
        type_name: Optional[str] = Field(None, description="Filter metrics by specific type. Leave empty to get all types.")
    ) -> str:
        """
        Get evaluation metrics grouped by type. Returns accuracy and recall for each type category.
        """
        if self._df is None or len(self._df) == 0:
            return "错误: 评估数据为空。"
        
        type_metrics = self._calculate_metrics_by_type()
        
        if not type_metrics:
            return "错误: 无法计算按类型分类的指标，数据可能为空或缺少type列。"
        
        result = "=" * 80 + "\n"
        result += "【按Type分类的评估指标】\n"
        result += "=" * 80 + "\n\n"
        
        if type_name and type_name in type_metrics:
            # 只返回指定type的指标
            metrics = type_metrics[type_name]
            result += f"Type: {type_name}\n"
            result += f"  样本数: {metrics.get('count', 0)}\n"
            result += f"  平均准确率: {metrics.get('average_accuracy', 0):.4f} ({metrics.get('average_accuracy', 0)*100:.2f}%)\n"
            result += f"  平均召回率: {metrics.get('average_recall', 0):.4f} ({metrics.get('average_recall', 0)*100:.2f}%)\n"
            if metrics.get('average_recall_at_3') is not None:
                result += f"  平均Recall@3: {metrics.get('average_recall_at_3', 0):.4f} ({metrics.get('average_recall_at_3', 0)*100:.2f}%)\n"
            if metrics.get('average_recall_at_5') is not None:
                result += f"  平均Recall@5: {metrics.get('average_recall_at_5', 0):.4f} ({metrics.get('average_recall_at_5', 0)*100:.2f}%)\n"
            if metrics.get('average_recall_at_10') is not None:
                result += f"  平均Recall@10: {metrics.get('average_recall_at_10', 0):.4f} ({metrics.get('average_recall_at_10', 0)*100:.2f}%)\n"
        else:
            # 返回所有type的指标
            for type_val, metrics in sorted(type_metrics.items()):
                result += f"\nType: {type_val}\n"
                result += f"  样本数: {metrics.get('count', 0)}\n"
                result += f"  平均准确率: {metrics.get('average_accuracy', 0):.4f} ({metrics.get('average_accuracy', 0)*100:.2f}%)\n"
                result += f"  平均召回率: {metrics.get('average_recall', 0):.4f} ({metrics.get('average_recall', 0)*100:.2f}%)\n"
                if metrics.get('average_recall_at_3') is not None:
                    result += f"  平均Recall@3: {metrics.get('average_recall_at_3', 0):.4f} ({metrics.get('average_recall_at_3', 0)*100:.2f}%)\n"
                if metrics.get('average_recall_at_5') is not None:
                    result += f"  平均Recall@5: {metrics.get('average_recall_at_5', 0):.4f} ({metrics.get('average_recall_at_5', 0)*100:.2f}%)\n"
                if metrics.get('average_recall_at_10') is not None:
                    result += f"  平均Recall@10: {metrics.get('average_recall_at_10', 0):.4f} ({metrics.get('average_recall_at_10', 0)*100:.2f}%)\n"
        
        result += "\n" + "=" * 80
        
        return result
    
    def get_metrics_by_theme_and_type(
        self,
        theme: Optional[str] = Field(None, description="Filter by specific theme. Leave empty to get all themes."),
        type_name: Optional[str] = Field(None, description="Filter by specific type. Leave empty to get all types.")
    ) -> str:
        """
        Get evaluation metrics grouped by both theme and type combination. Returns accuracy and recall for each theme-type combination.
        """
        if self._df is None or len(self._df) == 0:
            return "错误: 评估数据为空。"
        
        if 'theme' not in self._df.columns or 'type' not in self._df.columns:
            return "错误: 数据中缺少theme或type列。"
        
        result = "=" * 80 + "\n"
        result += "【按Theme和Type组合分类的评估指标】\n"
        result += "=" * 80 + "\n\n"
        
        # 根据筛选条件过滤数据
        filtered_df = self._df.copy()
        if theme:
            filtered_df = filtered_df[filtered_df['theme'] == theme]
        if type_name:
            filtered_df = filtered_df[filtered_df['type'] == type_name]
        
        if len(filtered_df) == 0:
            return f"错误: 没有找到匹配theme={theme}, type={type_name}的数据。"
        
        # 按theme和type组合分组
        for theme_val in sorted(filtered_df['theme'].unique()):
            theme_df = filtered_df[filtered_df['theme'] == theme_val]
            result += f"\nTheme: {theme_val}\n"
            result += "-" * 60 + "\n"
            
            for type_val in sorted(theme_df['type'].unique()):
                combined_df = theme_df[theme_df['type'] == type_val]
                combined_count = len(combined_df)
                
                avg_accuracy = combined_df['accuracy'].mean()
                avg_recall = combined_df['recall'].mean()
                
                avg_recall_at_3 = combined_df['recall@3'].mean() if 'recall@3' in combined_df.columns else None
                avg_recall_at_5 = combined_df['recall@5'].mean() if 'recall@5' in combined_df.columns else None
                avg_recall_at_10 = combined_df['recall@10'].mean() if 'recall@10' in combined_df.columns else None
                
                result += f"  Type: {type_val}\n"
                result += f"    样本数: {combined_count}\n"
                result += f"    平均准确率: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)\n"
                result += f"    平均召回率: {avg_recall:.4f} ({avg_recall*100:.2f}%)\n"
                if avg_recall_at_3 is not None:
                    result += f"    平均Recall@3: {avg_recall_at_3:.4f} ({avg_recall_at_3*100:.2f}%)\n"
                if avg_recall_at_5 is not None:
                    result += f"    平均Recall@5: {avg_recall_at_5:.4f} ({avg_recall_at_5*100:.2f}%)\n"
                if avg_recall_at_10 is not None:
                    result += f"    平均Recall@10: {avg_recall_at_10:.4f} ({avg_recall_at_10*100:.2f}%)\n"
                result += "\n"
        
        result += "=" * 80
        
        return result
    
    def get_all_metrics_summary(self) -> str:
        """
        Get a comprehensive summary of all evaluation metrics including overall metrics, metrics by theme, and metrics by type.
        """
        if self._df is None or len(self._df) == 0:
            return "错误: 评估数据为空。"
        
        result = ""
        
        # 总体指标
        overall = self._calculate_overall_metrics()
        if overall:
            result += "=" * 80 + "\n"
            result += "【总体评估指标】\n"
            result += "=" * 80 + "\n\n"
            result += f"总样本数: {overall.get('total_count', 0)}\n"
            result += f"平均准确率: {overall.get('average_accuracy', 0):.4f} ({overall.get('average_accuracy', 0)*100:.2f}%)\n"
            result += f"平均召回率: {overall.get('average_recall', 0):.4f} ({overall.get('average_recall', 0)*100:.2f}%)\n"
            if overall.get('average_recall_at_3') is not None:
                result += f"平均Recall@3: {overall.get('average_recall_at_3', 0):.4f} ({overall.get('average_recall_at_3', 0)*100:.2f}%)\n"
            if overall.get('average_recall_at_5') is not None:
                result += f"平均Recall@5: {overall.get('average_recall_at_5', 0):.4f} ({overall.get('average_recall_at_5', 0)*100:.2f}%)\n"
            if overall.get('average_recall_at_10') is not None:
                result += f"平均Recall@10: {overall.get('average_recall_at_10', 0):.4f} ({overall.get('average_recall_at_10', 0)*100:.2f}%)\n"
            result += "\n\n"
        
        # 按Theme分类
        theme_metrics = self._calculate_metrics_by_theme()
        if theme_metrics:
            result += "=" * 80 + "\n"
            result += "【按Theme分类的评估指标】\n"
            result += "=" * 80 + "\n\n"
            for theme_name, metrics in sorted(theme_metrics.items()):
                result += f"Theme: {theme_name}\n"
                result += f"  样本数: {metrics.get('count', 0)}\n"
                result += f"  平均准确率: {metrics.get('average_accuracy', 0):.4f} ({metrics.get('average_accuracy', 0)*100:.2f}%)\n"
                result += f"  平均召回率: {metrics.get('average_recall', 0):.4f} ({metrics.get('average_recall', 0)*100:.2f}%)\n"
                if metrics.get('average_recall_at_3') is not None:
                    result += f"  平均Recall@3: {metrics.get('average_recall_at_3', 0):.4f} ({metrics.get('average_recall_at_3', 0)*100:.2f}%)\n"
                result += "\n"
            result += "\n"
        
        # 按Type分类
        type_metrics = self._calculate_metrics_by_type()
        if type_metrics:
            result += "=" * 80 + "\n"
            result += "【按Type分类的评估指标】\n"
            result += "=" * 80 + "\n\n"
            for type_val, metrics in sorted(type_metrics.items()):
                result += f"Type: {type_val}\n"
                result += f"  样本数: {metrics.get('count', 0)}\n"
                result += f"  平均准确率: {metrics.get('average_accuracy', 0):.4f} ({metrics.get('average_accuracy', 0)*100:.2f}%)\n"
                result += f"  平均召回率: {metrics.get('average_recall', 0):.4f} ({metrics.get('average_recall', 0)*100:.2f}%)\n"
                if metrics.get('average_recall_at_3') is not None:
                    result += f"  平均Recall@3: {metrics.get('average_recall_at_3', 0):.4f} ({metrics.get('average_recall_at_3', 0)*100:.2f}%)\n"
                result += "\n"
            result += "\n"
        
        result += "=" * 80
        
        return result

