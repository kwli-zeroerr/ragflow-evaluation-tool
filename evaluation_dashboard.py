"""
评测结果仪表盘生成器
生成美观的HTML仪表盘，包含可视化图表和分页表格
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)


class EvaluationDashboard:
    """评测结果仪表盘生成器"""
    
    # 指标名称中文映射
    METRIC_NAMES_CN = {
        'accuracy': '准确率',
        'recall': '召回率',
        'recall@3': '召回率@3',
        'recall@5': '召回率@5',
        'recall@10': '召回率@10',
        'latency': '响应时间(秒)',
        'latency_avg': '平均响应时间(秒)',
        'latency_total': '总响应时间(秒)',
        'question': '问题',
        'answer': '答案',
        'reference': '参考章节',
        'type': '类型',
        'theme': '主题',
        'retrieved_count': '检索数量',
        'correct_count': '正确数量',
        'top1_theme_match': 'Top1主题匹配',
        'top1_chapter_match': 'Top1章节匹配',
        'top1_both_match': 'Top1同时匹配'
    }
    
    def __init__(self, results_df: pd.DataFrame, latency_stats: Optional[dict] = None):
        """
        初始化仪表盘生成器
        
        Args:
            results_df: 评测结果DataFrame
            latency_stats: latency统计信息，包含latency_avg和latency_total
        """
        self.df = results_df.copy()
        self.latency_stats = latency_stats or {}
        # metric_columns用于指标统计，排除latency（latency单独统计）
        self.metric_columns = [
            col for col in self.df.columns 
            if '@' in col or col in ['accuracy', 'recall']
        ]
    
    def _translate_metric_name(self, metric_name: str) -> str:
        """
        翻译指标名称为中文
        
        Args:
            metric_name: 英文指标名称
            
        Returns:
            中文指标名称
        """
        # 如果直接匹配
        if metric_name in self.METRIC_NAMES_CN:
            return self.METRIC_NAMES_CN[metric_name]
        
        # 处理 recall@k 格式
        if metric_name.startswith('recall@'):
            k = metric_name.replace('recall@', '')
            return f'召回率@{k}'
        
        # 处理其他 @ 格式
        if '@' in metric_name:
            parts = metric_name.split('@')
            if len(parts) == 2:
                base_name = self._translate_metric_name(parts[0])
                return f'{base_name}@{parts[1]}'
        
        # 默认返回原名称
        return metric_name
    
    def _calculate_summary(self) -> dict:
        """计算总体指标（排除latency，latency单独统计）"""
        summary = {}
        for col in self.metric_columns:
            if col in self.df.columns:
                summary[col] = self.df[col].mean()
        
        # 添加latency统计信息（平均值和总和），不包含单个latency
        if 'latency_avg' in self.latency_stats:
            summary['latency_avg'] = self.latency_stats['latency_avg']
        if 'latency_total' in self.latency_stats:
            summary['latency_total'] = self.latency_stats['latency_total']
        
        return summary
    
    def _calculate_type_stats(self) -> pd.DataFrame:
        """按类型统计（排除latency）"""
        if 'type' in self.df.columns and self.df['type'].notna().any():
            # 只统计metric_columns，排除latency
            available_columns = [col for col in self.metric_columns if col in self.df.columns]
            if available_columns:
                return self.df.groupby('type')[available_columns].mean()
        return None
    
    def _calculate_theme_stats(self) -> pd.DataFrame:
        """按主题统计（排除latency）"""
        if 'theme' in self.df.columns and self.df['theme'].notna().any():
            # 只统计metric_columns，排除latency
            available_columns = [col for col in self.metric_columns if col in self.df.columns]
            if available_columns:
                return self.df.groupby('theme')[available_columns].mean()
        return None
    
    def _calculate_latency_stats_by_type(self) -> Optional[pd.DataFrame]:
        """按类型统计latency"""
        if 'type' in self.df.columns and 'latency' in self.df.columns and self.df['type'].notna().any():
            latency_stats = self.df.groupby('type')['latency'].agg(['mean', 'min', 'max', 'count'])
            latency_stats.columns = ['平均响应时间(秒)', '最小响应时间(秒)', '最大响应时间(秒)', '测试数量']
            latency_stats['平均响应时间(秒)'] = latency_stats['平均响应时间(秒)'].round(3)
            latency_stats['最小响应时间(秒)'] = latency_stats['最小响应时间(秒)'].round(3)
            latency_stats['最大响应时间(秒)'] = latency_stats['最大响应时间(秒)'].round(3)
            return latency_stats
        return None
    
    def _calculate_latency_stats_by_theme(self) -> Optional[pd.DataFrame]:
        """按主题统计latency"""
        if 'theme' in self.df.columns and 'latency' in self.df.columns and self.df['theme'].notna().any():
            latency_stats = self.df.groupby('theme')['latency'].agg(['mean', 'min', 'max', 'count'])
            latency_stats.columns = ['平均响应时间(秒)', '最小响应时间(秒)', '最大响应时间(秒)', '测试数量']
            latency_stats['平均响应时间(秒)'] = latency_stats['平均响应时间(秒)'].round(3)
            latency_stats['最小响应时间(秒)'] = latency_stats['最小响应时间(秒)'].round(3)
            latency_stats['最大响应时间(秒)'] = latency_stats['最大响应时间(秒)'].round(3)
            return latency_stats
        return None
    
    def _generate_metrics_chart_data(self, summary: dict) -> dict:
        """生成指标图表数据"""
        # 排除latency、latency_avg和latency_total，因为它们有特殊的显示格式
        chart_summary = {k: v for k, v in summary.items() if k not in ['latency', 'latency_avg', 'latency_total']}
        labels = [self._translate_metric_name(k) for k in chart_summary.keys()]
        values = [round(v, 4) if isinstance(v, (int, float)) else v for v in chart_summary.values()]
        return {
            'labels': labels,
            'values': values,
            'original_keys': list(chart_summary.keys())  # 保留原始键名用于数据关联
        }
    
    def _generate_type_chart_data(self, type_stats: pd.DataFrame) -> dict:
        """生成按类型统计的图表数据"""
        if type_stats is None:
            return None
        
        types = type_stats.index.tolist()
        metrics = type_stats.columns.tolist()
        
        datasets = []
        for metric in metrics:
            datasets.append({
                'label': self._translate_metric_name(metric),
                'data': [round(v, 4) for v in type_stats[metric].values]
            })
        
        return {
            'labels': types,
            'datasets': datasets
        }
    
    def _generate_theme_chart_data(self, theme_stats: pd.DataFrame) -> dict:
        """生成按主题统计的图表数据"""
        if theme_stats is None:
            return None
        
        themes = theme_stats.index.tolist()
        metrics = theme_stats.columns.tolist()
        
        datasets = []
        for metric in metrics:
            datasets.append({
                'label': self._translate_metric_name(metric),
                'data': [round(v, 4) for v in theme_stats[metric].values]
            })
        
        return {
            'labels': themes,
            'datasets': datasets
        }
    
    def _prepare_detail_table_data(self) -> tuple:
        """准备详细结果表格数据
        
        Returns:
            (table_data, original_columns, translated_columns): 
            table_data: 表格数据列表
            original_columns: 原始列名列表
            translated_columns: 中文列名列表
        """
        # 选择要显示的列
        display_columns = [
            'question', 'answer', 'reference', 'type', 'theme',
            'accuracy', 'recall', 'recall@3', 'recall@5', 'recall@10', 'latency'
        ]
        
        # 只保留存在的列
        available_columns = [col for col in display_columns if col in self.df.columns]
        
        # 生成中文列名
        translated_columns = [self._translate_metric_name(col) for col in available_columns]
        
        # 准备数据（使用原始列名作为键）
        table_data = []
        for idx, row in self.df.iterrows():
            record = {}
            for col in available_columns:
                value = row[col]
                # 格式化数值
                if isinstance(value, (int, float)):
                    if col == 'latency':
                        record[col] = f"{value:.3f}s"
                    else:
                        record[col] = f"{value:.4f}" if value != int(value) else str(int(value))
                else:
                    record[col] = str(value) if pd.notna(value) else ""
            table_data.append(record)
        
        return table_data, available_columns, translated_columns
    
    def generate(self, output_path: str = "evaluation_dashboard.html") -> dict:
        """
        生成仪表盘HTML文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            总体指标字典
        """
        # 计算统计数据
        summary = self._calculate_summary()
        type_stats = self._calculate_type_stats()
        theme_stats = self._calculate_theme_stats()
        latency_stats_by_type = self._calculate_latency_stats_by_type()
        latency_stats_by_theme = self._calculate_latency_stats_by_theme()
        
        # 生成图表数据
        metrics_chart_data = self._generate_metrics_chart_data(summary)
        type_chart_data = self._generate_type_chart_data(type_stats)
        theme_chart_data = self._generate_theme_chart_data(theme_stats)
        
        # 准备详细表格数据
        table_data, original_columns, translated_columns = self._prepare_detail_table_data()
        
        # 生成HTML
        html_content = self._generate_html(
            summary=summary,
            metrics_chart_data=metrics_chart_data,
            type_stats=type_stats,
            type_chart_data=type_chart_data,
            theme_stats=theme_stats,
            theme_chart_data=theme_chart_data,
            latency_stats_by_type=latency_stats_by_type,
            latency_stats_by_theme=latency_stats_by_theme,
            table_data=table_data,
            original_columns=original_columns,
            translated_columns=translated_columns
        )
        
        # 保存文件
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"{'='*80}")
        logger.info(f"仪表盘已生成: {output_file.absolute()}")
        return summary
    
    def _generate_html(self, summary: dict, metrics_chart_data: dict,
                      type_stats: pd.DataFrame, type_chart_data: dict,
                      theme_stats: pd.DataFrame, theme_chart_data: dict,
                      latency_stats_by_type: Optional[pd.DataFrame],
                      latency_stats_by_theme: Optional[pd.DataFrame],
                      table_data: list, original_columns: list, translated_columns: list) -> str:
        """生成HTML内容"""
        
        # 创建summary的中文版本（用于显示）
        summary_cn = {self._translate_metric_name(k): v for k, v in summary.items()}
        
        # 将数据转换为JSON字符串（用于JavaScript）
        # JSON字符串在JavaScript中是安全的，可以直接使用
        metrics_chart_json = json.dumps(metrics_chart_data, ensure_ascii=False)
        type_chart_json = json.dumps(type_chart_data, ensure_ascii=False) if type_chart_data else "null"
        theme_chart_json = json.dumps(theme_chart_data, ensure_ascii=False) if theme_chart_data else "null"
        table_data_json = json.dumps(table_data, ensure_ascii=False)
        # 表格列名使用中文
        table_columns_json = json.dumps(translated_columns, ensure_ascii=False)
        # 同时传递原始列名用于数据访问
        original_columns_json = json.dumps(original_columns, ensure_ascii=False)
        # summary使用中文版本
        summary_json = json.dumps(summary_cn, ensure_ascii=False)
        
        # 准备latency统计表格的HTML
        latency_type_table_html = ""
        if latency_stats_by_type is not None:
            latency_type_table_html = latency_stats_by_type.to_html(
                classes='table table-striped table-hover stats-table', 
                table_id='latencyTypeStatsTable'
            )
        
        latency_theme_table_html = ""
        if latency_stats_by_theme is not None:
            latency_theme_table_html = latency_stats_by_theme.to_html(
                classes='table table-striped table-hover stats-table', 
                table_id='latencyThemeStatsTable'
            )
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RagFlow 检索评测仪表盘</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- DataTables CSS -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    
    <style>
        body {{
            background-color: #f5f7fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .dashboard-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-card {{
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .section-title {{
            color: #495057;
            font-weight: 600;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
        }}
        .table-container {{
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stats-table {{
            margin-top: 1rem;
        }}
        .stats-table th {{
            background-color: #667eea;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <div class="container">
            <h1 class="mb-2">📊 RagFlow 检索评测仪表盘</h1>
            <p class="mb-0">评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 测试用例数: {len(self.df)}</p>
        </div>
    </div>
    
    <div class="container">
        <!-- 总体指标卡片 -->
        <div class="row mb-4">
            <h2 class="section-title">总体指标</h2>
            <div class="row" id="metrics-cards">
                <!-- 指标卡片将通过JavaScript动态生成 -->
            </div>
        </div>
        
        <!-- 指标图表 -->
        <div class="chart-container">
            <h3 class="section-title">指标概览</h3>
            <canvas id="metricsChart" height="80"></canvas>
        </div>
        
        <!-- 按类型统计 -->
        {f'''
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="chart-container">
                    <h3 class="section-title">按类型统计</h3>
                    <canvas id="typeChart"></canvas>
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <h3 class="section-title">按类型统计表格</h3>
                    <div class="table-responsive">
                        {type_stats.rename(columns={col: self._translate_metric_name(col) for col in type_stats.columns}).to_html(classes='table table-striped table-hover stats-table', table_id='typeStatsTable')}
                    </div>
                </div>
            </div>
        </div>
        ''' if type_stats is not None else ''}
        
        <!-- 按主题统计 -->
        {f'''
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="chart-container">
                    <h3 class="section-title">按主题统计</h3>
                    <canvas id="themeChart"></canvas>
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <h3 class="section-title">按主题统计表格</h3>
                    <div class="table-responsive">
                        {theme_stats.rename(columns={col: self._translate_metric_name(col) for col in theme_stats.columns}).to_html(classes='table table-striped table-hover stats-table', table_id='themeStatsTable')}
                    </div>
                </div>
            </div>
        </div>
        ''' if theme_stats is not None else ''}
        
        <!-- Latency统计表格 -->
        {f'''
        <div class="row mb-4">
            {f'''
            <div class="col-md-6">
                <div class="table-container">
                    <h3 class="section-title">按类型响应时间统计</h3>
                    <div class="table-responsive">
                        {latency_type_table_html}
                    </div>
                </div>
            </div>
            ''' if latency_stats_by_type is not None else ''}
            {f'''
            <div class="col-md-6">
                <div class="table-container">
                    <h3 class="section-title">按主题响应时间统计</h3>
                    <div class="table-responsive">
                        {latency_theme_table_html}
                    </div>
                </div>
            </div>
            ''' if latency_stats_by_theme is not None else ''}
        </div>
        ''' if (latency_stats_by_type is not None or latency_stats_by_theme is not None) else ''}
        
        <!-- 详细结果表格 -->
        <div class="table-container">
            <h3 class="section-title">详细结果</h3>
            <div class="table-responsive">
                <table id="detailTable" class="table table-striped table-hover" style="width:100%">
                    <thead>
                        <tr id="table-header">
                            <!-- 表头将通过JavaScript动态生成 -->
                        </tr>
                    </thead>
                    <tbody id="table-body">
                        <!-- 表格内容将通过JavaScript动态生成 -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <!-- jQuery -->
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <!-- DataTables JS -->
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
    
    <script>
        // 数据
        const metricsChartData = {metrics_chart_json};
        const typeChartData = {type_chart_json};
        const themeChartData = {theme_chart_json};
        const tableData = {table_data_json};
        const tableColumns = {table_columns_json};  // 中文列名（用于显示）
        const originalColumns = {original_columns_json};  // 原始列名（用于数据访问）
        const summary = {summary_json};
        
        // 生成指标卡片
        function generateMetricCards() {{
            const container = document.getElementById('metrics-cards');
            const metrics = Object.keys(summary);
            const values = Object.values(summary);
            
            metrics.forEach((metric, index) => {{
                const col = document.createElement('div');
                col.className = 'col-md-3 col-sm-6 mb-3';
                
                const card = document.createElement('div');
                card.className = 'metric-card';
                
                const value = document.createElement('div');
                value.className = 'metric-value';
                
                // 格式化显示值
                let displayValue = values[index];
                if (typeof displayValue === 'number') {{
                    // 对于latency相关的指标，显示为秒，保留3位小数
                    if (metric.includes('latency')) {{
                        displayValue = displayValue.toFixed(3) + 's';
                    }} else {{
                        // 其他指标保留4位小数
                        displayValue = displayValue.toFixed(4);
                    }}
                }} else {{
                    displayValue = displayValue;
                }}
                value.textContent = displayValue;
                
                const label = document.createElement('div');
                label.className = 'metric-label';
                label.textContent = metric;
                
                card.appendChild(value);
                card.appendChild(label);
                col.appendChild(card);
                container.appendChild(col);
            }});
        }}
        
        // 生成指标图表
        function generateMetricsChart() {{
            const ctx = document.getElementById('metricsChart').getContext('2d');
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: metricsChartData.labels,
                    datasets: [{{
                        label: '指标值',
                        data: metricsChartData.values,
                        backgroundColor: 'rgba(102, 126, 234, 0.6)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 1.0
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }}
                }}
            }});
        }}
        
        // 生成类型统计图表
        function generateTypeChart() {{
            if (!typeChartData) return;
            
            const ctx = document.getElementById('typeChart').getContext('2d');
            const colors = [
                'rgba(102, 126, 234, 0.6)',
                'rgba(118, 75, 162, 0.6)',
                'rgba(237, 100, 166, 0.6)',
                'rgba(255, 154, 0, 0.6)',
                'rgba(52, 211, 153, 0.6)'
            ];
            
            const datasets = typeChartData.datasets.map((dataset, index) => ({{
                ...dataset,
                backgroundColor: colors[index % colors.length],
                borderColor: colors[index % colors.length].replace('0.6', '1'),
                borderWidth: 2
            }}));
            
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: typeChartData.labels,
                    datasets: datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 1.0
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            position: 'top'
                        }}
                    }}
                }}
            }});
        }}
        
        // 生成主题统计图表
        function generateThemeChart() {{
            if (!themeChartData) return;
            
            const ctx = document.getElementById('themeChart').getContext('2d');
            const colors = [
                'rgba(102, 126, 234, 0.6)',
                'rgba(118, 75, 162, 0.6)',
                'rgba(237, 100, 166, 0.6)',
                'rgba(255, 154, 0, 0.6)',
                'rgba(52, 211, 153, 0.6)'
            ];
            
            const datasets = themeChartData.datasets.map((dataset, index) => ({{
                ...dataset,
                backgroundColor: colors[index % colors.length],
                borderColor: colors[index % colors.length].replace('0.6', '1'),
                borderWidth: 2
            }}));
            
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: themeChartData.labels,
                    datasets: datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 1.0
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            position: 'top'
                        }}
                    }}
                }}
            }});
        }}
        
        // 生成详细结果表格
        function generateDetailTable() {{
            // 生成表头（使用中文列名）
            const header = document.getElementById('table-header');
            tableColumns.forEach(col => {{
                const th = document.createElement('th');
                th.textContent = col;
                header.appendChild(th);
            }});
            
            // 生成表格内容（使用原始列名访问数据）
            const tbody = document.getElementById('table-body');
            tableData.forEach(row => {{
                const tr = document.createElement('tr');
                originalColumns.forEach((origCol, index) => {{
                    const td = document.createElement('td');
                    td.textContent = row[origCol] || '';
                    tr.appendChild(td);
                }});
                tbody.appendChild(tr);
            }});
            
            // 初始化DataTables
            $('#detailTable').DataTable({{
                language: {{
                    "sProcessing": "处理中...",
                    "sLengthMenu": "显示 _MENU_ 项结果",
                    "sZeroRecords": "没有匹配结果",
                    "sInfo": "显示第 _START_ 至 _END_ 项结果，共 _TOTAL_ 项",
                    "sInfoEmpty": "显示第 0 至 0 项结果，共 0 项",
                    "sInfoFiltered": "(由 _MAX_ 项结果过滤)",
                    "sInfoPostFix": "",
                    "sSearch": "搜索:",
                    "sUrl": "",
                    "sEmptyTable": "表中数据为空",
                    "sLoadingRecords": "载入中...",
                    "sInfoThousands": ",",
                    "oPaginate": {{
                        "sFirst": "首页",
                        "sPrevious": "上页",
                        "sNext": "下页",
                        "sLast": "末页"
                    }},
                    "oAria": {{
                        "sSortAscending": ": 以升序排列此列",
                        "sSortDescending": ": 以降序排列此列"
                    }}
                }},
                pageLength: 10,
                lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "全部"]],
                order: [[0, 'asc']],
                scrollX: true
            }});
        }}
        
        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', function() {{
            generateMetricCards();
            generateMetricsChart();
            generateTypeChart();
            generateThemeChart();
            generateDetailTable();
        }});
    </script>
</body>
</html>"""
        
        return html

