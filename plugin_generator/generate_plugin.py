"""
生成OpenWebUI插件
输入: evaluation_results.csv (固定)
输出: evaluation_plugin.py (固定，生成在plugin_generator目录)
"""

import pandas as pd
import json
from pathlib import Path

# 固定路径
CSV_PATH = "../output/evaluation_results.csv"
OUTPUT_PATH = Path(__file__).parent / "evaluation_plugin.py"
TEMPLATE_PATH = Path(__file__).parent / "tools_template.py"

# 读取CSV文件
print(f"读取CSV文件: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"成功读取: {len(df)} 条记录")

# 只保留必要的列（保留question，移除answer, reference, latency等冗余字段）
required_columns = [
    'question', 'accuracy', 'recall', 'theme', 'type',
    'recall@3', 'recall@5', 'recall@10',
    'top1_theme_match', 'top1_chapter_match', 'top1_both_match'
]
# 只保留存在的列
available_columns = [col for col in required_columns if col in df.columns]
df_minimal = df[available_columns].copy()

print(f"精简数据: 保留 {len(available_columns)} 列 (原 {len(df.columns)} 列)")

# 转换为Python字典格式（使用更紧凑的格式）
data_list = df_minimal.to_dict('records')
# 使用紧凑格式，减少缩进
data_code = f'''EVALUATION_DATA = {json.dumps(data_list, ensure_ascii=False, indent=1)}'''

# 将 JSON 的 true/false 转换为 Python 的 True/False
data_code = data_code.replace(': true', ': True')
data_code = data_code.replace(': false', ': False')
data_code = data_code.replace(', true', ', True')
data_code = data_code.replace(', false', ', False')

# 读取工具模板
with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    template_content = f.read()

# 替换占位符数据
placeholder = "EVALUATION_DATA = []  # 占位符，生成脚本会替换为实际数据"
if placeholder in template_content:
    final_content = template_content.replace(placeholder, data_code)
else:
    # 兼容旧版本：在注释后插入
    insert_marker = "# EVALUATION_DATA 将在这里插入"
    header_part = template_content[:template_content.find(insert_marker)]
    footer_part = template_content[template_content.find(insert_marker) + len(insert_marker):]
    final_content = header_part + data_code + '\n\n' + footer_part

# 写入输出文件
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(final_content)

print(f"生成完成: {OUTPUT_PATH}")
print(f"文件大小: {Path(OUTPUT_PATH).stat().st_size / 1024 / 1024:.2f} MB")

