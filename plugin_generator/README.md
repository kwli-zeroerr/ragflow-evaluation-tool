# OpenWebUI插件生成器

一键生成评估指标分析插件，用于OpenWebUI平台。

## 功能

- 从CSV文件读取评估结果数据
- 自动将数据嵌入到插件代码中
- 生成完整的、可直接使用的OpenWebUI插件文件

## 使用方法

### 方法一：直接运行（推荐）

```bash
cd plugin_generator
python generate_plugin.py <输入CSV文件路径> [输出文件路径]
```

**示例：**

```bash
# 使用默认路径（当前目录的evaluation_results.csv）
python generate_plugin.py

# 指定CSV文件路径
python generate_plugin.py ../output/evaluation_results.csv

# 指定输入和输出路径
python generate_plugin.py ../output/evaluation_results.csv evaluation_plugin.py
```

### 方法二：在Python中使用

```python
from generate_plugin import generate_plugin

generate_plugin(
    csv_path="evaluation_results.csv",
    output_path="evaluation_plugin.py"
)
```

## 文件说明

- **generate_plugin.py** - 一键生成脚本（主程序）
- **tools_template.py** - 工具类模板（包含所有工具方法）
- **README.md** - 本说明文件

## 输入要求

CSV文件必须包含以下列：
- `accuracy` - 准确率
- `recall` - 召回率
- `theme` - 主题分类
- `type` - 类型分类
- 其他可选列：`recall@3`, `recall@5`, `recall@10`, `top1_theme_match`, `top1_chapter_match`, `top1_both_match`, `latency` 等

## 输出说明

生成的 `evaluation_plugin.py` 文件包含：
- 所有评估数据（嵌入在代码中）
- 完整的工具类（Tools类）
- 5个工具方法：
  - `get_overall_metrics()` - 总体指标
  - `get_metrics_by_theme()` - 按Theme分类
  - `get_metrics_by_type()` - 按Type分类
  - `get_metrics_by_theme_and_type()` - 按Theme和Type组合
  - `get_all_metrics_summary()` - 完整摘要

## 部署步骤

1. 运行生成脚本，生成 `evaluation_plugin.py`
2. 将生成的插件文件上传到OpenWebUI插件目录
3. 确保服务器上安装了 `pandas` 库
4. 重启OpenWebUI服务（如果需要）

## 注意事项

- 生成的插件文件大小取决于数据量（约0.65MB/1000条记录）
- 数据已直接嵌入代码中，无需额外文件
- 更新数据时，重新运行生成脚本即可

## 故障排查

### 问题：找不到CSV文件
**解决方案**：检查CSV文件路径是否正确，使用绝对路径或相对路径

### 问题：生成的插件文件无法导入
**解决方案**：
- 检查文件是否完整生成
- 确认数据格式正确
- 查看控制台错误信息

### 问题：缺少必需的列
**解决方案**：确保CSV文件包含 `accuracy`, `recall`, `theme`, `type` 列

