# RagFlow 检索系统自动化评测工具

一个功能完善的 RAG（检索增强生成）系统评测工具，用于自动化测试和评估 RagFlow 检索方案的效果。支持批量测试、多维度指标计算、可视化报告生成和 A/B 测试对比。

## ✨ 核心特性

- 🔄 **批量自动化评测** - 支持大规模测试用例的批量处理
- ⚡ **并发执行支持** - 支持多线程并发执行，大幅提升评测速度
- 📊 **多维度指标计算** - 准确率、召回率、Recall@K、响应时间等
- 🎯 **智能章节匹配** - 支持中英文章节格式，智能判断匹配关系
- 📈 **可视化报告** - 自动生成交互式 HTML 仪表盘
- 🧪 **A/B 测试对比** - 对比不同配置方案的效果差异
- 🔍 **异常检测** - 自动识别并记录异常测试结果
- 📝 **完整日志记录** - 详细的运行日志和 API 响应备份

## 📋 目录结构

```
automate_testing/
├── README.md                      # 项目说明文档
├── evaluation_tool.py            # 主程序（核心评测逻辑）
├── evaluation_dashboard.py        # 可视化报告生成器
├── evaluation_metrics_tools.py   # 评估指标分析工具
├── audit_theme.py                # 主题审核工具
│
├── input/                         # 输入数据目录
│   └── test.xlsx                 # 测试数据文件（Excel格式）
│
└── output/                        # 输出结果目录（自动生成）
    ├── logs/                     # 运行日志
    ├── api_responses/            # API响应备份
    ├── datasets.json             # 数据集配置
    ├── evaluation_results.csv    # 评测结果表格
    ├── evaluation_dashboard.html # 可视化报告
    └── anomalies.json            # 异常情况记录
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- 依赖包：`pandas`, `numpy`, `requests`, `openpyxl`, `matplotlib`, `seaborn`

### 安装步骤

1. **安装依赖**

```bash
pip install pandas numpy requests openpyxl matplotlib seaborn
```

2. **准备测试数据**

在 `input/` 目录下创建 `test.xlsx` 文件，包含以下列：

| 列名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | 文本 | ✅ | 测试问题 |
| `answer` | 文本 | ✅ | 标准答案 |
| `reference` | 文本 | ✅ | 标注的章节信息（如 "1.1"、"第一章"） |
| `type` | 文本 | ❌ | 问题类型（可选） |
| `theme` | 文本 | ❌ | 手册主题名称（可选） |

3. **配置 API 信息**

编辑 `evaluation_tool.py`，找到 `RagFlowClient` 初始化部分，修改配置文件 `config.py` 中的 API 配置：

```python
client = RagFlowClient(
    api_url="http://your-api-url:port/",  # 修改为你的 API 地址
    api_key="your-api-key"                 # 修改为你的 API 密钥
)
```

4. **运行评测**

```bash
python evaluation_tool.py
```

程序将自动：
- 读取测试数据
- 批量调用 API 进行检索
- 计算各项评测指标
- 生成可视化报告和结果文件

## ⚙️ 配置说明

### 检索参数配置

在 `evaluation_tool.py` 中的 `RetrievalConfig` 部分可调整检索参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_ids` | List[str] | - | 数据集ID列表（必需） |
| `top_k` | int | 5 | 返回最相似的K个结果 |
| `similarity_threshold` | float | 0.0 | 相似度阈值，低于此值的结果会被过滤 |
| `vector_similarity_weight` | float | None | 向量相似度权重（0-1之间） |
| `keyword` | bool | True | 是否启用关键词检索 |
| `rerank_id` | str | None | 重排序模型ID（如有） |
| `highlight` | bool | True | 是否启用高亮 |

### 输出目录配置

如需修改输入/输出目录，编辑 `evaluation_tool.py` 文件开头（约第 20-24 行）：

```python
input_dir = Path("your_input_dir")
output_dir = Path("your_output_dir")
```

### 并发执行配置

工具支持并发执行测试用例，可以大幅提升评测速度。在 `config.py` 中配置：

```python
# 并发线程数，设置为1表示顺序执行（单线程），大于1表示并发执行
MAX_WORKERS = 5  # 建议值：3-10，根据API服务器能力调整

# 每个请求之间的延迟（秒），用于避免API限流
DELAY_BETWEEN_REQUESTS = 0.5  # 建议值：0.1-1.0 秒
```

**使用建议**：
- **顺序执行**（`MAX_WORKERS = 1`）：适合API有严格限流或测试用例较少的情况
- **并发执行**（`MAX_WORKERS > 1`）：适合大规模测试，可显著提升速度
  - 建议从较小的并发数开始（如 3-5），逐步增加
  - 注意观察API服务器的响应和限流情况
  - 如果遇到限流错误，可以增加 `DELAY_BETWEEN_REQUESTS` 或减少 `MAX_WORKERS`

**性能提升示例**：
- 100个测试用例，顺序执行：约 100-200 秒（取决于API响应时间）
- 100个测试用例，5并发：约 20-40 秒（约5倍提升）

### A/B 测试开关配置

通过配置变量控制是否启用 A/B 测试：

```python
# 是否启用A/B测试
ENABLE_AB_TEST = False  # True: 启用A/B测试, False: 单次评测

# 单次评测配置（当ENABLE_AB_TEST=False时使用）
RETRIEVAL_CONFIG = {
    "top_k": 1024,
    "similarity_threshold": 0.2,
    "vector_similarity_weight": 0.3,
    "rerank_id": "",
    "highlight": False,
    "page": 1,
    "page_size": 30
}
```

**使用场景**：
- **单次评测**（`ENABLE_AB_TEST = False`）：适合日常评测，使用单一配置
- **A/B 测试**（`ENABLE_AB_TEST = True`）：适合对比不同配置方案的效果

## 📊 评测指标说明

### 准确率 (Accuracy)

- **定义**：检索结果中正确匹配的数量占总检索结果的比例
- **计算**：`accuracy = correct_count / total_retrieved`
- **意义**：评估检索结果的精确度

### 召回率 (Recall)

- **定义**：在返回的前 K 个结果中，是否至少有一个匹配标注章节
- **计算**：有匹配 = 1.0，无匹配 = 0.0
- **意义**：评估检索系统是否能找到正确答案

### Recall@K

- **定义**：前 K 个结果中是否至少有一个正确匹配
- **示例**：Recall@3 表示前 3 个结果中是否有正确答案
- **用途**：评估不同 K 值下的召回表现

### 章节匹配规则

工具支持智能章节匹配，规则如下：

- ✅ **正确匹配**：
  - 章节完全匹配（如检索到 "1.1"，标注是 "1.1"）
  - 检索到"大章"，标注是"小章"（如检索到 "1"，标注是 "1.1"）
  
- ❌ **错误匹配**：
  - 检索到"小章"，标注是"大章"（如检索到 "1.1.1"，标注是 "1.1"）
  - 章节完全不匹配

- 📝 **支持格式**：
  - 阿拉伯数字：`1`, `1.1`, `1.1.1`
  - 中文数字：`第一章`, `第一節`, `二十七`
  - 混合格式：`第1章`, `1.1节`

## 📈 输出文件说明

### 1. `evaluation_results.csv`

详细的评测结果表格，包含：
- 每个测试用例的问题、答案、标注章节
- 准确率、召回率、Recall@K 等指标
- 主题匹配和章节匹配情况
- 响应时间统计

### 2. `evaluation_dashboard.html`

交互式可视化报告，包含：
- 📊 总体指标统计（准确率、召回率、响应时间等）
- 📈 按类型/主题分类的指标对比
- 📋 详细的测试结果表格（支持分页和搜索）
- 🎨 美观的图表展示

在浏览器中打开即可查看完整报告。

### 3. `logs/evaluation_tool_*.log`

运行日志文件，记录：
- 每个测试用例的执行情况
- API 调用详情和响应状态
- 错误信息和异常情况
- 性能统计

### 4. `api_responses/test_*.json`

每个测试用例的完整 API 响应备份，用于：
- 调试和问题排查
- 详细分析检索结果
- 复现和验证问题

### 5. `anomalies.json`

异常情况记录，自动识别：
- 准确率为 0 但召回率 > 0 的情况
- 其他异常模式

## 🧪 高级功能

### A/B 测试对比

#### 启用 A/B 测试

在 `config.py` 中设置：

```python
ENABLE_AB_TEST = True  # 启用 A/B 测试
```

程序会自动运行两种配置方案的对比测试，并生成对比报告。

#### 单次评测模式

在 `config.py` 中设置：

```python
ENABLE_AB_TEST = False  # 关闭 A/B 测试，使用单次评测

# 配置检索参数
RETRIEVAL_CONFIG = {
    "top_k": 1024,
    "similarity_threshold": 0.2,
    "vector_similarity_weight": 0.3,
    # ... 其他参数
}
```

#### 编程方式使用 A/B 测试

如果需要自定义配置，也可以通过代码方式：

```python
from evaluation_tool import RagFlowClient, RetrievalConfig, ABTestComparator

# 创建客户端
client = RagFlowClient(api_url="...", api_key="...")

# 方案A：不使用关键词检索
config_a = RetrievalConfig(
    dataset_ids=["dataset_id"],
    top_k=5,
    keyword=False
)

# 方案B：使用关键词检索
config_b = RetrievalConfig(
    dataset_ids=["dataset_id"],
    top_k=5,
    keyword=True
)

# 运行A/B测试（支持并发）
comparator = ABTestComparator(client, client)
comparison = comparator.run_ab_test(
    test_cases, config_a, config_b,
    max_workers=5,  # 并发线程数
    delay_between_requests=0.5  # 请求延迟
)

# 查看对比结果
print(comparison)
```

会生成对比图表 `output/ab_comparison.png`，直观展示两种方案的指标差异。

### 限制测试数据量

如需快速测试，可限制测试数据量。在 `evaluation_tool.py` 中找到相应位置，取消注释：

```python
# 限制测试数据为前10条
if len(test_cases) > 10:
    test_cases = test_cases[:10]
    logger.info(f"测试数据已限制为前10条")
```

## 🏗️ 代码架构

### 主要类说明

1. **`RagFlowClient`** - RagFlow API 客户端
   - `search()`: 调用检索 API
   - `get_all_datasets_and_documents()`: 获取数据集和文档列表

2. **`TestCase`** - 测试用例数据结构
   - 包含问题、答案、标注章节等信息

3. **`RetrievalConfig`** - 检索配置
   - 封装所有检索参数

4. **`ChapterMatcher`** - 章节匹配器
   - `is_valid_match()`: 判断章节是否匹配
   - `normalize_chapter()`: 标准化章节格式
   - 支持中英文章节格式转换

5. **`MetricsCalculator`** - 指标计算器
   - `calculate_accuracy()`: 计算准确率和召回率
   - `recall_at_k()`: 计算 Recall@K

6. **`EvaluationRunner`** - 评测运行器
   - `run_single_test()`: 运行单个测试
   - `run_batch_evaluation()`: 批量评测
   - `generate_report()`: 生成报告

7. **`ABTestComparator`** - A/B 测试对比器
   - `run_ab_test()`: 运行 A/B 测试
   - `generate_comparison()`: 生成对比结果

8. **`EvaluationDashboard`** - 可视化报告生成器
   - `generate()`: 生成 HTML 仪表盘

## ❓ 常见问题

### Q1: 提示 "测试数据文件不存在"

**解决方案**：
- 确保 `input/test.xlsx` 文件存在
- 检查文件名是否正确（区分大小写）
- 确认文件路径配置正确

### Q2: 提示 "Excel文件缺少必需的列"

**解决方案**：
- 检查 Excel 文件是否包含 `question`、`answer`、`reference` 三列
- 确认列名拼写正确（区分大小写）
- 检查 Excel 文件格式是否正确

### Q3: API 调用失败

**解决方案**：
1. 检查 API 地址和密钥是否正确
2. 确认网络连接正常
3. 查看 `output/logs/` 目录下的日志文件获取详细错误信息
4. 检查 API 服务是否正常运行

### Q4: 章节匹配结果不符合预期

**解决方案**：
- 检查标注的章节格式是否正确
- 确认章节匹配规则（见"章节匹配规则"部分）
- 查看 `api_responses/` 中的 API 响应，检查实际返回的章节信息

### Q5: 如何自定义评测指标？

**解决方案**：
- 修改 `MetricsCalculator` 类添加新的指标计算方法
- 在 `EvaluationRunner.run_single_test()` 中调用新的计算方法
- 更新 `EvaluationDashboard` 以展示新指标

### Q6: 并发执行时遇到 API 限流错误

**解决方案**：
1. 减少并发线程数：将 `MAX_WORKERS` 从 5 减少到 3 或更小
2. 增加请求延迟：将 `DELAY_BETWEEN_REQUESTS` 从 0.5 增加到 1.0 或更高
3. 检查 API 服务器的限流策略，根据实际情况调整参数

### Q7: 如何切换单次评测和 A/B 测试模式？

**解决方案**：
- 在 `config.py` 中修改 `ENABLE_AB_TEST` 变量：
  - `False`：单次评测模式，使用 `RETRIEVAL_CONFIG` 配置
  - `True`：A/B 测试模式，对比两种配置方案
- 修改后直接运行程序即可，无需修改代码

## 📝 使用示例

### 示例 1：基本评测流程

```bash
# 1. 准备测试数据：input/test.xlsx
# 2. 配置 API：复制 config.py.example 为 config.py 并填入 API 配置
# 3. 运行评测：python evaluation_tool.py
# 4. 查看结果：打开 output/evaluation_dashboard.html
```

### 示例 2：调整检索参数

```python
# 在 evaluation_tool.py 中修改 RetrievalConfig
config = RetrievalConfig(
    dataset_ids=["your_dataset_id"],
    top_k=10,                    # 返回10个结果
    similarity_threshold=0.3,    # 提高相似度阈值
    keyword=False,               # 关闭关键词检索
    vector_similarity_weight=0.7 # 提高向量相似度权重
)
```

## 🔧 开发与贡献

### 项目结构

- `evaluation_tool.py` - 核心评测逻辑
- `evaluation_dashboard.py` - 可视化报告生成
- `evaluation_metrics_tools.py` - 指标分析工具
- `audit_theme.py` - 主题审核工具

### 扩展开发

如需扩展功能，建议：
1. 遵循现有的代码结构和命名规范
2. 添加适当的日志记录
3. 更新本 README 文档
4. 确保向后兼容性

## 📄 许可证

本项目仅供内部使用。

## 📞 技术支持

如遇问题，请：
1. 查看 `output/logs/` 目录下的日志文件
2. 检查 `output/api_responses/` 中的 API 响应
3. 确认测试数据格式是否正确
4. 参考本文档的"常见问题"部分

---

**祝使用愉快！** 🎉
