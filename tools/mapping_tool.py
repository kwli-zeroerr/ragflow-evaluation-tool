"""

章节映射规则维护工具

提供交互式菜单界面，用于分析失败案例、生成映射建议、验证规则等

"""

import json
import csv
import re
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


# 添加项目根目录到路径

project_root = Path(__file__).parent.parent

sys.path.insert(0, str(project_root))

from core.rules_manager import RuleManager
from core.chapter_identifier_extractor import ChapterIdentifierExtractor
from core.chapter_identifier_matcher import ChapterIdentifierMatcher


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def try_read_csv(file_path: Path, encodings: List[str] = None) -> Optional[List[Dict[str, Any]]]:


    """尝试使用多种编码读取CSV文件"""

    if encodings is None:

        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb232', 'gb8030', 'latin']

    for encoding in encodings:

        try:

            with open(file_path, 'r', encoding=encoding) as f:

                reader = csv.DictReader(f)

                return list(reader)

        except (UnicodeDecodeError, UnicodeError):

            continue

        except Exception as e:

            logger.debug(f"使用编码 {encoding} 读取失败: {str(e)}")

            continue

    logger.error(f"无法使用任何编码读取文件: {file_path}")

    return None

class MappingTool:


    """章节映射规则维护工具"""

    def __init__(self):


        """初始化，自动检测路径"""

        self.project_root = project_root

        self.results_file = self.auto_detect_results_file()

        self.api_dir = self.auto_detect_api_dir()

        self.mapping_file = self.project_root / "rules" / "chapter_identifier_mapping.json"

        self.rule_manager = RuleManager()

        self.identifier_mappings = self.rule_manager.get_identifier_mapping()

        self.identifier_matcher = ChapterIdentifierMatcher(self.identifier_mappings)

        # 记录最近的应用历史，用于撤回
        self.recent_applied_changes: List[Dict[str, Any]] = []

    def auto_detect_results_file(self) -> Optional[Path]:


        """自动检测最新的测试结果文件"""

        candidates = [

            self.project_root / "output" / "single" / "evaluation_results.csv",

            self.project_root / "output" / "evaluation_results.csv",

        ]

        # 优先使用固定路径

        for candidate in candidates:

            if candidate.exists():

                return candidate

        # 如果固定路径不存在，查找到最新的CSV文件

        output_dir = self.project_root / "output"

        if output_dir.exists():

            csv_files = list(output_dir.rglob("evaluation_results.csv"))

            if csv_files:

                # 按修改时间排序，返回最新的

                csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

                return csv_files[0]

        return None

    def auto_detect_api_dir(self) -> Optional[Path]:


        """自动检测API响应目录"""

        if self.results_file:

            # 从测试结果文件路径推导API响应目录

            api_dir = self.results_file.parent / "api_responses"

            if api_dir.exists():

                return api_dir

        # 尝试默认路径

        default_api_dir = self.project_root / "output" / "single" / "api_responses"

        if default_api_dir.exists():

            return default_api_dir

        return None

    def count_failures(self) -> int:


        """统计失败案例数量"""

        if not self.results_file or not self.results_file.exists():

            return 0

        rows = try_read_csv(self.results_file)

        if rows is None:

            return 0

        try:

            failure_count = 0

            for row in rows:

                # 判断是否失败：recall=0.0

                recall = float(row.get('recall', 0))

                if recall == 0.0:

                    failure_count += 1

            return failure_count

        except Exception as e:

            logger.error(f"统计失败案例失败: {str(e)}")

            return 0

    def run(self):


        """运行交互式菜单"""

        while True:

            self.show_main_menu()

            choice = input("\n请输入选项 [1-5]: ").strip()

            if choice == "5":

                print("\n感谢使用，再见！")

                break

            elif choice == "1":

                self.analyze_and_suggest()

            elif choice == "2":

                self.validate_mapping()

            elif choice == "3":

                self.show_statistics()

            elif choice == "4":

                self.manual_add_mapping()

            else:

                print("\n无效选项，请重新选择。")

            if choice != "5":

                input("\n按 Enter 键继续...")

    def show_main_menu(self):


        """显示主菜单"""

        failure_count = self.count_failures()

        mapping_count = len(self.identifier_mappings)

        print("\n" + "=" * 50)

        print("  章节映射规则维护工具 v.0")

        print("=" * 50)

        print(f"\n当前状态：")

        print(f"  - 测试结果文件: {self.results_file if self.results_file else '未找到'}")

        print(f"  - 映射规则文件: {self.mapping_file}")

        print(f"  - 失败案例 {failure_count} 条")

        print(f"    (说明: recall=0，表示检索章节与参考章节完全不匹配。)")

        print(f"    (解决: 使用功能分析并生成映射建议，补充映射规则后重新运行评估")

        print(f"  - 现有映射规则: {mapping_count} 条")

        print(f"\n请选择操作：")

        print(f"  [1] 自动分析并生成映射建议（推荐）")

        print(f"  [2] 验证现有映射规则")

        print(f"  [3] 查看映射规则统计")

        print(f"  [4] 手动添加映射规则")

        print(f"  [5] 退出")

    def analyze_and_suggest(self):


        """分析并生成建议（交互式）"""

        print("\n正确在分析测试结果...")

        if not self.results_file or not self.results_file.exists():

            print("错误: 未找到测试结果文件。")

            return

        if not self.api_dir or not self.api_dir.exists():

            print("错误: 未找到API响应目录")

            return

        # 询问用户分析范围

        print("\n请选择分析范围：")

        print("  [1] 仅分析失败案例（recall=0）")

        print("  [2] 分析所有测试结果（用于补充映射规则的标识符）")

        scope_choice = input("\n请输入选项 [1/2] (默认: 2): ").strip() or "2"

        if scope_choice == "1":

            # 读取失败案例

            failures = self._load_failures()

            print(f"  读取测试结果: {len(failures)} 条失败案例")

        else:

            # 读取所有测试结果            failures = self._load_all_test_cases()

            print(f"  读取测试结果: {len(failures)} 条测试案例")

        # 提取标识符并生成建议

        suggestions = []

        for failure in failures:

            suggestion = self._generate_suggestion(failure)

            if suggestion:

                suggestions.append(suggestion)

        print(f"  生成映射建议: {len(suggestions)} 条")

        # 按置信度分类

        high_confidence = [s for s in suggestions if s.get('confidence') == 'high']

        medium_confidence = [s for s in suggestions if s.get('confidence') == 'medium']

        low_confidence = [s for s in suggestions if s.get('confidence') == 'low']

        # 显示建议摘要

        print("\n" + "=" * 50)

        print("  映射建议摘要")

        print("=" * 50)

        print(f"\n高置信度建议（推荐应用）: {len(high_confidence)} 条")

        for s in high_confidence[:5]:  # 只显示前5条
            print(f"  - {s['reference_chapter']} → {s['retrieved_chapter']} "
                  f"({s.get('object_dict_id', 'N/A')}, {', '.join(s.get('keywords', [])[:2])})")

        if len(high_confidence) > 5:

            print(f"  ... 还有 {len(high_confidence) - 5} 条")

        print(f"\n中等置信度建议 {len(medium_confidence)} 条")

        print(f"低置信度建议: {len(low_confidence)} 条")

        # 操作选项

        print("\n请选择操作：")

        print("  [1] 应用所有高置信度建议")

        print("  [2] 查看详细建议列表（可选择性应用）")

        print("  [3] 保存建议到文件（稍后处理）")

        print("  [4] 取消，返回主菜单")

        action = input("\n请输入选项 [1-4]: ").strip()

        if action == "1":

            self._apply_suggestions(high_confidence)

        elif action == "2":

            self._show_detailed_suggestions(suggestions)

        elif action == "3":

            self._save_suggestions(suggestions)

        elif action == "4":

            return

        else:

            print("无效选项，返回主菜单")

    def _load_failures(self) -> List[Dict[str, Any]]:


        """加载失败案例"""

        failures = []

        rows = try_read_csv(self.results_file)

        if rows is None:

            return failures

        try:

            for row in rows:

                recall = float(row.get('recall', 0))

                if recall == 0.0:

                    # CSV列名：test_index, question, answer, original_answer, answer_title, reference, ...

                    # reference 是标注章节(ground truth)
                    # answer 是检索到的章节编号, answer_title 是检索到的完整章节文件
                    reference_chapter = row.get('reference', '')  # 标注章节

                    answer_title = row.get('answer_title', '') or row.get('original_answer', '')  # 检索到的完整章节文件
                    answer = row.get('answer', '')  # 检索到的章节编号(备用)                    

                    # 如果没有 answer_title, 使用 answer(章节编号)

                    if not answer_title:

                        answer_title = answer

                    failures.append({

                        'test_index': row.get('test_index', ''),

                        'question': row.get('question', ''),

                        'reference': reference_chapter,  # 标注章节

                        'answer': answer_title,  # 检索到的完整章节文件
                        'theme': row.get('theme', ''),

                    })

        except Exception as e:

            logger.error(f"加载失败案例失败: {str(e)}")

        return failures

    def _load_all_test_cases(self) -> List[Dict[str, Any]]:


        """加载所有测试案例(用于补充映射规则的标识符)"""

        test_cases = []

        rows = try_read_csv(self.results_file)

        if rows is None:

            return test_cases

        try:

            for row in rows:

                # 只要reference和answer都有值就加载

                reference = row.get('reference', '').strip()

                answer = row.get('answer', '').strip()

                if reference and answer:

                    test_cases.append({

                        'test_index': row.get('test_index', ''),

                        'question': row.get('question', ''),

                        'reference': reference,

                        'answer': answer,

                        'theme': row.get('theme', ''),

                    })

        except Exception as e:

            logger.error(f"加载测试案例失败: {str(e)}")

        return test_cases

    def _generate_suggestion(self, failure: Dict[str, Any]) -> Optional[Dict[str, Any]]:


        """为单个失败案例生成映射建议"""

        reference_chapter = failure.get('reference', '')

        retrieved_chapter = failure.get('answer', '')

        test_index = failure.get('test_index', '')

        if not reference_chapter or not retrieved_chapter:

            return None

        # 尝试从API响应中提取标识符

        api_response_file = self.api_dir / f"test_{test_index}_{failure.get('question', '')[:50]}.json"

        if not api_response_file.exists():

            # 尝试其他可能的文件名格式

            for f in self.api_dir.glob(f"test_{test_index}_*.json"):

                api_response_file = f

                break

        retrieved_identifiers = None

        if api_response_file.exists():

            try:

                with open(api_response_file, 'r', encoding='utf-8') as f:

                    api_data = json.load(f)

                    chunks = api_data.get('response', {}).get('data', {}).get('chunks', [])

                    if chunks:

                        # 从第一个chunk提取标识符
                        first_chunk = chunks[0]

                        important_keywords = first_chunk.get('important_keywords', [])

                        if important_keywords:

                            retrieved_identifiers = ChapterIdentifierExtractor.extract_from_important_keywords(

                                important_keywords

                            )

            except Exception as e:

                logger.debug(f"读取API响应失败: {str(e)}")

        # 如果无效法从API响应提取，从章节文本提取

        if not retrieved_identifiers or not retrieved_identifiers.get('object_dict_id'):

            retrieved_identifiers = ChapterIdentifierExtractor.extract_identifiers(retrieved_chapter)

        reference_identifiers = ChapterIdentifierExtractor.extract_identifiers(reference_chapter)

        # 计算置信度
        confidence = 'low'

        # 方法：如果两者都有对象字典编号，且相同，则为高置信度

        retrieved_obj_id = retrieved_identifiers.get('object_dict_id')

        reference_obj_id = reference_identifiers.get('object_dict_id')

        if retrieved_obj_id and reference_obj_id:

            # 统一转换为大写进行比较（因为提取器返回的是大写）

            if retrieved_obj_id.upper() == reference_obj_id.upper():

                confidence = 'high'

        # 方法2: 如果只有一方有对象字典编号, 但关键词匹配, 则为中等置信度
        if confidence == 'low':

            retrieved_keywords = retrieved_identifiers.get('keywords', [])

            reference_keywords = reference_identifiers.get('keywords', [])

            if retrieved_keywords and reference_keywords:

                # 检查关键词是否匹配(不区分大小写, 支持部分匹配)
                retrieved_keywords_lower = [k.lower() for k in retrieved_keywords]

                reference_keywords_lower = [k.lower() for k in reference_keywords]

                # 检查是否有任何关键词匹配
                keyword_match = False

                for rk in retrieved_keywords_lower:

                    for refk in reference_keywords_lower:

                        # 完全匹配或包含关                        if rk == refk or rk in refk or refk in rk:

                            keyword_match = True

                            break

                    if keyword_match:

                        break

                if keyword_match:

                    # 如果有关键词匹配，且至少一方有对象字典编号，则为中等置信度

                    if retrieved_obj_id or reference_obj_id:

                        confidence = 'medium'

                    # 如果两者都有关键词匹配但都没有对象字典编号，也为中等置信度

                    else:

                        confidence = 'medium'

        # 方法3: 如果只有一方有对象字典编号, 但章节编号相近(如 8.2.57 vs 8.2.55), 也为中等置信度
        if confidence == 'low':

            retrieved_chapter_num = self._extract_chapter_number(retrieved_chapter)

            reference_chapter_num = self._extract_chapter_number(reference_chapter)

            if retrieved_chapter_num and reference_chapter_num:

                # 检查章节编号是否相近(同一父章节下)
                retrieved_parts = retrieved_chapter_num.split('.')

                reference_parts = reference_chapter_num.split('.')

                # 如果前两级相同（8.2.57 8.2.55），且至少一方有对象字典编号

                if len(retrieved_parts) >= 2 and len(reference_parts) >= 2:

                    if retrieved_parts[0] == reference_parts[0] and retrieved_parts[1] == reference_parts[1]:

                        if retrieved_obj_id or reference_obj_id:

                            confidence = 'medium'

        # 方法4：基于标题相似度的匹配（去除章节号和英文部分后，标题完全相同）
        # 例如："5.6.1. CANopen PDO通信控制报文 / CANopen PDO communication message control"

        # 和 "5.5.1 CANopen PDO 通信控制报文" 的标题部分相同，应该生成映射建议

        if confidence == 'low':

            try:

                title_match = self._is_title_match(retrieved_chapter, reference_chapter)

                if title_match:

                    confidence = 'high'  # 标题完全相同, 高置信度
                    logger.debug(f"Test {test_index}: 标题匹配成功 - '{retrieved_chapter}' vs '{reference_chapter}'")

            except Exception as e:

                logger.warning(f"Test {test_index}: 标题匹配失败: {str(e)}")

                logger.debug(f"  retrieved_chapter: {retrieved_chapter}")

                logger.debug(f"  reference_chapter: {reference_chapter}")

        # 提取检索章节的编号部分

        retrieved_chapter_num = self._extract_chapter_number(retrieved_chapter)

        return {

            'reference_chapter': reference_chapter,

            'retrieved_chapter': retrieved_chapter,

            'retrieved_chapter_num': retrieved_chapter_num,

            'object_dict_id': retrieved_identifiers.get('object_dict_id'),

            'keywords': retrieved_identifiers.get('keywords', []),

            'confidence': confidence,

            'source': f"test_{test_index}",

        }

    def _extract_chapter_number(self, chapter_text: str) -> Optional[str]:


        """从章节文本中提取章节编号"""

        if not chapter_text:

            return None

        match = re.match(r'^(\d+(?:\.\d+)*)', chapter_text)

        if match:

            return match.group()

        return None

    def _extract_title_only(self, chapter: str) -> str:


        """
        提取章节标题(去除章节号和英文部分)

        例如:
        - "5.6.1. CANopen PDO通信控制报文 / CANopen PDO communication message control" 

          -> "CANopen PDO通信控制报文"

        - "5.5.1 CANopen PDO 通信控制报文"
          -> "CANopen PDO 通信控制报文"

        Args:
            chapter: 完整章节文本

        Returns:
            去除章节号和英文部分后的标题

        """

        if not chapter:

            return ""

        # 去除章节号（开头的数字.数字.数字...格式）
        # 匹配模式：数字.数字...（可能后面有空格、点）
        pattern = r'^(\d+(?:\.\d+)*)[\s\.]*'

        title = re.sub(pattern, '', chapter.strip())

        # 去除英文部分（用"/"分隔）
        # 如果包含"/"，只保留"/"前面的部分（这是中文标题部分）
        if '/' in title:

            title = title.split('/')[0].strip()

        # 注意：不要移除标题中的英文单词（如"CANopen"、"PDO"等）
        # 因为这些是标题的一部分，不是独立的英文说明

        # 只移除独立的英文说明部分（如"Chapter 1"、"Section 2"等）

        # 查找英文关键词（Chapter、Section、Part等），这些通常表示英文说明的开始
        english_keywords = ['Chapter', 'Section', 'Part', 'CHAPTER', 'SECTION', 'PART']

        for keyword in english_keywords:

            idx = title.find(keyword)

            if idx != -1:

                # 找到英文关键词，只保留前面的部分

                title = title[:idx].strip()

                break

        # 去除首尾空格和标点符号
        title = title.strip().strip('，。')

        return title

    def _is_title_match(self, chapter_a: str, chapter_b: str) -> bool:


        """
        基于标题相似度判断两个章节是否匹配
        规则: 去除章节号和英文部分(用"/"分隔)后, 如果标题完全相同(相似度为1), 则认为匹配

        例如:
        - "5.6.1. CANopen PDO通信控制报文 / CANopen PDO communication message control"
        - "5.5.1 CANopen PDO 通信控制报文"
        这两个章节的标题部分都是"CANopen PDO通信控制报文", 应该匹配

        Args:
            chapter_a: 章节A(完整文本)
            chapter_b: 章节B(完整文本)

        Returns:
            如果标题完全相同(相似度为1)则返回True, 否则返回False
        """

        if not chapter_a or not chapter_b:

            return False

        # 提取标题（去除章节号和英文部分）

        title_a = self._extract_title_only(chapter_a)

        title_b = self._extract_title_only(chapter_b)

        if not title_a or not title_b:

            return False

        # 标准化标题：去除空格、标点符号，统一格式

        def normalize_title(title: str) -> str:


            # 去除所有空            title = re.sub(r'\s+', '', title)

            # 去除常见标点符号

            title = re.sub(r'[，。、：；！？\.,:;!]', '', title)

            return title.strip()

        normalized_title_a = normalize_title(title_a)

        normalized_title_b = normalize_title(title_b)

        # 只有标准化后的标题完全相同（相似度为），才认为匹        return normalized_title_a == normalized_title_b

    def _record_change_before_apply(self, suggestion: Dict[str, Any]) -> Optional[Dict[str, Any]]:


        """记录应用前的状态, 用于撤回"""

        reference_chapter = suggestion.get('reference_chapter')

        if not reference_chapter:

            return None

        # 加载当前映射状        if self.mapping_file.exists():

            try:

                with open(self.mapping_file, 'r', encoding='utf-8') as f:

                    data = json.load(f)

                    mappings = data.get('mappings', {})

                    # 记录应用前的状                    old_mapping = mappings.get(reference_chapter)

                    return {

                        'reference_chapter': reference_chapter,

                        'old_mapping': old_mapping.copy() if old_mapping else None,

                        'suggestion': suggestion.copy()

                    }

            except Exception:

                return None

        return {

            'reference_chapter': reference_chapter,

            'old_mapping': None,

            'suggestion': suggestion.copy()

        }

    def _apply_suggestions(self, suggestions: List[Dict[str, Any]]):


        """应用映射建议"""

        if not suggestions:

            print("没有可应用的建议")

            return

        print(f"\n正确在应用 {len(suggestions)} 条映射建议...")

        # 加载现有映射

        if self.mapping_file.exists():

            try:

                with open(self.mapping_file, 'r', encoding='utf-8') as f:

                    data = json.load(f)

                    mappings = data.get('mappings', {})

            except Exception:

                mappings = {}

        else:

            mappings = {}

        # 添加新映射或更新现有映射的标识符

        added_count = 0

        updated_count = 0

        for suggestion in suggestions:

            reference_chapter = suggestion['reference_chapter']

            retrieved_chapter_num = suggestion.get('retrieved_chapter_num')

            if not retrieved_chapter_num:

                continue

            # 检查是否已存在

            if reference_chapter not in mappings:

                # 添加新映射
                mappings[reference_chapter] = {

                    'object_dict_id': suggestion.get('object_dict_id'),

                    'keywords': suggestion.get('keywords', []),

                    'equivalent_chapters': [retrieved_chapter_num]

                }

                added_count += 1

            else:

                # 更新现有映射的标识符（如果当前为空）

                existing_mapping = mappings[reference_chapter]

                object_dict_id = suggestion.get('object_dict_id')

                keywords = suggestion.get('keywords', [])

                # 如果现有映射的标识符为空，则更新

                if not existing_mapping.get('object_dict_id') and object_dict_id:

                    existing_mapping['object_dict_id'] = object_dict_id

                    updated_count += 1

                if not existing_mapping.get('keywords') and keywords:

                    existing_mapping['keywords'] = keywords

                    if updated_count == 0:

                        updated_count += 1

                elif keywords and existing_mapping.get('keywords'):

                    # 合并关键词，去重

                    existing_keywords = set(existing_mapping.get('keywords', []))

                    new_keywords = set(keywords)

                    merged_keywords = list(existing_keywords | new_keywords)

                    if len(merged_keywords) > len(existing_keywords):

                        existing_mapping['keywords'] = merged_keywords

                        if updated_count == 0:

                            updated_count += 1

        # 保存映射文件

        data = {

            'version': '1.0',

            'description': '基于对象字典编号和关键词的章节映射规则',

            'mappings': mappings,

            'fallback': 'keep_original'

        }

        try:

            self.mapping_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.mapping_file, 'w', encoding='utf-8') as f:

                json.dump(data, f, ensure_ascii=False, indent=2)

            if added_count > 0:

                print(f"成功添加 {added_count} 条新映射规则")

            if updated_count > 0:

                print(f"成功更新 {updated_count} 条现有映射规则的标识符")

            if added_count == 0 and updated_count == 0:

                print("ℹ️  没有需要更新或添加的映射规则")

            print(f"映射规则已保存到: {self.mapping_file}")

            # 重新加载映射

            self.rule_manager.reload_rules()

            self.identifier_mappings = self.rule_manager.get_identifier_mapping()

            self.identifier_matcher = ChapterIdentifierMatcher(self.identifier_mappings)

        except Exception as e:

            print(f"错误: 保存映射规则失败: {str(e)}")

    def _show_detailed_suggestions(self, suggestions: List[Dict[str, Any]]):


        """显示详细建议列表"""

        print("\n" + "=" * 50)

        print("  详细映射建议列表")

        print("=" * 50)

        print("\n提示:")

        print("  [/Enter] 应用当前建议")

        print("  [2] 跳过当前建议")

        print("  [3] 跳过当前并跳过剩余所有建议")

        print("  [4] 查看并撤回最近的应用")

        print("  [5] 退出并返回主菜单")

        skip_remaining = False

        for i, suggestion in enumerate(suggestions, 1):

            if skip_remaining:

                break

            confidence_label = {

                'high': '[高]',

                'medium': '[中]',

                'low': '[低]'

            }.get(suggestion.get('confidence', 'low'), '[低]')

            print(f"\n[{i}/{len(suggestions)}] {confidence_label} {suggestion['reference_chapter']} {suggestion.get('retrieved_chapter_num', 'N/A')}")

            print(f"  标识符 {suggestion.get('object_dict_id', 'N/A')}")

            print(f"  关键词 {', '.join(suggestion.get('keywords', [])[:3])}")

            print(f"  来源: {suggestion.get('source', 'N/A')}")

            while True:

                apply = input("  操作 [(应用)/2(跳过)/3(跳过剩余)/4(撤回)/5(退]: ").strip()

                if apply == "1" or apply == "":

                    # 记录应用前的状态，用于撤回

                    change_record = self._record_change_before_apply(suggestion)

                    self._apply_suggestions([suggestion])

                    if change_record:

                        self.recent_applied_changes.append(change_record)

                    break

                elif apply == '2':

                    break

                elif apply == '3':

                    skip_remaining = True

                    break

                elif apply == '4':

                    self._undo_recent_changes()

                    # 继续当前建议的处理                    continue

                elif apply == '5':

                    return

                else:

                    print("  无效选项，请重新输入")

    def _save_suggestions(self, suggestions: List[Dict[str, Any]]):


        """保存建议到文件"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = self.project_root / "output" / f"mapping_suggestions_{timestamp}.json"

        try:

            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:

                json.dump({'suggestions': suggestions}, f, ensure_ascii=False, indent=2)

            print(f"建议已保存到: {output_file}")

        except Exception as e:

            print(f"错误: 保存建议失败: {str(e)}")

    def validate_mapping(self):


        """验证现有映射规则"""

        print("\n正确在验证映射规则...")

        if not self.identifier_mappings:

            print("没有映射规则需要验证")

            return

        # 第一步：格式验证

        valid_count = 0

        invalid_count = 0

        invalid_mappings = []

        for old_chapter, mapping_info in self.identifier_mappings.items():

            # 简单验证: 检查格式是否正确
            if not isinstance(mapping_info, dict):

                invalid_count += 1

                invalid_mappings.append(old_chapter)

                continue

            equivalent_chapters = mapping_info.get('equivalent_chapters', [])

            if not equivalent_chapters:

                invalid_count += 1

                invalid_mappings.append(old_chapter)

                continue

            valid_count += 1

        print(f"  读取映射规则: {len(self.identifier_mappings)} 条")

        print(f"  验证格式: {'全部正确' if invalid_count == 0 else f'{invalid_count} 条格式错误'}")

        print(f"  测试有效性: {valid_count} 条有效, {invalid_count} 条无效")

        # 第二步：分析测试结果中的使用情况

        unused_mappings = []

        if self.results_file and self.results_file.exists():

            print("\n正确在分析测试结果中的映射规则使用情况...")

            unused_mappings = self._analyze_mapping_usage()

            print(f"  分析完成: {len(unused_mappings)} 条映射规则未被使用或已不再需要")

        # 显示验证结果

        print("\n" + "=" * 50)

        print("  验证结果")

        print("=" * 50)

        print(f"\n有效性映射: {valid_count} 条")

        print(f"无效映射(格式错误): {invalid_count} 条")

        print(f"未使用映射(可删除): {len(unused_mappings)} 条")

        # 显示无效映射

        if invalid_mappings:

            print(f"\n无效映射列表（前5条）:")

            for old_chapter in invalid_mappings[:5]:

                print(f"  - {old_chapter} (格式错误)")

            if len(invalid_mappings) > 5:

                print(f"  ... 还有 {len(invalid_mappings) - 5} 条")

        # 显示未使用映射        if unused_mappings:

            print(f"\n未使用映射列表（条）:")

            for old_chapter, reason in unused_mappings[:5]:

                print(f"  - {old_chapter} ({reason})")

            if len(unused_mappings) > 5:

                print(f"  ... 还有 {len(unused_mappings) - 5} 条")

        # 操作选项

        print("\n请选择操作：")

        option_num = 1

        options_map = {}

        if invalid_mappings:

            print(f"  [{option_num}] 删除无效映射规则（格式错误）")

            options_map[str(option_num)] = ('delete_invalid', invalid_mappings)

            option_num += 1

        if unused_mappings:

            print(f"  [{option_num}] 删除未使用的映射规则（已不再需要）")

            options_map[str(option_num)] = ('delete_unused', unused_mappings)

            option_num += 1

        if invalid_mappings and unused_mappings:

            print(f"  [{option_num}] 删除所有无效和未使用的映射规则")

            options_map[str(option_num)] = ('delete_all', (invalid_mappings, unused_mappings))

            option_num += 1

        print(f"  [{option_num}] 保留所有映射规则")

        options_map[str(option_num)] = ('keep', None)

        option_num += 1

        print(f"  [{option_num}] 返回主菜单")

        options_map[str(option_num)] = ('return', None)

        action = input(f"\n请输入选项 [1-{option_num}]: ").strip()

        if action in options_map:

            action_type, action_data = options_map[action]

            if action_type == 'delete_invalid':

                self._delete_invalid_mappings(action_data)

            elif action_type == 'delete_unused':

                self._delete_unused_mappings(action_data)

            elif action_type == 'delete_all':

                invalid, unused = action_data

                all_to_delete = invalid + [m[0] for m in unused]

                self._delete_invalid_mappings(all_to_delete)

            elif action_type == 'keep':

                print("已保留所有映射规则")

            elif action_type == 'return':

                return

        else:

            print("无效选项或没有可删除的映射规则")

    def _analyze_mapping_usage(self) -> List[Tuple[str, str]]:


        """

        分析映射规则在测试结果中的使用情况

        Returns:
            未使用的映射规则列表, 格式: [(old_chapter, reason), ...]

        """

        if not self.results_file or not self.results_file.exists():

            return []

        rows = try_read_csv(self.results_file)

        if rows is None:

            return []

        # 统计每个映射规则的使用情况
        mapping_usage = {}  # {old_chapter: {'used': bool, 'success_count': int, 'total_count': int}}

        # 初始化所有映射规则
        for old_chapter in self.identifier_mappings.keys():

            mapping_usage[old_chapter] = {

                'used': False,

                'success_count': 0,

                'total_count': 0

            }

        # 分析测试结果

        try:

            for row in rows:

                reference = row.get('reference', '').strip()

                answer = row.get('answer', '').strip()

                recall = float(row.get('recall', 0))

                if not reference or not answer:

                    continue

                # 检查这个测试案例是否使用了某个映射规则

                # 方法：检reference answer 是否匹配某个映射规则

                for old_chapter, mapping_info in self.identifier_mappings.items():

                    equivalent_chapters = mapping_info.get('equivalent_chapters', [])

                    if not equivalent_chapters:

                        continue

                    # 提取章节编号

                    reference_num = self._extract_chapter_number(reference)

                    answer_num = self._extract_chapter_number(answer)

                    old_chapter_num = self._extract_chapter_number(old_chapter)

                    # 检查是否使用了这个映射规则

                    used = False

                    # 情况：reference old_chapter，answer equivalent_chapter

                    if reference_num and old_chapter_num and reference_num == old_chapter_num:

                        if answer_num and answer_num in [self._extract_chapter_number(eq) for eq in equivalent_chapters]:

                            used = True

                    # 情况2：answer old_chapter，reference equivalent_chapter

                    elif answer_num and old_chapter_num and answer_num == old_chapter_num:

                        if reference_num and reference_num in [self._extract_chapter_number(eq) for eq in equivalent_chapters]:

                            used = True

                    # 情况3: 两者都是equivalent_chapter(这种情况较少见)
                    elif reference_num and answer_num:

                        ref_in_equiv = reference_num in [self._extract_chapter_number(eq) for eq in equivalent_chapters]

                        ans_in_equiv = answer_num in [self._extract_chapter_number(eq) for eq in equivalent_chapters]

                        if ref_in_equiv and ans_in_equiv:

                            used = True

                    if used:

                        mapping_usage[old_chapter]['used'] = True

                        mapping_usage[old_chapter]['total_count'] += 1

                        if recall == 1.0:

                            mapping_usage[old_chapter]['success_count'] += 1

        except Exception as e:

            logger.error(f"分析映射规则使用情况失败: {str(e)}")

            return []

        # 找出未使用的映射规则

        unused_mappings = []

        for old_chapter, usage_info in mapping_usage.items():

            if not usage_info['used']:

                unused_mappings.append((old_chapter, "未在测试结果中使用"))

            elif usage_info['total_count'] > 0 and usage_info['success_count'] == usage_info['total_count']:

                # 所有使用该映射规则的案例都已成功, 可能不再需要
                unused_mappings.append((old_chapter, f"所有案例已成功({usage_info['success_count']}/{usage_info['total_count']})"))

        return unused_mappings

    def _delete_unused_mappings(self, unused_mappings: List[Tuple[str, str]]):


        """删除未使用的映射规则"""

        if not unused_mappings:

            print("没有未使用的映射规则需要删除")

            return

        unused_chapters = [m[0] for m in unused_mappings]

        # 显示将要删除的映射规则        print(f"\n将要删除 {len(unused_chapters)} 条未使用的映射规则：")

        for old_chapter, reason in unused_mappings[:10]:

            print(f"  - {old_chapter} ({reason})")

        if len(unused_mappings) > 0:

            print(f"  ... 还有 {len(unused_mappings) - 10} 条")

        confirm = input(f"\n确认删除{len(unused_chapters)} 条映射规则？[Y/N]: ").strip().upper()

        if confirm == "Y":

            self._delete_invalid_mappings(unused_chapters)

        else:

            print("已取消删除操作")

    def _delete_invalid_mappings(self, invalid_chapters: List[str]):


        """删除无效映射规则"""

        # 重新加载并过滤
        if self.mapping_file.exists():

            try:

                with open(self.mapping_file, 'r', encoding='utf-8') as f:

                    data = json.load(f)

                    mappings = data.get('mappings', {})

                    # 删除无效映射

                    for chapter in invalid_chapters:

                        mappings.pop(chapter, None)

                    # 保存

                    data['mappings'] = mappings

                    with open(self.mapping_file, 'w', encoding='utf-8') as f:

                        json.dump(data, f, ensure_ascii=False, indent=2)

                    print(f"已删除{len(invalid_chapters)} 条无效映射规则")

                    # 重新加载

                    self.rule_manager.reload_rules()

                    self.identifier_mappings = self.rule_manager.get_identifier_mapping()

                    self.identifier_matcher = ChapterIdentifierMatcher(self.identifier_mappings)

            except Exception as e:

                print(f"错误: 删除失败: {str(e)}")

    def show_statistics(self):


        """查看映射规则统计"""

        print("\n映射规则统计")

        print("=" * 50)

        print(f"\n总映射规则数量: {len(self.identifier_mappings)} 条")

        # 按置信度分类（如果有的话        # 这里简化处理，实际可以根据标识符的完整性分        

        print(f"\n覆盖失败案例: {self.count_failures()} 条")

        input("\n按 Enter 键返回主菜单...")

    def manual_add_mapping(self):


        """手动添加映射规则"""

        print("\n手动添加映射规则")

        print("=" * 50)

        old_chapter = input("\n请输入旧章节编号: ").strip()

        new_chapter = input("请输入新章节编号: ").strip()

        object_dict_id = input("请输入对象字典编号(可选, 如x6082): ").strip() or None

        keywords_input = input("请输入关键词(用逗号分隔, 如\"结果速度,End velocity\"): ").strip()

        keywords = [k.strip() for k in keywords_input.split(',') if k.strip()] if keywords_input else []

        print(f"\n确认添加映射规则：")

        print(f"  旧章节 {old_chapter}")

        print(f"  新章节 {new_chapter}")

        print(f"  对象字典编号: {object_dict_id or 'N/A'}")

        print(f"  关键词 {', '.join(keywords) if keywords else 'N/A'}")

        confirm = input("\n  [1] 确认添加  [2] 取消: ").strip()

        if confirm == "1":

            # 加载现有映射

            if self.mapping_file.exists():

                try:

                    with open(self.mapping_file, 'r', encoding='utf-8') as f:

                        data = json.load(f)

                        mappings = data.get('mappings', {})

                except Exception:

                    mappings = {}

            else:

                mappings = {}

            # 添加新映射
            mappings[old_chapter] = {

                'object_dict_id': object_dict_id,

                'keywords': keywords,

                'equivalent_chapters': [new_chapter]

            }

            # 保存

            data = {

                'version': '1.0',

                'description': '基于对象字典编号和关键词的章节映射规则',

                'mappings': mappings,

                'fallback': 'keep_original'

            }

            try:

                self.mapping_file.parent.mkdir(parents=True, exist_ok=True)

                with open(self.mapping_file, 'w', encoding='utf-8') as f:

                    json.dump(data, f, ensure_ascii=False, indent=2)

                print("映射规则已添加")

                # 重新加载

                self.rule_manager.reload_rules()

                self.identifier_mappings = self.rule_manager.get_identifier_mapping()

                self.identifier_matcher = ChapterIdentifierMatcher(self.identifier_mappings)

            except Exception as e:

                print(f"错误: 添加失败: {str(e)}")

def main():


    """主函数"""

    tool = MappingTool()

    tool.run()

if __name__ == "__main__":

    main()

