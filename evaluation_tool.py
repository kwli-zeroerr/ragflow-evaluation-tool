"""
RagFlow 检索方案评测自动化程序
支持批量测试、多指标计算、结果对比和可视化
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
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 创建统一的输入和输出目录
input_dir = Path("input")
input_dir.mkdir(exist_ok=True)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# 确保日志目录存在
logs_dir = output_dir / "logs"
logs_dir.mkdir(exist_ok=True)

# 创建一个安全的StreamHandler，处理编码错误
class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # 尝试写入，如果编码错误则使用错误处理
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # 如果编码失败，使用ASCII安全的方式替换
                safe_msg = msg.encode('ascii', 'replace').decode('ascii')
                stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# 配置日志处理器
file_handler = logging.FileHandler(
    logs_dir / f'evaluation_tool_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    encoding='utf-8'
)
stream_handler = SafeStreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """测试用例结构"""
    question: str
    answer: str
    reference: str  # 标注的章节信息
    type: Optional[str] = None
    theme: Optional[str] = None


@dataclass
class RetrievalConfig:
    """检索配置参数"""
    dataset_ids: List[str]  # 必需：数据集ID列表
    top_k: int = 5
    similarity_threshold: float = 0.0
    vector_similarity_weight: Optional[float] = None
    document_ids: Optional[List[str]] = None
    page: Optional[int] = None
    page_size: Optional[int] = None
    rerank_id: Optional[str] = None
    keyword: Optional[bool] = None
    highlight: Optional[bool] = None
    cross_languages: Optional[List[str]] = None
    metadata_condition: Optional[Dict[str, Any]] = None
    use_kg: Optional[bool] = None


class RagFlowClient:
    """RagFlow API客户端"""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self._local_dataset_cache: Optional[Dict[str, Any]] = None
        # 缓存解析后的dataset_ids和document_ids，避免每次search都重新解析
        self._cached_dataset_ids: Optional[List[str]] = None
        self._cached_document_ids: Optional[List[str]] = None
    
    def _load_local_datasets(self) -> Optional[Dict[str, Any]]:
        """
        从本地 datasets.json 加载数据集映射，键为数据集名称，值包含:
        { "id": "<dataset_id>", "documents": { "<doc_name>": "<doc_id>", ... } }
        """
        if self._local_dataset_cache is not None:
            return self._local_dataset_cache
        try:
            path = output_dir / "datasets.json"
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as f:
                self._local_dataset_cache = json.load(f)
            return self._local_dataset_cache
        except Exception as e:
            logger.warning(f"加载本地 datasets.json 失败: {e}")
            return None
    
    def get_datasets_by_theme(self, theme: Optional[str]) -> List[Dict[str, Any]]:
        """
        根据主题(theme)筛选数据集，按数据集名称精确匹配。
        返回匹配的数据集列表（每项至少包含'id'和'name'键）。
        """
        if not theme:
            return []
        # 优先使用本地 datasets.json
        local = self._load_local_datasets()
        if isinstance(local, dict) and theme in local and isinstance(local[theme], dict):
            ds_obj = local[theme]
            ds_id = ds_obj.get("id")
            if ds_id:
                return [{"id": ds_id, "name": theme}]
        
        # 回退到远端API列举
        result = self.list_datasets()
        # 兼容不同响应格式
        datasets = result.get("data", [])
        if not datasets and isinstance(result, list):
            datasets = result
        matched: List[Dict[str, Any]] = []
        for ds in datasets:
            ds_id = ds.get("id") or ds.get("dataset_id") or ds.get("_id")
            ds_name = ds.get("name") or ds.get("dataset_name") or ""
            if ds_name == theme:
                matched.append({"id": ds_id, "name": ds_name})
        return matched
    
    def list_datasets(self, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
        """列出所有数据集"""
        endpoint = f"{self.api_url}/api/v1/datasets"
        params = {
            "page": page,
            "page_size": page_size
        }
        try:
            response = requests.get(endpoint, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"获取数据集列表失败: {str(e)}")
            return {"error": str(e), "data": []}
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """获取数据集详细信息"""
        endpoint = f"{self.api_url}/api/v1/datasets/{dataset_id}"
        try:
            response = requests.get(endpoint, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"获取数据集信息失败: {str(e)}")
            return {"error": str(e)}
    
    def list_documents(self, dataset_id: str, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
        """列出指定数据集中的所有文档"""
        endpoint = f"{self.api_url}/api/v1/datasets/{dataset_id}/documents"
        params = {
            "page": page,
            "page_size": page_size
        }
        try:
            response = requests.get(endpoint, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"获取文档列表失败: {str(e)}")
            return {"error": str(e), "data": []}
    
    def get_document_info(self, dataset_id: str, document_id: str) -> Dict[str, Any]:
        """获取文档详细信息"""
        endpoint = f"{self.api_url}/api/v1/datasets/{dataset_id}/documents/{document_id}"
        try:
            response = requests.get(endpoint, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"获取文档信息失败: {str(e)}")
            return {"error": str(e)}
    
    def get_all_datasets_and_documents(self) -> Tuple[List[str], List[str]]:
        """
        从本地 datasets.json 提取所有 dataset_ids 和 document_ids
        返回: (dataset_ids, document_ids)
        数据格式: {dataset_name: {id: "", documents: {doc_name: doc_id, ...}}}
        使用缓存避免重复解析
        """
        # 如果已缓存，直接返回
        if self._cached_dataset_ids is not None and self._cached_document_ids is not None:
            return self._cached_dataset_ids, self._cached_document_ids
        
        local = self._load_local_datasets()
        if not local or not isinstance(local, dict):
            logger.warning("无法从本地 datasets.json 加载数据，返回空列表")
            self._cached_dataset_ids = []
            self._cached_document_ids = []
            return [], []
        
        dataset_ids = []
        document_ids = []
        
        for dataset_name, dataset_info in local.items():
            if not isinstance(dataset_info, dict):
                continue
            
            # 提取 dataset_id
            dataset_id = dataset_info.get('id')
            if dataset_id:
                dataset_ids.append(dataset_id)
            
            # 提取 document_ids
            documents = dataset_info.get('documents', {})
            if isinstance(documents, dict):
                for doc_name, doc_id in documents.items():
                    if doc_id:
                        document_ids.append(doc_id)
        
        # 缓存结果
        self._cached_dataset_ids = dataset_ids
        self._cached_document_ids = document_ids
        
        logger.info(f"从本地数据集加载: {len(dataset_ids)} 个数据集, {len(document_ids)} 个文档（已缓存）")
        return dataset_ids, document_ids
    
    def search(self, question: str, theme: str, config: RetrievalConfig) -> Dict[str, Any]:
        """调用RagFlow检索API - 使用所有datasets和documents"""
        endpoint = f"{self.api_url}/api/v1/retrieval"

        # 从本地 datasets.json 加载所有 dataset_ids 和 document_ids
        all_dataset_ids, all_document_ids = self.get_all_datasets_and_documents()
        
        # 优先使用本地加载的数据，如果本地没有则使用config中的
        effective_dataset_ids = all_dataset_ids if all_dataset_ids else (config.dataset_ids or [])
        effective_document_ids = all_document_ids if all_document_ids else (config.document_ids or [])
        
        payload = {
            "question": question,
            "dataset_ids": effective_dataset_ids,
            "top_k": config.top_k,
            "similarity_threshold": config.similarity_threshold
        }
        
        # 添加可选参数
        if effective_document_ids:
            payload["document_ids"] = effective_document_ids
        if config.page is not None:
            payload["page"] = config.page
        if config.page_size is not None:
            payload["page_size"] = config.page_size
        if config.vector_similarity_weight is not None:
            payload["vector_similarity_weight"] = config.vector_similarity_weight
        if config.rerank_id is not None:
            payload["rerank_id"] = config.rerank_id
        if config.keyword is not None:
            payload["keyword"] = config.keyword
        if config.highlight is not None:
            payload["highlight"] = config.highlight
        if config.cross_languages is not None:
            payload["cross_languages"] = config.cross_languages
        if config.metadata_condition is not None:
            payload["metadata_condition"] = config.metadata_condition
        if config.use_kg is not None:
            payload["use_kg"] = config.use_kg
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API调用失败: {question[:50]}... - {str(e)}")
            return {"error": str(e), "chunks": []}


class ChapterMatcher:
    """章节匹配器 - 判断章节层级关系"""
    
    # 中文数字到阿拉伯数字的映射
    CHINESE_NUM_MAP = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
        '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
        '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25,
        '二十六': 26, '二十七': 27, '二十八': 28, '二十九': 29, '三十': 30,
        '百': 100, '千': 1000, '万': 10000
    }
    
    @staticmethod
    def remove_english_text(text: str) -> str:
        """移除文本中的英文部分，只保留中文部分
        例如："第一章 储存与使用 Chapter 1 Storage and Handling" -> "第一章 储存与使用"
        """
        if not text:
            return text
        
        # 查找英文关键词（Chapter、Section、Part等），去掉它们及后面的所有内容
        english_keywords = ['Chapter', 'Section', 'Part', 'CHAPTER', 'SECTION', 'PART']
        for keyword in english_keywords:
            idx = text.find(keyword)
            if idx != -1:
                # 找到关键词，只保留前面的部分
                chinese_part = text[:idx].strip()
                return chinese_part
        
        # 如果没有找到英文关键词，尝试查找第一个连续的英文字母单词（长度>=4）
        # 这通常表示英文内容的开始
        pattern = r'\b[A-Za-z]{4,}\b'
        match = re.search(pattern, text)
        if match:
            # 找到长英文单词，只保留前面的部分
            chinese_part = text[:match.start()].strip()
            # 如果中文部分不为空，返回它
            if chinese_part:
                return chinese_part
        
        # 如果以上都不匹配，返回原文本（可能全是中文或没有明显的英文部分）
        return text.strip()
    
    @staticmethod
    def chinese_to_arabic(chinese_num: str) -> Optional[int]:
        """将中文数字转换为阿拉伯数字
        支持：一、二、三...十、十一、十二...二十、二十一...二十七等
        """
        if not chinese_num:
            return None
        
        chinese_num = chinese_num.strip()
        
        # 如果已经是阿拉伯数字，直接返回
        if chinese_num.isdigit():
            return int(chinese_num)
        
        # 处理简单的中文数字（1-99）
        if chinese_num in ChapterMatcher.CHINESE_NUM_MAP:
            return ChapterMatcher.CHINESE_NUM_MAP[chinese_num]
        
        # 处理"二十"、"三十"等十的倍数
        if chinese_num.endswith('十'):
            base = chinese_num[:-1]
            if base == '':
                return 10
            if base in ChapterMatcher.CHINESE_NUM_MAP:
                return ChapterMatcher.CHINESE_NUM_MAP[base] * 10
        
        # 处理"二十一"、"二十七"等十以上的数字
        if len(chinese_num) >= 2 and '十' in chinese_num:
            parts = chinese_num.split('十')
            if len(parts) == 2:
                if parts[0] == '':
                    tens = 1
                else:
                    tens = ChapterMatcher.CHINESE_NUM_MAP.get(parts[0], 0)
                ones = ChapterMatcher.CHINESE_NUM_MAP.get(parts[1], 0)
                return tens * 10 + ones
        
        # 如果无法转换，返回None
        return None
    
    @staticmethod
    def arabic_to_chinese(num: int) -> Optional[str]:
        """将阿拉伯数字转换为中文数字
        支持：1-99的数字转换
        """
        if num < 0 or num > 99:
            return None
        
        # 反转映射表
        reverse_map = {v: k for k, v in ChapterMatcher.CHINESE_NUM_MAP.items() if v <= 30}
        
        if num in reverse_map:
            return reverse_map[num]
        
        # 处理十以上的数字
        if num >= 10:
            tens = num // 10
            ones = num % 10
            if tens == 1:
                if ones == 0:
                    return '十'
                else:
                    return '十' + reverse_map.get(ones, '')
            else:
                tens_chinese = reverse_map.get(tens, '')
                if ones == 0:
                    return tens_chinese + '十'
                else:
                    return tens_chinese + '十' + reverse_map.get(ones, '')
        
        return None
    
    @staticmethod
    def normalize_chapter(chapter: str) -> str:
        """标准化章节格式，提取纯数字章节编号，并去掉英文部分"""
        if not chapter:
            return ""
        
        chapter = str(chapter).strip()
        
        # 移除英文部分
        chapter = ChapterMatcher.remove_english_text(chapter)
        
        # 移除末尾的点（如 "1.1." -> "1.1"）
        chapter = chapter.rstrip('.')
        
        # 提取数字格式的章节编号（如 "1.1"、"1.2"、"10.1"）
        # 匹配模式：数字.数字.数字...（可能后面有空格和文字）
        pattern = r'^(\d+(?:\.\d+)*)'
        match = re.match(pattern, chapter)
        if match:
            return match.group(1)
        
        # 如果没有匹配到数字格式，尝试匹配中文格式
        patterns = [
            r'第[一二三四五六七八九十\d]+章[^第]*',  # 匹配"第一章"或"第1章"
            r'第[一二三四五六七八九十\d]+章第[一二三四五六七八九十\d]+节',  # 匹配"第一章第一节"
            r'第[一二三四五六七八九十\d]+节',  # 匹配"第一节"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, chapter)
            if match:
                matched = match.group(0).rstrip('.')
                # 再次移除英文部分
                matched = ChapterMatcher.remove_english_text(matched)
                return matched
        
        # 如果没有匹配到任何格式，返回原文本（去除末尾点）
        return chapter.rstrip('.')
    
    @staticmethod
    def extract_chapter_info(text: str) -> Optional[str]:
        """从文本中提取章节信息，改进提取逻辑以避免误提取"""
        if not text or pd.isna(text):
            return None
        
        text_str = str(text).strip()
        if not text_str:
            return None
        
        # 移除英文部分
        text_str = ChapterMatcher.remove_english_text(text_str)
        
        # 先尝试匹配数字格式：1.1, 1.2.3 等
        # 改进逻辑：
        # 1. "5.5 48v供电电源" -> "5.5" (空格后跟非数字文字，停止匹配)
        # 2. "5.2.CAN 通信接口" -> "5.2" (点后直接跟非数字文字，停止匹配)
        # 3. "5.2.3 内容" -> "5.2.3" (点后跟数字，继续匹配)
        
        # 使用正则表达式匹配数字格式的章节编号
        # 匹配模式：数字.数字.数字... 但遇到空格、点后非数字、或结束就停止
        # 方法：逐步匹配，每次检查下一个字符
        numeric_parts = []
        i = 0
        while i < len(text_str) and text_str[i].isdigit():
            # 匹配一个或多个数字
            num_match = re.match(r'^\d+', text_str[i:])
            if num_match:
                num_str = num_match.group(0)
                numeric_parts.append(num_str)
                i += len(num_str)
                
                # 检查后面是否是点
                if i < len(text_str) and text_str[i] == '.':
                    # 检查点后面是否是数字
                    if i + 1 < len(text_str) and text_str[i + 1].isdigit():
                        # 点后是数字，继续匹配
                        i += 1  # 跳过点
                        continue
                    else:
                        # 点后不是数字（如 "5.2.CAN"），停止匹配
                        break
                elif i < len(text_str) and text_str[i] in (' ', '\t', '\n'):
                    # 后面是空格，停止匹配（如 "5.5 48v"）
                    break
                else:
                    # 后面是其他字符，停止匹配
                    break
            else:
                break
        
        # 如果匹配到了数字部分，组合成章节编号
        if numeric_parts:
            numeric = '.'.join(numeric_parts)
            # 处理类似 "8.2.40.0x..." 的情况
            if i < len(text_str) and text_str[i] in ('x', 'X') and numeric.endswith('.0'):
                # 去掉末尾的 ".0"
                numeric = '.'.join(numeric_parts[:-1])
            return numeric
        
        # 匹配中文格式：第一章、第一节、二十六等
        chinese_patterns = [
            r'第[一二三四五六七八九十\d]+章第[一二三四五六七八九十\d]+节',  # 匹配"第一章第一节"
            r'第[一二三四五六七八九十\d]+章[^第]*',  # 匹配"第一章"或"第1章"
            r'第[一二三四五六七八九十\d]+节',  # 匹配"第一节"
            r'([一二三四五六七八九十百千万\d]+)[、，。]',  # 匹配"二十六、"或"二十六，"（纯数字章节）
        ]
        
        for pattern in chinese_patterns:
            match = re.search(pattern, text_str)
            if match:
                matched_text = match.group(0).strip()
                # 移除英文部分
                matched_text = ChapterMatcher.remove_english_text(matched_text)
                # 对于"二十六、"这种格式，只返回数字部分
                if '、' in matched_text or '，' in matched_text or '。' in matched_text:
                    # 提取数字部分（中文数字或阿拉伯数字）
                    num_match = re.search(r'([一二三四五六七八九十百千万\d]+)', matched_text)
                    if num_match:
                        return num_match.group(1).strip()
                return matched_text.rstrip('、，。')
        
        # 如果文本中包含章节信息，尝试提取
        # 例如："1.1 关于本手册" -> "1.1"
        # 例如："第一章 介绍" -> "第一章"
        # 例如："二十六、故障排查说明.md" -> "二十六"
        chapter_match = re.search(r'(\d+(?:\.\d+)*|第[一二三四五六七八九十\d]+章|第[一二三四五六七八九十\d]+节|[一二三四五六七八九十百千万\d]+)[、，。]?', text_str)
        if chapter_match:
            chapter = chapter_match.group(1).strip()
            # 移除英文部分
            chapter = ChapterMatcher.remove_english_text(chapter)
            # 如果是纯中文数字（如"二十六"），直接返回
            if re.match(r'^[一二三四五六七八九十百千万]+$', chapter):
                return chapter
            # 如果是数字格式，返回
            if re.match(r'^\d+(?:\.\d+)*$', chapter):
                return chapter
            # 如果是"第X章"格式，返回
            if '第' in chapter and ('章' in chapter or '节' in chapter):
                return chapter.rstrip('、，。')
        
        # 使用normalize_chapter提取标准化章节
        normalized = ChapterMatcher.normalize_chapter(text_str)
        return normalized if normalized and normalized != text_str.rstrip('.') else None
    
    @staticmethod
    def get_chapter_levels(chapter: str) -> List[int]:
        """将章节编号转换为数字列表，用于比较层级
        例如: "1.1" -> [1, 1], "1.2.3" -> [1, 2, 3], "1" -> [1]
        支持中文数字：如 "二十七" -> [27], "第一章" -> [1]
        """
        if not chapter:
            return []
        
        normalized = ChapterMatcher.normalize_chapter(chapter)
        if not normalized:
            return []
        
        # 如果是数字格式
        if re.match(r'^\d+(?:\.\d+)*$', normalized):
            try:
                return [int(x) for x in normalized.split('.')]
            except (ValueError, AttributeError):
                return []
        
        # 如果是中文格式，尝试转换
        # 处理 "第X章" 格式
        chapter_match = re.match(r'第([一二三四五六七八九十\d]+)章', normalized)
        if chapter_match:
            num_str = chapter_match.group(1)
            num = ChapterMatcher.chinese_to_arabic(num_str)
            if num is not None:
                return [num]
        
        # 处理 "第X节" 格式
        section_match = re.match(r'第([一二三四五六七八九十\d]+)节', normalized)
        if section_match:
            num_str = section_match.group(1)
            num = ChapterMatcher.chinese_to_arabic(num_str)
            if num is not None:
                return [num]
        
        # 处理纯中文数字（如 "二十七"）
        if re.match(r'^[一二三四五六七八九十百千万]+$', normalized):
            num = ChapterMatcher.chinese_to_arabic(normalized)
            if num is not None:
                return [num]
        
        # 如果无法转换，返回空列表
        return []
    
    @staticmethod
    def is_parent_chapter(chapter_a: str, chapter_b: str) -> bool:
        """
        判断chapter_a是否是chapter_b的父章节（大章）
        例如: "1" 是 "1.1" 的父章节, "1.1" 是 "1.1.1" 的父章节
        支持中文数字：如 "二十七" 是 "27.1" 的父章节
        """
        if not chapter_a or not chapter_b:
            return False
        
        # 标准化章节
        chapter_a = ChapterMatcher.normalize_chapter(chapter_a)
        chapter_b = ChapterMatcher.normalize_chapter(chapter_b)
        
        if not chapter_a or not chapter_b:
            return False
        
        # 完全匹配
        if chapter_a == chapter_b:
            return False  # 不是父子关系，是同一章节
        
        # 获取章节层级
        levels_a = ChapterMatcher.get_chapter_levels(chapter_a)
        levels_b = ChapterMatcher.get_chapter_levels(chapter_b)
        
        # 如果都是数字格式，比较层级
        if levels_a and levels_b:
            # chapter_a是chapter_b的父章节，当且仅当：
            # 1. chapter_a的层级数小于chapter_b
            # 2. chapter_a的所有层级数字与chapter_b的前面对应层级相同
            if len(levels_a) < len(levels_b):
                if levels_a == levels_b[:len(levels_a)]:
                    return True
        
        # 处理中文数字和阿拉伯数字的对应关系
        # 如果chapter_a是中文数字，chapter_b是阿拉伯数字格式
        # 例如: "二十七" 对应 "27.1" 的大章
        if levels_a and levels_b:
            # 如果chapter_a只有一个层级（大章），chapter_b有多个层级
            if len(levels_a) == 1 and len(levels_b) > 0:
                # 检查chapter_a的第一层是否等于chapter_b的第一层
                if levels_a[0] == levels_b[0]:
                    return True
        
        # 对于中文格式或其他格式，使用前缀匹配
        if chapter_b.startswith(chapter_a):
            remaining = chapter_b[len(chapter_a):].strip()
            if remaining and (remaining.startswith('第') or remaining.startswith('.') or remaining.startswith('节')):
                return True
        
        return False
    
    @staticmethod
    def is_valid_match(retrieved_chapter: str, reference_chapter: str) -> bool:
        """
        判断检索结果是否有效匹配标注章节
        规则：
        - [OK] 召回"大章"，标注为"小章" → 正确
        - [FAIL] 召回"小章"，标注为"大章" → 错误
        - [OK] 章节完全匹配 → 正确
        - [OK] 中文数字和阿拉伯数字对应关系 → 正确（如 "六" 对应 "6"）
        """
        if not retrieved_chapter or not reference_chapter:
            return False
        
        # 标准化章节
        retrieved_normalized = ChapterMatcher.normalize_chapter(retrieved_chapter)
        reference_normalized = ChapterMatcher.normalize_chapter(reference_chapter)
        
        if not retrieved_normalized or not reference_normalized:
            return False
        
        # 完全匹配（忽略末尾的点）
        if retrieved_normalized == reference_normalized:
            return True
        
        # 检查中文数字和阿拉伯数字的对应关系
        # 例如: "六" 对应 "6", "二十七" 对应 "27"
        retrieved_levels = ChapterMatcher.get_chapter_levels(retrieved_normalized)
        reference_levels = ChapterMatcher.get_chapter_levels(reference_normalized)
        
        if retrieved_levels and reference_levels:
            # 如果层级完全相同，则认为匹配
            if retrieved_levels == reference_levels:
                return True
            
            # 检查是否是中文数字和阿拉伯数字的对应关系
            # 如果都是单层级的数字，且数值相等，则认为匹配
            if len(retrieved_levels) == 1 and len(reference_levels) == 1:
                if retrieved_levels[0] == reference_levels[0]:
                    return True
        
        # 召回"大章"，标注为"小章" → 正确
        # 例如：检索到"1.1"，标注是"1.1.1" → 错误（检索到的是小章）
        # 例如：检索到"1"，标注是"1.1" → 正确（检索到的是大章）
        # 例如：检索到"二十七"，标注是"27.1" → 正确（检索到的是大章）
        if ChapterMatcher.is_parent_chapter(retrieved_normalized, reference_normalized):
            return True
        
        # 召回"小章"，标注为"大章" → 错误
        # 例如：检索到"1.1.1"，标注是"1.1" → 错误
        if ChapterMatcher.is_parent_chapter(reference_normalized, retrieved_normalized):
            return False
        
        # 其他情况（可能是不同章节）→ 错误
        return False


class MetricsCalculator:
    """指标计算器 - 基于章节匹配逻辑"""
    
    @staticmethod
    def calculate_accuracy(retrieved_chapters: List[str], reference_chapter: str) -> Dict[str, float]:
        """
        计算准确率和召回率
        返回: {'correct_count': int, 'total_count': int, 'accuracy': float, 'recall': float}
        """
        if not reference_chapter:
            return {'correct_count': 0, 'total_count': 0, 'accuracy': 0.0, 'recall': 0.0}
        
        correct_count = 0
        total_retrieved = len(retrieved_chapters)
        
        # 检查每个检索结果是否有效匹配
        for retrieved_chapter in retrieved_chapters:
            if ChapterMatcher.is_valid_match(retrieved_chapter, reference_chapter):
                correct_count += 1
        
        # 准确率 = 正确匹配数 / 检索结果总数
        accuracy = correct_count / total_retrieved if total_retrieved > 0 else 0.0
        
        # 召回率 = 是否至少有一个正确匹配（0或1）
        recall = 1.0 if correct_count > 0 else 0.0
        
        return {
            'correct_count': correct_count,
            'total_count': total_retrieved,
            'accuracy': accuracy,
            'recall': recall
        }
    
    @staticmethod
    def recall_at_k(retrieved_chapters: List[str], reference_chapter: str, k: int) -> float:
        """计算Recall@K - 前K个结果中是否至少有一个正确匹配"""
        if not reference_chapter:
            return 0.0
        retrieved_k = retrieved_chapters[:k]
        for chapter in retrieved_k:
            if ChapterMatcher.is_valid_match(chapter, reference_chapter):
                return 1.0
        return 0.0

class EvaluationRunner:
    """评测运行器"""
    
    def __init__(self, client: RagFlowClient, config: RetrievalConfig):
        self.client = client
        self.config = config
        self.calculator = MetricsCalculator()
        self.results = []
        # 创建API响应保存目录
        self.api_responses_dir = output_dir / "api_responses"
        self.api_responses_dir.mkdir(exist_ok=True)
        # 线程安全锁
        self.results_lock = Lock()
        self.logger_lock = Lock()
    
    def load_test_set(self, test_set_path: str) -> List[TestCase]:
        """从Excel文件加载测试集"""
        try:
            # 读取Excel文件
            df = pd.read_excel(test_set_path)
            
            # 验证必需的列
            required_columns = ['question', 'answer', 'reference']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Excel文件缺少必需的列: {missing_columns}")
            
            # 使用itertuples()替代iterrows()，性能更好
            test_cases = []
            for row in df.itertuples(index=False):
                test_cases.append(TestCase(
                    question=str(row.question) if pd.notna(row.question) else '',
                    answer=str(row.answer) if pd.notna(row.answer) else '',
                    reference=str(row.reference) if pd.notna(row.reference) else '',
                    type=str(row.type) if hasattr(row, 'type') and pd.notna(row.type) else None,
                    theme=str(row.theme) if hasattr(row, 'theme') and pd.notna(row.theme) else None
                ))
            
            logger.info(f"从Excel加载测试集: {len(test_cases)} 条用例")
            return test_cases
        except Exception as e:
            logger.error(f"加载测试集失败: {str(e)}")
            raise
    
    def _save_api_response(self, test_case: TestCase, response: Dict[str, Any], test_index: int):
        """保存API响应到单独的文件"""
        # 创建安全的文件名（移除特殊字符）
        safe_question = re.sub(r'[<>:"/\\|?*]', '_', test_case.question[:50])
        filename = f"test_{test_index:03d}_{safe_question}.json"
        filepath = self.api_responses_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'question': test_case.question,
                    'reference': test_case.reference,
                    'theme': test_case.theme,
                    'response': response
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"API响应已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存API响应失败: {str(e)}")
    
    def _extract_chapter_from_important_keywords(self, important_keywords: List[str]) -> Optional[str]:
        """
        从important_keywords列表中提取章节信息
        
        约定：important_keywords[0] = 书名, [1] = 大章, [2] = 小章（如果有）
        优先返回小章，没有则返回大章，没有则None

        例如：
        important_keywords = ["手册名称", "二十六、故障排查说明.md", "26.3.电机堵转报错的排查步骤"]
        - 返回 "26.3"（小章编号）

        important_keywords = ["手册名称", "二十六、故障排查说明.md"]
        - 返回 "二十六"（大章编号）
        """
        if not important_keywords or len(important_keywords) < 2:
            return None

        # 优先提取小章（第三个元素）
        if len(important_keywords) >= 3 and important_keywords[2]:
            chapter_info = ChapterMatcher.extract_chapter_info(str(important_keywords[2]))
            if chapter_info:
                return chapter_info

        # 没有小章则提取大章（第二个元素）
        chapter_info = ChapterMatcher.extract_chapter_info(str(important_keywords[1]))
        if chapter_info:
            return chapter_info

        # 都没有则返回None
        return None
    
    def run_single_test(self, test_case: TestCase, test_index: int = 0) -> Dict[str, Any]:
        """运行单个测试用例"""
        logger.info("=" * 80)
        logger.info(f"测试Question: {test_case.question}")       
        logger.info(f"手册Theme: {test_case.theme}")       
        logger.info(f"标注Reference: {test_case.reference}")
        
        start_time = time.time()
        response = self.client.search(test_case.question, test_case.theme, self.config)
        latency = time.time() - start_time
        
        # 保存完整的API响应到单独的文件
        self._save_api_response(test_case, response, test_index)
        
        if "error" in response:
            logger.warning(f"API返回错误: {response.get('error')}")
            return {
                "test_index": test_index,
                "question": test_case.question,
                "answer": test_case.answer,
                "reference": test_case.reference,
                "type": test_case.type,
                "theme": test_case.theme,
                "error": response["error"],
                "latency": latency
            }
        
        # 从检索结果中提取章节信息
        # 尝试多种可能的字段名
        data = response.get('data', {})
        chunks = data.get('chunks', []) if isinstance(data, dict) else []
        logger.info(f"检索到的chunks总数: {len(chunks)}")
 
        # 基于相似度排序所有chunks
        def get_similarity_score(chunk: Dict[str, Any]) -> Optional[float]:
            # 常见字段：similarity / score / relevance；若有distance则反向
            if isinstance(chunk, dict):
                if isinstance(chunk.get('similarity'), (int, float)):
                    return float(chunk['similarity'])
                if isinstance(chunk.get('score'), (int, float)):
                    return float(chunk['score'])
                if isinstance(chunk.get('relevance'), (int, float)):
                    return float(chunk['relevance'])
                if isinstance(chunk.get('distance'), (int, float)):
                    # 距离越小越好 → 转为相似度
                    return -float(chunk['distance'])
            return None
        
        # 第一步：提取所有chunks的信息，按相似度排序，取topK
        all_chunks_info = []
        for chunk_idx, chunk in enumerate(chunks):
            important_keywords = chunk.get('important_keywords', [])
            chapter_info = None
            if important_keywords:
                chapter_info = self._extract_chapter_from_important_keywords(important_keywords)
            
            sim = get_similarity_score(chunk)
            all_chunks_info.append({
                'index': chunk_idx + 1,
                'chapter': chapter_info,
                'similarity': sim,
                'important_keywords': important_keywords,
                'chunk': chunk
            })
        
        # 按相似度排序
        def sort_key(item):
            return (item['similarity'] is not None, item['similarity'] if item['similarity'] is not None else float('-inf'))
        
        all_chunks_info.sort(key=sort_key, reverse=True)
        
        # 定义topK值：用于召回率等指标计算（对应context=10）
        top_k_for_recall = 10
        # 只取前top_k_for_recall个结果进行验证（节省时间）
        top_k_chunks = all_chunks_info[:top_k_for_recall]
        logger.info(f"选取top {top_k_for_recall} 个结果进行验证")
        
        # 提取标注信息
        reference_chapter = ChapterMatcher.extract_chapter_info(test_case.reference)
        theme_str = str(test_case.theme).strip() if test_case.theme else ""
        
        # 使用相似度最高的第一个结果进行评测（用于评估RAG相似度算法）
        # 这是RAG返回给使用者的第一个结果，用来判断相似度算法的效果
        top1_result = top_k_chunks[0] if top_k_chunks else None
        final_chapter = top1_result.get('chapter') if top1_result else None
        
        # 验证topK中所有结果：important_keywords[0] 是否等于 theme，章节是否等于 reference
        # 用于recall@k计算时，只考虑同时满足两个条件的结果
        
        def verify_item(item):
            """验证单个结果是否同时满足theme和章节匹配"""
            important_keywords = item.get('important_keywords', [])
            chapter_info = item.get('chapter')
            
            # 验证 important_keywords[0] 是否等于 theme（精确匹配）
            theme_match = (
                bool(important_keywords) and 
                bool(theme_str) and 
                str(important_keywords[0]).strip() == theme_str
            )
            
            # 验证章节是否等于 reference（使用is_valid_match判断）
            chapter_match = (
                bool(chapter_info) and 
                bool(reference_chapter) and 
                ChapterMatcher.is_valid_match(chapter_info, reference_chapter)
            )
            
            # 记录验证结果
            item['theme_match'] = theme_match
            item['chapter_match'] = chapter_match
            item['both_match'] = theme_match and chapter_match
            
            return theme_match, chapter_match
        
        # 验证topK中所有结果
        for item in top_k_chunks:
            verify_item(item)
        
        # 获取第一个结果的验证状态（用于accuracy）
        top1_theme_match = False
        top1_chapter_match = False
        if top1_result:
            top1_theme_match = top1_result.get('theme_match', False)
            top1_chapter_match = top1_result.get('chapter_match', False)
        
        # 输出验证结果
        logger.info("-" * 80)
        logger.info(f"标注Theme: {test_case.theme}")
        logger.info(f"标注章节(reference): {reference_chapter}")
        logger.info(f"Top {top_k_for_recall} 个结果的验证情况:")
        
        # 显示topK中每个结果的验证情况
        for rank, item in enumerate(top_k_chunks, 1):
            important_keywords = item.get('important_keywords', [])
            first_keyword = str(important_keywords[0]) if important_keywords and len(important_keywords) > 0 else "无"
            chapter_display = item.get('chapter') if item.get('chapter') else "无章节信息"
            theme_match = item.get('theme_match', False)
            chapter_match = item.get('chapter_match', False)
            both_match = item.get('both_match', False)
            marker = " <-- RAG第一结果" if rank == 1 else ""
            
            logger.info(f"  #{rank}: 标题='{first_keyword}', 章节='{chapter_display}', "
                       f"相似度: {item['similarity'] if item['similarity'] is not None else '无'}")
            logger.info(f"    theme匹配: {'✓' if theme_match else '✗'}, "
                       f"章节匹配: {'✓' if chapter_match else '✗'}, "
                       f"同时满足: {'✓' if both_match else '✗'}{marker}")
        
        # 统计topK中的验证情况（合并为一次遍历，提高性能）
        both_match_count = theme_match_count = chapter_match_count = 0
        for item in top_k_chunks:
            if item.get('both_match', False):
                both_match_count += 1
            if item.get('theme_match', False):
                theme_match_count += 1
            if item.get('chapter_match', False):
                chapter_match_count += 1
        
        logger.info(f"统计: 同时满足={both_match_count}/{top_k_for_recall}, "
                   f"仅theme匹配={theme_match_count}/{top_k_for_recall}, "
                   f"仅章节匹配={chapter_match_count}/{top_k_for_recall}")
        
        logger.info("-" * 80)
        
        # 计算各项指标
        # 1. accuracy：使用RAG返回的第一个结果（用于评估相似度算法的准确度）
        binary_accuracy = 1.0 if top1_chapter_match else 0.0
        
        # 2. 提取topK中所有同时有theme和chapter信息的结果（不管对错，只要同时有theme和chapter）
        # 用于recall@k计算：只要topk中有一个章节匹配reference，就返回1
        all_topk_chapters = []  # 所有topk中有chapter信息的结果（同时有theme和chapter）
        validated_chapters = []  # 同时满足theme和章节匹配的结果（用于其他指标）
        theme_matched_chapters = []  # theme匹配后的所有章节（用于日志显示）
        
        for item in top_k_chunks:
            chapter = item.get('chapter')
            important_keywords = item.get('important_keywords', [])
            # 检查是否同时有theme和chapter信息
            has_theme = bool(important_keywords) and len(important_keywords) > 0
            has_chapter = bool(chapter)
            
            if has_theme and has_chapter:
                # 同时有theme和chapter信息，加入all_topk_chapters用于recall@k计算
                all_topk_chapters.append(chapter)
            
            if chapter:
                if item.get('both_match', False):
                    validated_chapters.append(chapter)
                if item.get('theme_match', False):
                    theme_matched_chapters.append(chapter)
        
        logger.info(f"正确章节: {reference_chapter}")
        logger.info(f"实际章节: {final_chapter if final_chapter else "无"}")
        logger.info(f"候选章节（Top {top_k_for_recall}，theme匹配后的所有章节）: {", ".join(theme_matched_chapters) if theme_matched_chapters else "无"}")
        
        
        # 计算recall相关指标：基于同时满足两个条件的结果
        accuracy_metrics = self.calculator.calculate_accuracy(validated_chapters, reference_chapter)
        
        logger.info(f"准确率结果(Theme和章节是否匹配): {'是' if binary_accuracy == 1.0 else '否'}")
                    #f"{binary_accuracy:.4f}")
        logger.info(f"召回率结果(Theme和章节是否在TOP{top_k_for_recall}结果里): {'是' if accuracy_metrics['recall'] == 1.0 else '否'}")
                    #f"{accuracy_metrics['recall']:.4f}")
        
        # 计算recall@k：使用所有topk中同时有theme和chapter的结果（不管对错）
        # 只要topk中有一个章节匹配reference，就返回1
        # 获取top3, top5, top10的章节列表（用于日志显示）
        top3_chapters = all_topk_chapters[:3] if len(all_topk_chapters) >= 3 else all_topk_chapters
        top5_chapters = all_topk_chapters[:5] if len(all_topk_chapters) >= 5 else all_topk_chapters
        top10_chapters = all_topk_chapters[:10] if len(all_topk_chapters) >= 10 else all_topk_chapters
        
        recall_at_3 = self.calculator.recall_at_k(all_topk_chapters, reference_chapter, 3)
        recall_at_5 = self.calculator.recall_at_k(all_topk_chapters, reference_chapter, 5)
        recall_at_10 = self.calculator.recall_at_k(all_topk_chapters, reference_chapter, 10)
        
        # 在日志中显示recall@k的结果和对应的章节（显示所有topk章节，不管对错）
        logger.info('-'*80)
        logger.info(f"Recall@3: {'是' if recall_at_3 == 1.0 else '否'}, Top3章节: {', '.join(top3_chapters) if top3_chapters else '无'}")
        logger.info(f"Recall@5: {'是' if recall_at_5 == 1.0 else '否'}, Top5章节: {', '.join(top5_chapters) if top5_chapters else '无'}")
        logger.info(f"Recall@10: {'是' if recall_at_10 == 1.0 else '否'}, Top10章节: {', '.join(top10_chapters) if top10_chapters else '无'}")
        
        metrics = {
            "test_index": test_index,  # 保存测试索引，用于排序
            "question": test_case.question,
            # 将最终选择的章节作为 answer，便于在 HTML 中展示（RAG返回的第一个结果）
            "answer": final_chapter if final_chapter else "",
            "reference": test_case.reference,
            "type": test_case.type,
            "theme": test_case.theme,
            "retrieved_count": len(validated_chapters),  # 同时满足theme和章节匹配的结果数量
            "latency": latency,
            "accuracy": binary_accuracy,  # RAG第一结果的章节是否匹配reference（用于评估算法准确度）
            "recall": accuracy_metrics['recall'],  # 基于topK个结果
            "correct_count": accuracy_metrics['correct_count'],
            "recall@3": recall_at_3,  # 基于topK个结果
            "recall@5": recall_at_5,  # 基于topK个结果
            "recall@10": recall_at_10,  # 基于topK个结果
            # RAG第一结果的验证状态
            "top1_theme_match": top1_theme_match,
            "top1_chapter_match": top1_chapter_match,
            "top1_both_match": top1_theme_match and top1_chapter_match,
            # 保存topk章节信息，用于异常分析
            "top3_chapters": top3_chapters,
            "top5_chapters": top5_chapters,
            "top10_chapters": top10_chapters,
        }
        
        return metrics
    
    def _save_anomalies(self, anomalies: List[Dict[str, Any]], output_path: Optional[str] = None):
        """保存异常情况到文件"""
        logger.info('-'*80)
        if output_path is None:
            output_path = str(output_dir / "anomalies.json")
        try:
            # 将anomalies保存为JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(anomalies, f, ensure_ascii=False, indent=2)
            logger.info(f"异常情况已保存到: {output_path}")
            
            # 同时保存为CSV文件，便于查看
            if anomalies:
                csv_path = output_path.replace('.json', '.csv')
                # 将列表转换为字符串，便于CSV保存
                anomalies_for_csv = []
                for anomaly in anomalies:
                    anomaly_copy = anomaly.copy()
                    for col in ['top3_chapters', 'top5_chapters', 'top10_chapters']:
                        if col in anomaly_copy:
                            if isinstance(anomaly_copy[col], list):
                                anomaly_copy[col] = ', '.join(anomaly_copy[col])
                            elif not anomaly_copy[col]:
                                anomaly_copy[col] = ''
                    anomalies_for_csv.append(anomaly_copy)
                
                anomalies_df = pd.DataFrame(anomalies_for_csv)
                anomalies_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                logger.info(f"异常情况CSV已保存到: {csv_path}")
        except Exception as e:
            logger.error(f"保存异常情况失败: {str(e)}")
    
    def run_batch_evaluation(self, test_cases: List[TestCase], max_workers: int = 1, delay_between_requests: float = 0.5) -> pd.DataFrame:
        """
        批量运行评测
        
        参数:
            test_cases: 测试用例列表
            max_workers: 并发线程数，默认为1（顺序执行）。设置为大于1的值启用并发执行
            delay_between_requests: 每个请求之间的延迟（秒），用于避免API限流，默认0.5秒
                                  注意：并发模式下，延迟会在每个线程中独立应用
        
        返回:
            pd.DataFrame: 评测结果数据框
        """
        total_cases = len(test_cases)
        logger.info(f"开始批量评测，共 {total_cases} 条用例")
        logger.info(f"API响应将保存到目录: {self.api_responses_dir.absolute()}")
        
        if max_workers > 1:
            logger.info(f"使用并发模式，并发线程数: {max_workers}")
        else:
            logger.info("使用顺序执行模式")
        
        anomalies = []  # 记录异常情况
        anomalies_lock = Lock()  # 异常列表的线程安全锁
        completed_count = 0  # 已完成数量
        completed_lock = Lock()  # 完成计数的线程安全锁
        
        def run_single_test_with_index(args):
            """包装函数，用于并发执行"""
            idx, test_case = args
            try:
                # 线程安全的进度日志
                with self.logger_lock:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"进度: {idx}/{total_cases}")
                    logger.info(f"{'='*80}\n")
                
                # 执行测试
                result = self.run_single_test(test_case, test_index=idx)
                
                # 线程安全地添加结果
                with self.results_lock:
                    self.results.append(result)
                
                # 检查是否为异常情况
                if result.get('accuracy', 1.0) == 0.0 or result.get('recall', 1.0) == 0.0:
                    anomaly = {
                        "test_index": idx,
                        "question": result.get('question', ''),
                        "reference": result.get('reference', ''),
                        "theme": result.get('theme', ''),
                        "type": result.get('type', ''),
                        "answer": result.get('answer', ''),
                        "accuracy": result.get('accuracy', 0.0),
                        "recall": result.get('recall', 0.0),
                        "recall@3": result.get('recall@3', 0.0),
                        "recall@5": result.get('recall@5', 0.0),
                        "recall@10": result.get('recall@10', 0.0),
                        "top3_chapters": result.get('top3_chapters', []),
                        "top5_chapters": result.get('top5_chapters', []),
                        "top10_chapters": result.get('top10_chapters', []),
                    }
                    with anomalies_lock:
                        anomalies.append(anomaly)
                    with self.logger_lock:
                        logger.warning(f"发现异常情况 (Test {idx}): 准确率={result.get('accuracy', 0.0)}, 召回率={result.get('recall', 0.0)}")
                
                # 更新完成计数
                with completed_lock:
                    nonlocal completed_count
                    completed_count += 1
                    if completed_count % 10 == 0 or completed_count == total_cases:
                        logger.info(f"已完成 {completed_count}/{total_cases} 个测试用例 ({completed_count/total_cases*100:.1f}%)")
                
                # 延迟以避免API限流（每个线程独立延迟）
                if delay_between_requests > 0:
                    time.sleep(delay_between_requests)
                
                return result
            except Exception as e:
                with self.logger_lock:
                    logger.error(f"测试用例 {idx} 执行失败: {str(e)}")
                # 返回错误结果
                error_result = {
                    "test_index": idx,
                    "question": test_case.question if test_case else "",
                    "answer": "",
                    "reference": test_case.reference if test_case else "",
                    "type": test_case.type if test_case else None,
                    "theme": test_case.theme if test_case else None,
                    "error": str(e),
                    "latency": 0.0,
                    "accuracy": 0.0,
                    "recall": 0.0,
                }
                with self.results_lock:
                    self.results.append(error_result)
                with completed_lock:
                    completed_count += 1
                return error_result
        
        # 根据 max_workers 决定使用并发还是顺序执行
        if max_workers > 1:
            # 并发执行
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_index = {
                    executor.submit(run_single_test_with_index, (idx, test_case)): idx
                    for idx, test_case in enumerate(test_cases, 1)
                }
                
                # 等待所有任务完成（可选：可以在这里添加进度显示）
                for future in as_completed(future_to_index):
                    try:
                        future.result()  # 获取结果，如果有异常会在这里抛出
                    except Exception as e:
                        idx = future_to_index[future]
                        with self.logger_lock:
                            logger.error(f"任务 {idx} 执行异常: {str(e)}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"并发执行完成，总耗时: {elapsed_time:.2f}秒")
        else:
            # 顺序执行（原有逻辑）
            for idx, test_case in enumerate(test_cases, 1):
                run_single_test_with_index((idx, test_case))
        
        # 按 test_index 排序结果，确保顺序一致
        self.results.sort(key=lambda x: x.get('test_index', 0))
        
        # 保存异常情况
        if anomalies:
            self._save_anomalies(anomalies)
            logger.info(f"共发现 {len(anomalies)} 个异常情况")
        else:
            logger.info("未发现异常情况")
        
        df = pd.DataFrame(self.results)
        
        # 将topk_chapters列表转换为字符串，便于保存到CSV
        for col in ['top3_chapters', 'top5_chapters', 'top10_chapters']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x) if x else '')
        
        return df
    
    def generate_report(self, df: pd.DataFrame, output_path: Optional[str] = None):
        """生成评测报告（使用仪表盘）"""
        from evaluation_dashboard import EvaluationDashboard
        
        if output_path is None:
            output_path = str(output_dir / "evaluation_dashboard.html")
        
        # 计算latency统计信息（平均值和总和）
        latency_stats = {}
        if 'latency' in df.columns:
            latency_stats['latency_avg'] = df['latency'].mean()
            latency_stats['latency_total'] = df['latency'].sum()
        
        # 使用仪表盘生成器
        dashboard = EvaluationDashboard(df, latency_stats)
        summary = dashboard.generate(output_path)
        
        logger.info(f"报告已生成: {output_path}")
        logger.info(f"Latency统计: 平均值={latency_stats.get('latency_avg', 0):.3f}s, 总和={latency_stats.get('latency_total', 0):.3f}s")
        return summary


class ABTestComparator:
    """A/B测试对比器"""
    
    def __init__(self, client_a: RagFlowClient, client_b: RagFlowClient):
        self.client_a = client_a
        self.client_b = client_b
    
    def run_ab_test(self, test_cases: List[TestCase], 
                    config_a: RetrievalConfig, 
                    config_b: RetrievalConfig,
                    max_workers: int = 1,
                    delay_between_requests: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """运行A/B对比测试
        返回: (comparison_df, df_a, df_b) - 对比结果、A组结果、B组结果
        
        参数:
            test_cases: 测试用例列表
            config_a: A组配置
            config_b: B组配置
            max_workers: 并发线程数，默认为1（顺序执行）
            delay_between_requests: 每个请求之间的延迟（秒），默认0.5秒
        """
        logger.info("开始A/B对比测试")
        
        runner_a = EvaluationRunner(self.client_a, config_a)
        runner_b = EvaluationRunner(self.client_b, config_b)
        
        df_a = runner_a.run_batch_evaluation(test_cases, max_workers=max_workers, delay_between_requests=delay_between_requests)
        df_b = runner_b.run_batch_evaluation(test_cases, max_workers=max_workers, delay_between_requests=delay_between_requests)
        
        # 生成对比报告
        comparison = self.generate_comparison(df_a, df_b)
        return comparison, df_a, df_b
    
    def generate_comparison(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
        """生成对比结果"""
        metric_columns = [col for col in df_a.columns if '@' in col or col in ['latency', 'accuracy', 'recall']]
        
        comparison_data = []
        for metric in metric_columns:
            comparison_data.append({
                'Metric': metric,
                'Plan A': df_a[metric].mean(),
                'Plan B': df_b[metric].mean(),
                'Improvement': ((df_b[metric].mean() - df_a[metric].mean()) / df_a[metric].mean() * 100)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 可视化对比
        self.plot_comparison(comparison_df)
        
        return comparison_df
    
    def plot_comparison(self, comparison_df: pd.DataFrame):
        """绘制对比图表"""
        import matplotlib.pyplot as plt  # 延迟导入，仅在需要时加载
        
        metrics = comparison_df['Metric']
        x = range(len(metrics))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], comparison_df['Plan A'], width, label='Plan A')
        plt.bar([i + width/2 for i in x], comparison_df['Plan B'], width, label='Plan B')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('A/B Test Comparison')
        plt.xticks(x, metrics, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        output_path = output_dir / 'ab_comparison.png'
        plt.savefig(output_path)
        logger.info(f"对比图表已保存: {output_path}")


# 辅助函数：查询数据集和文档ID
def find_dataset_and_document_ids(client: RagFlowClient) -> Dict[str, List[str]]:
    """
    查询并显示所有可用的数据集ID和文档ID，并返回字典
    返回格式: {dataset_id: [document_id1, document_id2, ...]}
    使用方法：
        client = RagFlowClient(api_url="...", api_key="...")
        dataset_doc_dict = find_dataset_and_document_ids(client)
    """
    print("=" * 60)
    print("查询数据集列表...")
    print("=" * 60)
    
    # 创建字典存储 dataset_id 和对应的 document_id 列表
    #dataset_document_dict: Dict[str, List[str]] = {}
    dataset_document_dict: Dict[str, Dict[str, str]] = {}

    # 查询所有数据集
    datasets_result = client.list_datasets()
    
    if "error" in datasets_result:
        print(f"[ERROR] 查询失败: {datasets_result['error']}")
        print("\n提示：请检查API地址和API Key是否正确")
        return dataset_document_dict
    
    # 处理不同的响应格式
    datasets = datasets_result.get("data", [])
   
    if not datasets and isinstance(datasets_result, list):
        datasets = datasets_result
    
    if not datasets:
        print("[WARN] 未找到任何数据集")
        return dataset_document_dict
    
    print(f"\n找到 {len(datasets)} 个数据集:\n")
    
    for idx, dataset in enumerate(datasets, 1):
        # 处理不同的响应格式
        dataset_id = dataset.get("id") or dataset.get("dataset_id") or dataset.get("_id")
        dataset_name = dataset.get("name") or dataset.get("dataset_name") or "未命名"
        
        print(f"{idx}. 数据集名称: {dataset_name}")
        print(f"   数据集ID: {dataset_id}")
        
        # 初始化该数据集的文档列表
        documents_dict = {}
        
        # 查询该数据集下的文档
        print(f"   正在查询文档列表...")
        documents_result = client.list_documents(dataset_id)
        
        if "error" not in documents_result:
            documents = documents_result.get("data", []).get('docs', [])
            
            if documents:
                print(f"   找到 {len(documents)} 个文档:")
                for doc_idx, doc in enumerate(documents, 1):
                    doc_id = doc.get('id')
                    doc_name = doc.get('name') 
                    print(f"      {doc_idx}. {doc_name} (ID: {doc_id})")
                    if doc_id:
                        documents_dict[doc_name] = doc_id
            else:
                print(f"   [WARN] 该数据集下没有文档")
        else:
            print(f"   [WARN] 查询文档失败: {documents_result.get('error', '未知错误')}")
        
        # 将 dataset_id 和对应的 document_ids 存入字典
        if dataset_name and dataset_id:
            dataset_document_dict[dataset_name] = {'id': dataset_id, 'documents': documents_dict}
        
        print()
    
    print("=" * 60)
    print(f"总计: {len(dataset_document_dict)} 个数据集")
    total_docs = sum(len(v.get('documents', {})) for v in dataset_document_dict.values())
    print(f"总计: {total_docs} 个文档")
    print("=" * 60)
    
    return dataset_document_dict


if __name__ == "__main__":
    # 配置RagFlow客户端
    # 请注意：不同环境有不同的api_url和api_key，需要根据实际情况选择使用哪个环境

    # 从配置文件加载 API 配置
    try:
        import config
        env = getattr(config, 'DEFAULT_ENV', 'test').lower()
        
        if env == 'prod':
            logger.info("================================================")
            logger.info("使用生产环境")
            logger.info("================================================")
            api_url = getattr(config, 'PROD_API_URL', '')
            api_key = getattr(config, 'PROD_API_KEY', '')
        else:
            logger.info("================================================")
            logger.info("使用测试环境")
            logger.info("================================================")
            api_url = getattr(config, 'TEST_API_URL', '')
            api_key = getattr(config, 'TEST_API_KEY', '')
        
        if not api_url or not api_key:
            raise ValueError("配置文件中的 API URL 或 API Key 为空")
        
        client = RagFlowClient(api_url=api_url, api_key=api_key)
        
    except ImportError:
        logger.error("=" * 80)
        logger.error("错误: 找不到 config.py 配置文件")
        logger.error("=" * 80)
        logger.error("请按照以下步骤操作:")
        logger.error("1. 复制 config.py.example 为 config.py")
        logger.error("2. 在 config.py 中填入你的 API 配置信息")
        logger.error("3. config.py 已被添加到 .gitignore，不会被提交到 GitHub")
        logger.error("=" * 80)
        sys.exit(1)
    except Exception as e:
        logger.error(f"加载配置失败: {str(e)}")
        sys.exit(1)
    
    # 如果 datasets.json 不存在，自动生成
    import os
    import json
    dataset_id_json_path = output_dir / "datasets.json"
    if not dataset_id_json_path.exists():
        print("datasets.json 不存在，正在生成...")
        dataset_document_dict = find_dataset_and_document_ids(client)
        with open(dataset_id_json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_document_dict, f, ensure_ascii=False, indent=4)
        print(f"已生成并保存数据集和文档ID字典到 {dataset_id_json_path}。")
    else:
        print(f"已找到 {dataset_id_json_path}，search 方法将自动加载。")
    
    # 加载测试集（Excel格式）
    test_set_path = input_dir / "test.xlsx"
    if not test_set_path.exists():
        logger.error(f"测试数据文件不存在: {test_set_path}")
        print(f"错误: 测试数据文件不存在，请将 test.xlsx 放到 {input_dir} 目录下")
        sys.exit(1)
    
    # 创建临时runner来加载测试集
    temp_runner = EvaluationRunner(client, RetrievalConfig(dataset_ids=[]))
    test_cases = temp_runner.load_test_set(str(test_set_path))
    
    # # 限制测试数据为10条
    # if len(test_cases) > 10:
    #     test_cases = test_cases[:10]
    #     logger.info(f"测试数据已限制为前10条")
    
    # ========== 并发配置 ==========
    # 从配置文件读取并发参数，如果没有则使用默认值
    max_workers = getattr(config, 'MAX_WORKERS', 1)  # 默认顺序执行
    delay_between_requests = getattr(config, 'DELAY_BETWEEN_REQUESTS', 0.5)  # 默认0.5秒延迟
    
    if max_workers > 1:
        logger.info(f"并发模式已启用: {max_workers} 个并发线程")
        logger.info(f"请求延迟: {delay_between_requests} 秒")
    else:
        logger.info("使用顺序执行模式（单线程）")
    
    # ========== 判断是否启用A/B测试 ==========
    enable_ab_test = getattr(config, 'ENABLE_AB_TEST', False)  # 默认不启用A/B测试
    
    if enable_ab_test:
        # ========== A/B测试模式 ==========
        print(f"\n将使用 {len(test_cases)} 条测试数据进行A/B测试")
        
        # 配置A: vector_similarity_weight = 0.3 (默认值)
        logger.info("=" * 80)
        logger.info("配置 A: vector_similarity_weight = 0.3 (默认值)")
        logger.info("=" * 80)
        config_a = RetrievalConfig(
            dataset_ids=[],  # 留空，search 方法会自动加载
            document_ids=[],  # 留空，search 方法会自动加载
            top_k=1024,  # API默认值
            similarity_threshold=0.2,  # API默认值
            vector_similarity_weight=0.3,  # A组：默认权重
            rerank_id="",  # API默认值
            highlight=False,  # API默认值
            page=1,  # API默认值
            page_size=30  # API默认值
        )
        
        # 配置B: vector_similarity_weight = 0.7 (更高权重)
        logger.info("=" * 80)
        logger.info("配置 B: vector_similarity_weight = 0.7 (更高权重)")
        logger.info("=" * 80)
        config_b = RetrievalConfig(
            dataset_ids=[],  # 留空，search 方法会自动加载
            document_ids=[],  # 留空，search 方法会自动加载
            top_k=1024,  # API默认值
            similarity_threshold=0.2,  # API默认值
            vector_similarity_weight=0.7,  # B组：更高权重
            rerank_id="",  # API默认值
            highlight=False,  # API默认值
            page=1,  # API默认值
            page_size=30  # API默认值
        )
        
        # 创建A/B测试对比器（使用同一个客户端，只是配置不同）
        ab_comparator = ABTestComparator(client, client)
        
        # 运行A/B测试（支持并发）
        comparison_df, results_df_a, results_df_b = ab_comparator.run_ab_test(
            test_cases, config_a, config_b,
            max_workers=max_workers,
            delay_between_requests=delay_between_requests
        )
        
        # 保存A/B测试对比结果
        comparison_csv_path = output_dir / "ab_test_comparison.csv"
        comparison_df.to_csv(comparison_csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"A/B测试对比结果已保存到: {comparison_csv_path}")
        
        # 生成A组和B组的详细报告
        logger.info("=" * 80)
        logger.info("生成A组详细报告")
        logger.info("=" * 80)
        runner_a = EvaluationRunner(client, config_a)
        summary_a = runner_a.generate_report(results_df_a, str(output_dir / "evaluation_results_a.html"))
        csv_path_a = output_dir / "evaluation_results_a.csv"
        results_df_a.to_csv(csv_path_a, index=False, encoding='utf-8-sig')
        logger.info(f"A组评测结果已保存到: {csv_path_a}")
        
        logger.info("=" * 80)
        logger.info("生成B组详细报告")
        logger.info("=" * 80)
        runner_b = EvaluationRunner(client, config_b)
        summary_b = runner_b.generate_report(results_df_b, str(output_dir / "evaluation_results_b.html"))
        csv_path_b = output_dir / "evaluation_results_b.csv"
        results_df_b.to_csv(csv_path_b, index=False, encoding='utf-8-sig')
        logger.info(f"B组评测结果已保存到: {csv_path_b}")
        
        print("\n" + "=" * 80)
        print("===== A/B测试完成 =====")
        print("=" * 80)
        print("\n对比结果:")
        print(comparison_df.to_string(index=False))
        print("\nA组总体指标:")
        for metric, value in summary_a.items():
            print(f"  {metric}: {value:.4f}")
        print("\nB组总体指标:")
        for metric, value in summary_b.items():
            print(f"  {metric}: {value:.4f}")
        print("\n改进百分比 (B vs A):")
        for _, row in comparison_df.iterrows():
            improvement = row['Improvement']
            print(f"  {row['Metric']}: {improvement:+.2f}%")
    else:
        # ========== 单次评测模式 ==========
        print(f"\n将使用 {len(test_cases)} 条测试数据进行单次评测")
        
        # 从配置文件读取检索配置
        retrieval_config_dict = getattr(config, 'RETRIEVAL_CONFIG', {})
        
        # 创建检索配置
        retrieval_config = RetrievalConfig(
            dataset_ids=[],  # 留空，search 方法会自动加载
            document_ids=[],  # 留空，search 方法会自动加载
            top_k=retrieval_config_dict.get('top_k', 1024),
            similarity_threshold=retrieval_config_dict.get('similarity_threshold', 0.2),
            vector_similarity_weight=retrieval_config_dict.get('vector_similarity_weight', 0.3),
            rerank_id=retrieval_config_dict.get('rerank_id', ''),
            highlight=retrieval_config_dict.get('highlight', False),
            page=retrieval_config_dict.get('page', 1),
            page_size=retrieval_config_dict.get('page_size', 30)
        )
        
        logger.info("=" * 80)
        logger.info("检索配置:")
        logger.info(f"  top_k: {retrieval_config.top_k}")
        logger.info(f"  similarity_threshold: {retrieval_config.similarity_threshold}")
        logger.info(f"  vector_similarity_weight: {retrieval_config.vector_similarity_weight}")
        logger.info("=" * 80)
        
        # 创建评测运行器
        runner = EvaluationRunner(client, retrieval_config)
        
        # 运行批量评测（支持并发）
        results_df = runner.run_batch_evaluation(
            test_cases,
            max_workers=max_workers,
            delay_between_requests=delay_between_requests
        )
        
        # 生成报告
        logger.info("=" * 80)
        logger.info("生成评测报告")
        logger.info("=" * 80)
        summary = runner.generate_report(results_df)
        
        # 保存结果
        csv_path = output_dir / "evaluation_results.csv"
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"评测结果已保存到: {csv_path}")
        
        print("\n" + "=" * 80)
        print("===== 评测完成 =====")
        print("=" * 80)
        print("\n总体指标:")
        for metric, value in summary.items():
            print(f"  {metric}: {value:.4f}")
