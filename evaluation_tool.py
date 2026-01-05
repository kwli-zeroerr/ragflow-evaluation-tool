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
from threading import Lock, Event

# 导入规则系统
try:
    from core.rules_manager import RuleManager
    from core.chapter_mapper import ChapterMapper
    from core.manual_mapper import ManualMapper
    from core.format_extractor import FormatExtractor
    from core.chapter_identifier_extractor import ChapterIdentifierExtractor
    from core.chapter_identifier_matcher import ChapterIdentifierMatcher
    RULES_AVAILABLE = True
except ImportError:
    RULES_AVAILABLE = False
    # logger 还未定义，使用 print 或稍后记录
    import sys
    print("警告: 规则系统模块未找到，将使用默认硬编码逻辑", file=sys.stderr)

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

# 配置文件处理器和控制台处理器
file_handler = logging.FileHandler(
    logs_dir / f'evaluation_tool_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    encoding='utf-8'
)
stream_handler = SafeStreamHandler(sys.stdout)

# 设置格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# 配置根logger（使用简单的日志配置，提升性能）
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler]
)

# 获取logger
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
    
    def __init__(self, api_url: str, api_key: str, knowledge_id: Optional[str] = None, 
                 use_ragflow_format: bool = False, use_ragflow_index: bool = False, 
                 ragflow_timeout: int = 30):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.knowledge_id = knowledge_id  # OpenWebUI 知识库ID
        self.use_ragflow_format = use_ragflow_format  # 是否使用 RAGflow 格式
        self.use_ragflow_index = use_ragflow_index  # 是否使用 RAGFlow ES 索引
        self.ragflow_timeout = ragflow_timeout  # OpenWebUI 接口超时时间
        self.is_openwebui = knowledge_id is not None  # 是否为 OpenWebUI 接口
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
        
        logger.info("")
        logger.info("--- 数据集加载 ---")
        logger.info(f"从本地数据集加载: {len(dataset_ids)} 个数据集, {len(document_ids)} 个文档（已缓存）")
        logger.info("")
        return dataset_ids, document_ids
    
    def _build_common_params(self, question: str, config: RetrievalConfig) -> Dict[str, Any]:
        """构建共同的请求参数"""
        params = {
            "top_k": config.top_k,
        }
        
        # 添加向量相似度权重
        if config.vector_similarity_weight is not None:
            params["vector_similarity_weight"] = config.vector_similarity_weight
        
        # 添加重排序ID
        if config.rerank_id is not None:
            params["rerank_id"] = config.rerank_id
        
        return params
    
    def _build_standard_payload(self, question: str, config: RetrievalConfig) -> Dict[str, Any]:
        """构建标准 RagFlow API 的请求负载"""
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
        
        return payload
    
    def _build_openwebui_params(self, question: str, config: RetrievalConfig) -> Dict[str, Any]:
        """构建 OpenWebUI RAGFlow API 的请求参数"""
        params = self._build_common_params(question, config)
        
        # OpenWebUI 使用 "query" 而不是 "question"
        params["query"] = question
        
        # 添加 RAGFlow 相关参数
        if self.use_ragflow_format:
            params["ragflow_format"] = True
        
        if self.use_ragflow_index:
            params["use_ragflow_index"] = True
        
        return params
    
    def _get_retry_config(self) -> Tuple[int, float, bool]:
        """获取重试配置"""
        try:
            import config as config_module
            max_retries = getattr(config_module, 'MAX_RETRIES', 3)
            retry_delay = getattr(config_module, 'RETRY_DELAY', 1.0)
            use_exponential_backoff = getattr(config_module, 'USE_EXPONENTIAL_BACKOFF', True)
        except (ImportError, AttributeError):
            max_retries = 3
            retry_delay = 1.0
            use_exponential_backoff = True
        return max_retries, retry_delay, use_exponential_backoff
    
    def _execute_with_retry(self, request_func, question: str, max_retries: int, 
                           retry_delay: float, use_exponential_backoff: bool) -> Dict[str, Any]:
        """执行请求并处理重试逻辑（通用方法）"""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return request_func()
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max_retries:
                    # 计算重试延迟（指数退避或固定延迟）
                    if use_exponential_backoff:
                        delay = retry_delay * (2 ** attempt)
                    else:
                        delay = retry_delay
                    
                    logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {question[:50]}... - {str(e)}，{delay:.2f}秒后重试")
                    time.sleep(delay)
                else:
                    # 最后一次尝试也失败了
                    logger.error(f"API调用失败 (已重试 {max_retries} 次): {question[:50]}... - {str(e)}")
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    if use_exponential_backoff:
                        delay = retry_delay * (2 ** attempt)
                    else:
                        delay = retry_delay
                    logger.warning(f"API调用异常 (尝试 {attempt + 1}/{max_retries + 1}): {question[:50]}... - {str(e)}，{delay:.2f}秒后重试")
                    time.sleep(delay)
                else:
                    logger.error(f"API调用异常 (已重试 {max_retries} 次): {question[:50]}... - {str(e)}")
        
        # 所有重试都失败了
        return {"error": str(last_error), "data": {"chunks": []}}
    
    def search(self, question: str, theme: str, config: RetrievalConfig) -> Dict[str, Any]:
        """调用RagFlow检索API - 支持标准RagFlow API和OpenWebUI RAGFlow接口"""
        
        # 如果是 OpenWebUI 接口，使用不同的调用方式
        if self.is_openwebui:
            return self._search_openwebui(question, config)
        
        # 标准 RagFlow API
        endpoint = f"{self.api_url}/api/v1/retrieval"
        payload = self._build_standard_payload(question, config)
        max_retries, retry_delay, use_exponential_backoff = self._get_retry_config()
        
        def request_func():
            response = requests.post(endpoint, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        
        return self._execute_with_retry(request_func, question, max_retries, retry_delay, use_exponential_backoff)
    
    def _search_openwebui(self, question: str, config: RetrievalConfig) -> Dict[str, Any]:
        """调用 OpenWebUI RAGFlow 接口"""
        params = self._build_openwebui_params(question, config)
        max_retries, retry_delay, use_exponential_backoff = self._get_retry_config()
        
        # 优先使用 GET 方法
        get_endpoint = f"{self.api_url}/api/v1/knowledge/{self.knowledge_id}/vector/search-test"
        
        def request_func_get():
            response = requests.get(
                get_endpoint,
                params=params,
                headers=self.headers,
                timeout=self.ragflow_timeout
            )
            if response.status_code == 200:
                result = response.json()
                return self._convert_openwebui_response(result)
            else:
                raise requests.exceptions.HTTPError(f"HTTP {response.status_code}: {response.text[:200]}")
        
        # 尝试 GET 方法
        try:
            result = self._execute_with_retry(request_func_get, question, max_retries, retry_delay, use_exponential_backoff)
            if "error" not in result:
                return result
        except Exception:
            pass
        
        # GET 失败，尝试 POST 方法
        return self._search_openwebui_post(question, params)
    
    def _search_openwebui_post(self, question: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """使用 POST 方法调用 OpenWebUI RAGFlow 接口（备用方法）"""
        post_endpoint = f"{self.api_url}/api/v1/knowledge/{self.knowledge_id}/vector/search"
        
        # POST 方法需要 highlight 参数
        payload = params.copy()
        payload["highlight"] = False
        
        def request_func_post():
            response = requests.post(
                post_endpoint,
                headers=self.headers,
                json=payload,
                timeout=self.ragflow_timeout
            )
            if response.status_code == 200:
                result = response.json()
                return self._convert_openwebui_response(result)
            else:
                raise requests.exceptions.HTTPError(f"HTTP {response.status_code}: {response.text[:200]}")
        
        max_retries, retry_delay, use_exponential_backoff = self._get_retry_config()
        return self._execute_with_retry(request_func_post, question, max_retries, retry_delay, use_exponential_backoff)
    
    def _convert_openwebui_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """将 OpenWebUI 响应格式转换为标准 RagFlow 格式"""
        if "error" in result:
            return result
        
        # OpenWebUI 格式：chunks 在顶层或 data.chunks
        chunks = result.get("chunks", [])
        if not chunks:
            data = result.get("data", {})
            chunks = data.get("chunks", [])
        
        # 转换 chunks 格式
        converted_chunks = []
        for chunk in chunks:
            # OpenWebUI 格式字段映射到标准格式
            # important_kwd 是 OpenWebUI 的字段名，important_keywords 是标准格式字段名
            important_kwd = chunk.get("important_kwd", chunk.get("important_keywords", []))
            
            # docnm_kwd 通常是手册名称，应该添加到 important_keywords 的开头
            docnm_kwd = chunk.get("docnm_kwd", "")
            
            # 构建 important_keywords 列表：[手册名称, 大章, 小章]
            important_keywords = []
            if docnm_kwd:
                important_keywords.append(docnm_kwd)
            if important_kwd:
                if isinstance(important_kwd, list):
                    important_keywords.extend(important_kwd)
                else:
                    important_keywords.append(important_kwd)
            
            converted_chunk = {
                "chunk_id": chunk.get("chunk_id", ""),
                "similarity": chunk.get("similarity", chunk.get("score", 0.0)),
                "content": chunk.get("content", chunk.get("content_with_weight", chunk.get("text", ""))),
                "important_keywords": important_keywords,
            }
            
            # 添加其他可能存在的字段
            if "vector_similarity" in chunk:
                converted_chunk["vector_similarity"] = chunk["vector_similarity"]
            if "term_similarity" in chunk:
                converted_chunk["term_similarity"] = chunk["term_similarity"]
            
            converted_chunks.append(converted_chunk)
        
        # 返回标准格式
        return {
            "data": {
                "chunks": converted_chunks
            },
            "total": result.get("total", len(converted_chunks))
        }


class ChapterMatcher:
    """章节匹配器 - 判断章节层级关系"""
    
    # 类级别的规则管理器（单例模式，延迟初始化）
    _rule_manager: Optional['RuleManager'] = None
    _chapter_mapper: Optional['ChapterMapper'] = None
    _identifier_matcher: Optional['ChapterIdentifierMatcher'] = None
    
    # 默认中文数字映射（降级方案）
    _DEFAULT_CHINESE_NUM_MAP = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
        '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
        '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25,
        '二十六': 26, '二十七': 27, '二十八': 28, '二十九': 29, '三十': 30,
        '百': 100, '千': 1000, '万': 10000
    }
    
    # 保持向后兼容的类属性
    CHINESE_NUM_MAP = _DEFAULT_CHINESE_NUM_MAP
    
    @classmethod
    def _get_rule_manager(cls) -> Optional['RuleManager']:
        """获取规则管理器（延迟初始化）"""
        if not RULES_AVAILABLE:
            return None
        
        if cls._rule_manager is None:
            try:
                cls._rule_manager = RuleManager()
                # 更新 CHINESE_NUM_MAP 以保持向后兼容
                chinese_map = cls._rule_manager.get_chinese_num_map()
                if chinese_map:
                    cls.CHINESE_NUM_MAP = chinese_map
            except Exception as e:
                logger.warning(f"初始化规则管理器失败: {str(e)}，使用默认逻辑")
                return None
        
        return cls._rule_manager
    
    @classmethod
    def _get_chapter_mapper(cls) -> Optional['ChapterMapper']:
        """获取章节映射器（延迟初始化）"""
        if not RULES_AVAILABLE:
            return None
        
        if cls._chapter_mapper is None:
            rule_manager = cls._get_rule_manager()
            if rule_manager:
                try:
                    cls._chapter_mapper = ChapterMapper(rule_manager)
                except Exception as e:
                    logger.warning(f"初始化章节映射器失败: {str(e)}")
                    return None
        
        return cls._chapter_mapper
    
    @classmethod
    def _get_identifier_matcher(cls) -> Optional['ChapterIdentifierMatcher']:
        """获取标识符匹配器（延迟初始化）"""
        if not RULES_AVAILABLE:
            return None
        
        if cls._identifier_matcher is None:
            rule_manager = cls._get_rule_manager()
            if rule_manager:
                try:
                    identifier_mappings = rule_manager.get_identifier_mapping()
                    cls._identifier_matcher = ChapterIdentifierMatcher(identifier_mappings)
                except Exception as e:
                    logger.warning(f"初始化标识符匹配器失败: {str(e)}")
                    return None
        
        return cls._identifier_matcher
    
    @classmethod
    def _get_chinese_num_map(cls) -> Dict[str, int]:
        """获取中文数字映射表（从规则或默认值）"""
        rule_manager = cls._get_rule_manager()
        if rule_manager:
            chinese_map = rule_manager.get_chinese_num_map()
            if chinese_map:
                return chinese_map
        return cls._DEFAULT_CHINESE_NUM_MAP
    
    @staticmethod
    def remove_english_text(text: str) -> str:
        """移除文本中的英文部分，只保留中文部分
        例如："第一章 储存与使用 Chapter 1 Storage and Handling" -> "第一章 储存与使用"
        """
        if not text:
            return text
        
        # 从规则管理器获取是否启用移除英文文本的配置
        rule_manager = ChapterMatcher._get_rule_manager()
        if rule_manager:
            remove_english = rule_manager.get_matching_rule("normalization_rules.remove_english_text", True)
            if not remove_english:
                return text.strip()
        
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
        
        # 从规则管理器获取中文数字映射表
        chinese_map = ChapterMatcher._get_chinese_num_map()
        
        # 处理简单的中文数字（1-99）
        if chinese_num in chinese_map:
            return chinese_map[chinese_num]
        
        # 处理"二十"、"三十"等十的倍数
        if chinese_num.endswith('十'):
            base = chinese_num[:-1]
            if base == '':
                return 10
            if base in chinese_map:
                return chinese_map[base] * 10
        
        # 处理"二十一"、"二十七"等十以上的数字
        if len(chinese_num) >= 2 and '十' in chinese_num:
            parts = chinese_num.split('十')
            if len(parts) == 2:
                if parts[0] == '':
                    tens = 1
                else:
                    tens = chinese_map.get(parts[0], 0)
                ones = chinese_map.get(parts[1], 0)
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
        
        # 从规则管理器获取是否移除末尾点的配置
        rule_manager = ChapterMatcher._get_rule_manager()
        remove_trailing_dots = True
        if rule_manager:
            remove_trailing_dots = rule_manager.get_matching_rule("normalization_rules.remove_trailing_dots", True)
        
        # 移除末尾的点（如 "1.1." -> "1.1"）
        if remove_trailing_dots:
            chapter = chapter.rstrip('.')
        else:
            chapter = chapter
        
        # 从规则管理器获取提取模式
        rule_manager = ChapterMatcher._get_rule_manager()
        if rule_manager:
            numeric_pattern = rule_manager.get_extraction_rule("patterns.numeric", r"^\d+(?:\.\d+)*")
            chinese_chapter_pattern = rule_manager.get_extraction_rule("patterns.chinese_chapter", r"第[一二三四五六七八九十\d]+章[^第]*")
            chinese_section_pattern = rule_manager.get_extraction_rule("patterns.chinese_section", r"第[一二三四五六七八九十\d]+节")
        else:
            numeric_pattern = r'^(\d+(?:\.\d+)*)'
            chinese_chapter_pattern = r'第[一二三四五六七八九十\d]+章[^第]*'
            chinese_section_pattern = r'第[一二三四五六七八九十\d]+节'
        
        # 提取数字格式的章节编号（如 "1.1"、"1.2"、"10.1"）
        # 匹配模式：数字.数字.数字...（可能后面有空格和文字）
        pattern = numeric_pattern if numeric_pattern.startswith('^') else f'^({numeric_pattern})'
        match = re.match(pattern, chapter)
        if match:
            return match.group(1) if match.groups() else match.group(0)
        
        # 如果没有匹配到数字格式，尝试匹配中文格式
        patterns = [
            chinese_chapter_pattern,  # 匹配"第一章"或"第1章"
            r'第[一二三四五六七八九十\d]+章第[一二三四五六七八九十\d]+节',  # 匹配"第一章第一节"
            chinese_section_pattern,  # 匹配"第一节"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, chapter)
            if match:
                matched = match.group(0).rstrip('.')
                # 再次移除英文部分
                matched = ChapterMatcher.remove_english_text(matched)
                return matched
        
        # 处理"X、 标题"格式（如"四、 关节正转方向"）
        # 这个格式在extract_chapter_info中已经处理，但normalize_chapter中也需要处理
        chinese_num_with_punctuation = re.match(r'^([一二三四五六七八九十百千万\d]+)[、，。]', chapter)
        if chinese_num_with_punctuation:
            matched = chinese_num_with_punctuation.group(1)
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
        
        # 排除版本号模式（如 V3.40, 3.40 在文本末尾）
        # 版本号通常是：V数字.数字 或 数字.数字 在文本末尾，且数字较大（通常 >= 1.0）
        version_pattern = r'[Vv]?\d+\.\d+$'
        if re.search(version_pattern, text_str):
            # 检查是否是版本号（数字部分通常 >= 1.0）
            version_match = re.search(r'(\d+)\.(\d+)$', text_str)
            if version_match:
                major = int(version_match.group(1))
                minor = int(version_match.group(2))
                # 如果第一个数字 >= 1 且第二个数字通常较小（0-99），可能是版本号
                # 但章节号也可能符合这个模式，所以需要更严格的判断
                # 版本号通常在文件名末尾，且前面有下划线或空格
                if major >= 1 and minor <= 99:
                    # 检查前面是否有下划线或特定模式（如 _V3.40）
                    before_version = text_str[:version_match.start()]
                    if re.search(r'[_\s]+[Vv]?\d+\.\d+$', text_str) or before_version.endswith('_V') or before_version.endswith('_v'):
                        # 很可能是版本号，跳过
                        return None
        
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
        - [NEW] 基于对象字典编号和关键词的精确匹配 → 正确
        """
        if not retrieved_chapter or not reference_chapter:
            return False
        
        # 首先尝试基于标识符的精确匹配（优先级最高）
        identifier_matcher = ChapterMatcher._get_identifier_matcher()
        if identifier_matcher:
            try:
                # 提取标识符
                retrieved_identifiers = ChapterIdentifierExtractor.extract_identifiers(retrieved_chapter)
                reference_identifiers = ChapterIdentifierExtractor.extract_identifiers(reference_chapter)
                
                # 尝试基于标识符匹配
                is_match, mapped_chapter = identifier_matcher.match_by_identifier(
                    retrieved_chapter,
                    reference_chapter,
                    retrieved_identifiers,
                    reference_identifiers
                )
                
                if is_match:
                    return True
            except Exception as e:
                logger.warning(f"标识符匹配失败: {str(e)}，继续使用传统匹配方法")
        
        # 应用章节映射（如果配置了映射规则）
        # 确保检索结果和测试集数据都映射到统一格式（新格式）
        chapter_mapper = ChapterMatcher._get_chapter_mapper()
        if chapter_mapper:
            # 检索结果映射到新格式
            retrieved_chapter = chapter_mapper.map_retrieved_chapter(retrieved_chapter, target_format="new")
            # 测试集数据映射到新格式（如果还没有映射）
            reference_chapter = chapter_mapper.map_reference(reference_chapter, target_format="new")
        
        # 标准化章节
        retrieved_normalized = ChapterMatcher.normalize_chapter(retrieved_chapter)
        reference_normalized = ChapterMatcher.normalize_chapter(reference_chapter)
        
        if not retrieved_normalized or not reference_normalized:
            return False
        
        # 完全匹配（忽略末尾的点）
        if retrieved_normalized == reference_normalized:
            return True
        
        # 从规则管理器获取匹配规则
        rule_manager = ChapterMatcher._get_rule_manager()
        allow_parent_match_child = True
        allow_child_match_parent = False
        chinese_arabic_enabled = True
        
        if rule_manager:
            allow_parent_match_child = rule_manager.get_matching_rule(
                "parent_child_rules.allow_parent_match_child", True
            )
            allow_child_match_parent = rule_manager.get_matching_rule(
                "parent_child_rules.allow_child_match_parent", False
            )
            chinese_arabic_enabled = rule_manager.get_matching_rule(
                "chinese_arabic_mapping.enabled", True
            )
        
        # 检查中文数字和阿拉伯数字的对应关系
        # 例如: "六" 对应 "6", "二十七" 对应 "27"
        if chinese_arabic_enabled:
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
        
        # 召回"大章"，标注为"小章" → 正确（如果规则允许）
        # 例如：检索到"1.1"，标注是"1.1.1" → 错误（检索到的是小章）
        # 例如：检索到"1"，标注是"1.1" → 正确（检索到的是大章）
        # 例如：检索到"二十七"，标注是"27.1" → 正确（检索到的是大章）
        if allow_parent_match_child:
            if ChapterMatcher.is_parent_chapter(retrieved_normalized, reference_normalized):
                return True
        
        # 召回"小章"，标注为"大章" → 错误（如果规则允许，则返回True；否则返回False）
        # 例如：检索到"1.1.1"，标注是"1.1" → 错误（默认）或正确（如果规则允许）
        if allow_child_match_parent:
            if ChapterMatcher.is_parent_chapter(reference_normalized, retrieved_normalized):
                return True
        else:
            if ChapterMatcher.is_parent_chapter(reference_normalized, retrieved_normalized):
                return False
        
        # 其他情况（可能是不同章节）→ 错误
        return False
    
    @staticmethod
    def get_mapped_chapter(retrieved_chapter: str, reference_chapter: str) -> Tuple[bool, Optional[str]]:
        """
        获取映射后的章节号（如果identifier匹配成功）
        
        Args:
            retrieved_chapter: 检索到的章节（完整文本或章节编号）
            reference_chapter: 参考章节（完整文本或章节编号）
        
        Returns:
            (是否匹配, 映射后的章节号)
        """
        if not retrieved_chapter or not reference_chapter:
            return (False, None)
        
        identifier_matcher = ChapterMatcher._get_identifier_matcher()
        if identifier_matcher:
            try:
                retrieved_identifiers = ChapterIdentifierExtractor.extract_identifiers(retrieved_chapter)
                reference_identifiers = ChapterIdentifierExtractor.extract_identifiers(reference_chapter)
                is_match, mapped_chapter = identifier_matcher.match_by_identifier(
                    retrieved_chapter,
                    reference_chapter,
                    retrieved_identifiers,
                    reference_identifiers
                )
                if is_match:
                    # 如果mapped_chapter不为None，使用它；否则使用参考章节号
                    if mapped_chapter:
                        return (True, mapped_chapter)
                    else:
                        # 如果没有映射，但匹配成功，提取参考章节的章节号
                        reference_chapter_num = ChapterMatcher.extract_chapter_info(reference_chapter)
                        return (True, reference_chapter_num if reference_chapter_num else None)
            except Exception:
                pass
        return (False, None)


class MetricsCalculator:
    """指标计算器 - 基于章节匹配逻辑"""
    
    @staticmethod
    def calculate_accuracy(retrieved_chapters: List[str], reference_chapter: str) -> Dict[str, float]:
        """
        计算准确率和召回率（兼容旧接口）
        返回: {'correct_count': int, 'total_count': int, 'accuracy': float, 'recall': float}
        """
        return MetricsCalculator.calculate_accuracy_topk(retrieved_chapters, reference_chapter)
    
    @staticmethod
    def calculate_accuracy_top1(retrieved_chapters: List[str], reference_chapter: str) -> Dict[str, float]:
        """
        计算 Top1 准确率和召回率
        返回: {'correct_count': int, 'total_count': int, 'accuracy': float, 'recall': float}
        """
        if not reference_chapter or not retrieved_chapters:
            return {'correct_count': 0, 'total_count': 1, 'accuracy': 0.0, 'recall': 0.0}
        
        # 只检查第一个结果
        top1_chapter = retrieved_chapters[0] if retrieved_chapters else None
        is_match = ChapterMatcher.is_valid_match(top1_chapter, reference_chapter) if top1_chapter else False
        
        return {
            'correct_count': 1 if is_match else 0,
            'total_count': 1,
            'accuracy': 1.0 if is_match else 0.0,
            'recall': 1.0 if is_match else 0.0
        }
    
    @staticmethod
    def calculate_accuracy_topk(retrieved_chapters: List[str], reference_chapter: str, 
                                start_from: int = 1) -> Dict[str, float]:
        """
        计算 TopK 准确率和召回率（从指定位置开始，默认从第2个开始，即忽略Top1）
        
        Args:
            retrieved_chapters: 检索到的章节列表
            reference_chapter: 标注章节
            start_from: 从第几个结果开始计算（1-based，1表示从第一个开始，2表示从第二个开始）
            
        Returns:
            {'correct_count': int, 'total_count': int, 'accuracy': float, 'recall': float}
        """
        if not reference_chapter:
            return {'correct_count': 0, 'total_count': 0, 'accuracy': 0.0, 'recall': 0.0}
        
        # 从指定位置开始（start_from=1 表示从第一个开始，start_from=2 表示从第二个开始）
        # 转换为 0-based 索引
        start_idx = max(0, start_from - 1)
        topk_chapters = retrieved_chapters[start_idx:] if start_idx < len(retrieved_chapters) else []
        
        if not topk_chapters:
            return {'correct_count': 0, 'total_count': 0, 'accuracy': 0.0, 'recall': 0.0}
        
        correct_count = 0
        total_retrieved = len(topk_chapters)
        
        # 检查每个检索结果是否有效匹配
        for retrieved_chapter in topk_chapters:
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
    def recall_at_k(retrieved_chapters: List[str], reference_chapter: str, k: int, 
                    start_from: int = 1) -> float:
        """
        计算Recall@K - 前K个结果中是否至少有一个正确匹配
        
        Args:
            retrieved_chapters: 检索到的章节列表
            reference_chapter: 标注章节
            k: K值（计算前K个结果）
            start_from: 从第几个结果开始计算（1-based，默认从第2个开始，即忽略Top1）
            
        Returns:
            Recall@K 值（0.0 或 1.0）
        """
        if not reference_chapter:
            return 0.0
        
        # 从指定位置开始
        start_idx = max(0, start_from - 1)
        # 取前k个结果（从start_idx开始）
        end_idx = min(start_idx + k, len(retrieved_chapters))
        retrieved_k = retrieved_chapters[start_idx:end_idx]
        
        for chapter in retrieved_k:
            if ChapterMatcher.is_valid_match(chapter, reference_chapter):
                return 1.0
        return 0.0

class EvaluationRunner:
    """评测运行器"""
    
    def __init__(self, client: RagFlowClient, config: RetrievalConfig, output_base_dir: Optional[Path] = None):
        # 初始化规则系统（如果可用）
        self.rule_manager = None
        self.chapter_mapper = None
        self.manual_mapper = None
        self.format_extractor = None
        if RULES_AVAILABLE:
            try:
                self.rule_manager = RuleManager()
                self.chapter_mapper = ChapterMapper(self.rule_manager)
                self.manual_mapper = ManualMapper(self.rule_manager)
                self.format_extractor = FormatExtractor(self.rule_manager)
            except Exception as e:
                logger.warning(f"初始化规则系统失败: {str(e)}，将使用默认逻辑")
        self.client = client
        self.config = config
        self.calculator = MetricsCalculator()
        self.results = []
        # 设置输出基础目录，默认为output_dir
        self.output_base_dir = output_base_dir if output_base_dir is not None else output_dir
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        # 创建API响应保存目录
        self.api_responses_dir = self.output_base_dir / "api_responses"
        self.api_responses_dir.mkdir(parents=True, exist_ok=True)
        # 线程安全锁
        self.results_lock = Lock()
        self.logger_lock = Lock()
        # 记录实际运行时间（从开始到结束的真实时间）
        self.total_runtime: Optional[float] = None
        # 日志详细程度：True=详细（单线程模式），False=简洁（并发模式）
        self.verbose_logging: bool = True
    
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
            mapped_count = 0
            theme_mapped_count = 0
            format_detected_count = 0
            
            # 检测测试集的整体格式（通过采样前几条数据）
            test_set_format = self._detect_test_set_format(df)
            if test_set_format:
                logger.info(f"检测到测试集格式: {test_set_format}")
                format_detected_count = len(df)
            
            for row in df.itertuples(index=False):
                reference = str(row.reference) if pd.notna(row.reference) else ''
                theme = str(row.theme) if hasattr(row, 'theme') and pd.notna(row.theme) else None
                
                # 应用章节映射（如果配置了映射规则）
                # 默认映射到新格式（target_format="new"）
                if self.chapter_mapper and reference:
                    original_reference = reference
                    # 根据测试集格式决定目标格式
                    target_format = "new" if test_set_format != "old" else "new"  # 默认统一映射到新格式
                    reference = self.chapter_mapper.map_reference(reference, target_format=target_format)
                    if reference != original_reference:
                        mapped_count += 1
                        logger.debug(f"章节映射: '{original_reference}' -> '{reference}'")
                
                # 应用手册名称映射（如果配置了映射规则）
                # 默认映射到新格式（target_format="new"）
                if self.manual_mapper and theme:
                    original_theme = theme
                    # 根据测试集格式决定目标格式
                    target_format = "new" if test_set_format != "old" else "new"  # 默认统一映射到新格式
                    theme = self.manual_mapper.map_theme(theme, target_format=target_format)
                    if theme != original_theme:
                        theme_mapped_count += 1
                        logger.debug(f"手册映射: '{original_theme}' -> '{theme}'")
                
                test_cases.append(TestCase(
                    question=str(row.question) if pd.notna(row.question) else '',
                    answer=str(row.answer) if pd.notna(row.answer) else '',
                    reference=reference,
                    type=str(row.type) if hasattr(row, 'type') and pd.notna(row.type) else None,
                    theme=theme
                ))
            
            logger.info(f"从Excel加载测试集: {len(test_cases)} 条用例")
            if format_detected_count > 0:
                logger.info(f"格式识别: {format_detected_count} 条用例已识别格式")
            if mapped_count > 0:
                logger.info(f"已应用章节映射: {mapped_count} 条用例的 reference 字段已映射")
            if theme_mapped_count > 0:
                logger.info(f"已应用手册映射: {theme_mapped_count} 条用例的 theme 字段已映射")
            return test_cases
        except Exception as e:
            logger.error(f"加载测试集失败: {str(e)}")
            raise
    
    def _detect_test_set_format(self, df: pd.DataFrame) -> Optional[str]:
        """
        检测测试集的格式（旧格式/新格式）
        
        Args:
            df: 测试集DataFrame
            
        Returns:
            "old" 或 "new" 或 None（如果无法识别）
        """
        if not self.chapter_mapper or not self.manual_mapper:
            return None
        
        # 采样前10条数据（或全部数据，如果少于10条）
        sample_size = min(10, len(df))
        if sample_size == 0:
            return None
        
        old_format_count = 0
        new_format_count = 0
        
        for idx in range(sample_size):
            row = df.iloc[idx]
            reference = str(row.reference) if pd.notna(row.reference) else ''
            theme = str(row.theme) if hasattr(row, 'theme') and pd.notna(row.theme) else None
            
            # 检测章节格式
            if reference:
                ref_format = self.chapter_mapper.detect_chapter_format(reference)
                if ref_format == "old":
                    old_format_count += 1
                elif ref_format == "new":
                    new_format_count += 1
            
            # 检测手册格式
            if theme:
                theme_format = self.manual_mapper.detect_theme_format(theme)
                if theme_format == "old":
                    old_format_count += 1
                elif theme_format == "new":
                    new_format_count += 1
        
        # 根据多数决定格式
        if old_format_count > new_format_count:
            return "old"
        elif new_format_count > old_format_count:
            return "new"
        else:
            return None  # 无法确定
    
    def _save_api_response(self, test_case: TestCase, response: Dict[str, Any], test_index: int):
        """保存API响应到单独的文件"""
        # 创建安全的文件名（移除特殊字符和控制字符，包括换行符、回车符、制表符等）
        safe_question = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', test_case.question[:50])
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
            # 使用线程安全的方式记录日志
            with self.logger_lock:
                logger.info(f"[Test {test_index}] API响应已保存到: {filepath}")
        except Exception as e:
            with self.logger_lock:
                logger.error(f"[Test {test_index}] 保存API响应失败: {str(e)}")
    
    def _is_theme_similar(self, theme1: str, theme2: str) -> bool:
        """
        判断两个theme是否相似（允许版本号差异）
        例如：'eRob机器人关节模组用户手册_V3.39' 和 'eRob机器人关节模组用户手册_V3.40' 应该匹配
        """
        if not theme1 or not theme2:
            return False
        
        # 移除版本号部分（V3.39, V3.40, 3.39等）
        def remove_version(text: str) -> str:
            # 匹配 V数字.数字 或 数字.数字 在末尾
            text = re.sub(r'[Vv]?\d+\.\d+$', '', text)
            # 移除末尾的下划线或空格
            text = re.sub(r'[_\s]+$', '', text)
            return text.strip()
        
        base1 = remove_version(theme1)
        base2 = remove_version(theme2)
        
        # 如果基础名称相同，认为是相似的（允许版本号差异）
        return base1 == base2 and len(base1) > 0
    
    def _detect_keywords_format(self, important_keywords: List[str]) -> str:
        """
        检测 important_keywords 的格式
        
        Args:
            important_keywords: important_keywords 列表
            
        Returns:
            "new" 或 "old" - 新格式或旧格式
        """
        if not important_keywords or len(important_keywords) < 2:
            return "old"  # 默认旧格式
        
        # 新格式特征：第一个元素包含 " > " 符号（完整路径格式）
        first_item = str(important_keywords[0]).strip()
        if " > " in first_item:
            return "new"
        
        # 旧格式特征：第一个元素是文档名字，不包含 " > "
        return "old"
    
    def _extract_chapter_from_important_keywords(self, important_keywords: List[str]) -> Optional[str]:
        """
        从important_keywords列表中提取章节信息
        
        支持两种格式：
        1. 新格式：[0]=完整路径（文档名 > 章节）, [1]=文档名字, [2]=章节
        2. 旧格式：[0]=文档名字, [1]=大章, [2]=小章
        
        优先返回小章，没有则返回大章，没有则None

        例如（新格式）：
        important_keywords = [
            "eRob CANopen  and EtherCAT User Manual V1.9.docx > 一、 介绍 / Chapter 1 Introduction",
            "eRob CANopen  and EtherCAT User Manual V1.9.docx",
            "一、 介绍 / Chapter 1 Introduction"
        ]
        - 从 [2] 提取章节信息

        例如（旧格式）：
        important_keywords = ["手册名称", "二十六、故障排查说明.md", "26.3. 电机堵转报错的排查步骤"]
        - 返回 "26.3"（小章编号）

        important_keywords = ["手册名称", "二十六、故障排查说明.md"]
        - 返回 "二十六"（大章编号）
        """
        if not important_keywords or len(important_keywords) < 2:
            return None

        # 检测格式
        format_type = self._detect_keywords_format(important_keywords)
        
        # 从规则管理器获取索引结构（如果可用）
        rule_manager = self.rule_manager or ChapterMatcher._get_rule_manager()
        
        if format_type == "new":
            # 新格式：[0]=完整路径, [1]=文档名字, [2]=章节
            if rule_manager:
                chapter_index = rule_manager.get_extraction_rule("important_keywords_structure.new_format.chapter_index", 2)
            else:
                chapter_index = 2
            
            # 从 [2] 提取章节信息
            if len(important_keywords) > chapter_index and important_keywords[chapter_index]:
                chapter_text = str(important_keywords[chapter_index]).strip()
                
                # 优先使用格式提取器提取小章
                if self.format_extractor:
                    chapter_info = self.format_extractor.extract_minor_chapter(chapter_text)
                    if chapter_info:
                        return chapter_info
                    
                    # 如果没有小章，提取大章
                    chapter_info = self.format_extractor.extract_major_chapter(chapter_text)
                    if chapter_info:
                        return chapter_info
                
                # 降级方案：使用原有的提取逻辑
                chapter_info = ChapterMatcher.extract_chapter_info(chapter_text)
                if chapter_info:
                    return chapter_info
        else:
            # 旧格式：[0]=文档名字, [1]=大章, [2]=小章
            if rule_manager:
                major_chapter_index = rule_manager.get_extraction_rule("important_keywords_structure.old_format.major_chapter_index", 1)
                minor_chapter_index = rule_manager.get_extraction_rule("important_keywords_structure.old_format.minor_chapter_index", 2)
            else:
                major_chapter_index = 1
                minor_chapter_index = 2

            # 优先使用格式提取器提取小章（如果可用）
            if self.format_extractor and len(important_keywords) > minor_chapter_index and important_keywords[minor_chapter_index]:
                minor_text = str(important_keywords[minor_chapter_index])
                chapter_info = self.format_extractor.extract_minor_chapter(minor_text)
                if chapter_info:
                    return chapter_info

            # 如果没有小章，使用格式提取器提取大章
            if self.format_extractor and len(important_keywords) > major_chapter_index and important_keywords[major_chapter_index]:
                major_text = str(important_keywords[major_chapter_index])
                chapter_info = self.format_extractor.extract_major_chapter(major_text)
                if chapter_info:
                    return chapter_info

            # 降级方案：使用原有的提取逻辑
            # 优先提取小章（minor_chapter_index）
            if len(important_keywords) > minor_chapter_index and important_keywords[minor_chapter_index]:
                chapter_info = ChapterMatcher.extract_chapter_info(str(important_keywords[minor_chapter_index]))
                if chapter_info:
                    return chapter_info

            # 没有小章则提取大章（major_chapter_index）
            if len(important_keywords) > major_chapter_index and important_keywords[major_chapter_index]:
                chapter_info = ChapterMatcher.extract_chapter_info(str(important_keywords[major_chapter_index]))
                if chapter_info:
                    return chapter_info

        # 都没有则返回None
        return None
    
    def run_single_test(self, test_case: TestCase, test_index: int = 0) -> Dict[str, Any]:
        """运行单个测试用例"""
        # 为日志添加测试索引前缀，便于在并行执行时追踪
        log_prefix = f"[Test {test_index}]"
        verbose = getattr(self, 'verbose_logging', True)  # 默认详细模式
        
        if verbose:
            # 详细模式（单线程）：显示完整信息
            logger.info(f"{log_prefix} {'=' * 80}")
            logger.info(f"{log_prefix} 测试Question: {test_case.question}")
            logger.info(f"{log_prefix} 手册Theme: {test_case.theme}")
            logger.info(f"{log_prefix} 标注Reference: {test_case.reference}")
        else:
            # 简洁模式（并发）：合并为一行
            logger.info(f"{log_prefix} {'=' * 80}")
            logger.info(f"{log_prefix} Q: {test_case.question[:60]}{'...' if len(test_case.question) > 60 else ''} | Theme: {test_case.theme} | Ref: {test_case.reference}")
        
        start_time = time.time()
        response = self.client.search(test_case.question, test_case.theme, self.config)
        latency = time.time() - start_time
        
        # 保存完整的API响应到单独的文件
        self._save_api_response(test_case, response, test_index)
        
        if "error" in response:
            logger.warning(f"{log_prefix} API返回错误: {response.get('error')}")
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
        if verbose:
            logger.info(f"{log_prefix} 检索到的chunks总数: {len(chunks)}")
        else:
            logger.debug(f"{log_prefix} 检索到的chunks总数: {len(chunks)}")
 
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
            chapter_full_text = None  # 完整的章节文本（用于标识符匹配）
            if important_keywords:
                chapter_info = self._extract_chapter_from_important_keywords(important_keywords)
                # 获取完整的章节文本（用于标识符匹配）
                # 新格式：[2] 是章节，旧格式：[2] 是小章或 [1] 是大章
                format_type = self._detect_keywords_format(important_keywords)
                if format_type == "new":
                    # 新格式：[2] 是章节
                    if len(important_keywords) > 2:
                        chapter_full_text = str(important_keywords[2]).strip()
                else:
                    # 旧格式：优先使用 [2]（小章），如果没有则使用 [1]（大章）
                    if len(important_keywords) > 2:
                        chapter_full_text = str(important_keywords[2]).strip()
                    elif len(important_keywords) > 1:
                        chapter_full_text = str(important_keywords[1]).strip()
            
            sim = get_similarity_score(chunk)
            all_chunks_info.append({
                'index': chunk_idx + 1,
                'chapter': chapter_info,
                'chapter_full_text': chapter_full_text,  # 完整的章节文本
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
        if verbose:
            logger.info(f"{log_prefix} 选取top {top_k_for_recall} 个结果进行验证")
        else:
            logger.debug(f"{log_prefix} 选取top {top_k_for_recall} 个结果进行验证")
        
        # 提取标注信息
        reference_chapter = ChapterMatcher.extract_chapter_info(test_case.reference)
        theme_str = str(test_case.theme).strip() if test_case.theme else ""
        
        # 使用相似度最高的第一个结果进行评测（用于评估RAG相似度算法）
        # 这是RAG返回给使用者的第一个结果，用来判断相似度算法的效果
        top1_result = top_k_chunks[0] if top_k_chunks else None
        original_chapter = top1_result.get('chapter') if top1_result else None
        original_chapter_title = top1_result.get('chapter_full_text') if top1_result else None
        
        # 获取映射后的章节号
        mapped_chapter = None
        if original_chapter and reference_chapter:
            retrieved_chapter_for_mapping = original_chapter_title if original_chapter_title else original_chapter
            reference_chapter_full = test_case.reference  # 使用完整的参考章节文本
            is_mapped, mapped_chapter_num = ChapterMatcher.get_mapped_chapter(
                retrieved_chapter_for_mapping,
                reference_chapter_full
            )
            if is_mapped and mapped_chapter_num:
                mapped_chapter = mapped_chapter_num
        
        # 保持向后兼容，final_chapter用于后续逻辑
        final_chapter = original_chapter
        
        # 验证topK中所有结果：important_keywords[0] 是否等于 theme，章节是否等于 reference
        # 用于recall@k计算时，只考虑同时满足两个条件的结果
        
        def verify_item(item):
            """验证单个结果是否同时满足theme和章节匹配"""
            important_keywords = item.get('important_keywords', [])
            chapter_info = item.get('chapter')
            chapter_full_text = item.get('chapter_full_text')  # 完整的章节文本
            
            # 验证 important_keywords 中的文档名字是否等于 theme（支持映射和别名）
            theme_match = False
            if important_keywords and theme_str:
                # 检测格式并获取文档名字索引
                format_type = self._detect_keywords_format(important_keywords)
                if format_type == "new":
                    # 新格式：[1] 是文档名字
                    book_name_index = 1
                else:
                    # 旧格式：[0] 是文档名字
                    book_name_index = 0
                
                if len(important_keywords) > book_name_index:
                    retrieved_theme = str(important_keywords[book_name_index]).strip()
                else:
                    retrieved_theme = str(important_keywords[0]).strip()  # 降级方案
                
                # 标准化处理：移除文件扩展名和版本号差异
                def normalize_theme_for_match(theme: str) -> str:
                    """标准化theme用于匹配：移除文件扩展名，统一版本号格式"""
                    if not theme:
                        return ""
                    # 移除文件扩展名（.docx, .md, .pdf等）
                    theme = re.sub(r'\.(docx|md|pdf|txt)$', '', theme, flags=re.IGNORECASE)
                    # 统一版本号格式：V3.39, V3.40, 3.39, 3.40 -> 统一为 V3.x 格式
                    # 这里我们保留版本号，但允许版本号差异（通过后续的模糊匹配处理）
                    return theme.strip()
                
                normalized_retrieved = normalize_theme_for_match(retrieved_theme)
                normalized_theme = normalize_theme_for_match(theme_str)
                
                # 精确匹配（标准化后）
                if normalized_retrieved == normalized_theme:
                    theme_match = True
                # 模糊匹配：允许版本号差异（如 V3.39 vs V3.40）
                elif self._is_theme_similar(normalized_retrieved, normalized_theme):
                    theme_match = True
                # 如果配置了手册映射，检查别名匹配
                elif self.manual_mapper:
                    # 获取 theme 的所有别名
                    theme_aliases = self.manual_mapper.get_aliases(theme_str)
                    if retrieved_theme in theme_aliases or normalized_retrieved in [normalize_theme_for_match(a) for a in theme_aliases]:
                        theme_match = True
                    # 也检查检索到的 theme 的别名
                    retrieved_aliases = self.manual_mapper.get_aliases(retrieved_theme)
                    if theme_str in retrieved_aliases or normalized_theme in [normalize_theme_for_match(a) for a in retrieved_aliases]:
                        theme_match = True
            
            # 验证章节是否等于 reference（使用is_valid_match判断）
            # 优先使用完整的章节文本进行标识符匹配，如果没有则使用提取的章节编号
            retrieved_chapter_for_match = chapter_full_text if chapter_full_text else chapter_info
            chapter_match = (
                bool(retrieved_chapter_for_match) and 
                bool(reference_chapter) and 
                ChapterMatcher.is_valid_match(retrieved_chapter_for_match, test_case.reference)
            )
            
            # 记录验证结果
            item['theme_match'] = theme_match
            item['chapter_match'] = chapter_match
            item['both_match'] = theme_match and chapter_match
            
            return theme_match, chapter_match
        
        # 验证topK中所有结果
        for item in top_k_chunks:
            verify_item(item)
        
        # 获取第一个结果的验证状态（用于top1指标）
        top1_theme_match = False
        top1_chapter_match = False
        if top1_result:
            top1_theme_match = top1_result.get('theme_match', False)
            top1_chapter_match = top1_result.get('chapter_match', False)
        
        # 统计topK中的验证情况（合并为一次遍历，提高性能）
        both_match_count = theme_match_count = chapter_match_count = 0
        matched_chapters = []  # 记录匹配的章节
        for item in top_k_chunks:
            if item.get('both_match', False):
                both_match_count += 1
            if item.get('theme_match', False):
                theme_match_count += 1
            if item.get('chapter_match', False):
                chapter_match_count += 1
                # 记录匹配的章节（用于日志显示）
                chapter = item.get('chapter')
                if chapter:
                    matched_chapters.append(chapter)
        
        # 输出验证结果（根据verbose模式决定详细程度）
        if verbose:
            # 详细模式：显示完整信息
            logger.info(f"{log_prefix} {'-' * 80}")
            logger.info(f"{log_prefix} 标注Theme: {test_case.theme}")
            logger.info(f"{log_prefix} 标注章节(reference): {reference_chapter}")
            logger.info(f"{log_prefix} Top {top_k_for_recall} 个结果的验证情况:")
            
            # 显示topK中每个结果的验证情况
            for rank, item in enumerate(top_k_chunks, 1):
                important_keywords = item.get('important_keywords', [])
                # 检测格式并获取文档名字
                if important_keywords and len(important_keywords) > 0:
                    format_type = self._detect_keywords_format(important_keywords)
                    if format_type == "new":
                        book_name_index = 1 if len(important_keywords) > 1 else 0
                    else:
                        book_name_index = 0
                    first_keyword = str(important_keywords[book_name_index]) if len(important_keywords) > book_name_index else "无"
                else:
                    first_keyword = "无"
                chapter_display = item.get('chapter') if item.get('chapter') else "无章节信息"
                theme_match = item.get('theme_match', False)
                chapter_match = item.get('chapter_match', False)
                both_match = item.get('both_match', False)
                logger.info(f"{log_prefix}   #{rank}: 标题='{first_keyword}', 章节='{chapter_display}', "
                           f"相似度: {item['similarity'] if item['similarity'] is not None else '无'}")
                logger.info(f"{log_prefix}     theme匹配: {'✓' if theme_match else '✗'}, "
                           f"章节匹配: {'✓' if chapter_match else '✗'}, "
                           f"同时满足: {'✓' if both_match else '✗'}")
            
            logger.info(f"{log_prefix} 统计: 同时满足={both_match_count}/{top_k_for_recall}, "
                       f"仅theme匹配={theme_match_count}/{top_k_for_recall}, "
                       f"仅章节匹配={chapter_match_count}/{top_k_for_recall}")
        else:
            # 简洁模式：只显示统计信息
            logger.debug(f"{log_prefix} {'-' * 80}")
            logger.debug(f"{log_prefix} 标注Theme: {test_case.theme} | 标注章节: {reference_chapter}")
            logger.debug(f"{log_prefix} 统计: 同时满足={both_match_count}/{top_k_for_recall}, "
                       f"仅theme={theme_match_count}/{top_k_for_recall}, "
                       f"仅章节={chapter_match_count}/{top_k_for_recall}")
        
        # 提取topK中所有同时有theme和chapter信息的结果（不管对错，只要同时有theme和chapter）
        # 用于recall@k计算：只要topk中有一个章节匹配reference，就返回1
        # 注意：我们直接使用 verify_item 的结果来判断匹配，而不是重新调用 is_valid_match
        all_topk_chapters_for_recall = []  # 所有topk中有chapter信息的结果（用于recall@k计算，存储匹配状态）
        validated_chapters = []  # 同时满足theme和章节匹配的结果（用于其他指标）
        theme_matched_chapters = []  # theme匹配后的所有章节（用于日志显示）
        
        for item in top_k_chunks:
            chapter = item.get('chapter')
            important_keywords = item.get('important_keywords', [])
            # 检查是否同时有theme和chapter信息
            has_theme = bool(important_keywords) and len(important_keywords) > 0
            has_chapter = bool(chapter)
            
            if has_theme and has_chapter:
                # verify_item 已经计算了 chapter_match，我们直接使用这个结果
                # 存储章节信息和匹配状态（用于recall@k计算）
                all_topk_chapters_for_recall.append({
                    'chapter': chapter,
                    'chapter_match': item.get('chapter_match', False)
                })
            
            if chapter:
                if item.get('both_match', False):
                    validated_chapters.append(chapter)
                if item.get('theme_match', False):
                    theme_matched_chapters.append(chapter)
        
        # 显示章节信息（根据verbose模式决定详细程度）
        actual_chapter = mapped_chapter if mapped_chapter else (final_chapter if final_chapter else "无")
        if verbose:
            logger.info(f"{log_prefix} 正确章节: {reference_chapter}")
            if mapped_chapter:
                logger.info(f"{log_prefix} 实际章节: {mapped_chapter}（映射后）")
            else:
                logger.info(f"{log_prefix} 实际章节: {final_chapter if final_chapter else '无'}")
            logger.info(f"{log_prefix} 候选章节（Top {top_k_for_recall}，theme匹配后的所有章节）: {', '.join(theme_matched_chapters) if theme_matched_chapters else '无'}")
        else:
            chapter_status = f"{actual_chapter}（映射后）" if mapped_chapter else actual_chapter
            logger.debug(f"{log_prefix} 正确章节: {reference_chapter} | 实际章节: {chapter_status}")
        
        # 计算各项指标 - TopK 指标包含 Top1（从第1个结果开始）
        # 计算recall@k：直接使用 verify_item 的结果（已经考虑了映射和标识符匹配）
        # 获取top3, top5, top10的章节列表（用于日志显示，从Top1开始）
        all_topk_chapters = [item['chapter'] for item in all_topk_chapters_for_recall]
        top3_chapters = all_topk_chapters[:3] if len(all_topk_chapters) >= 3 else all_topk_chapters
        top5_chapters = all_topk_chapters[:5] if len(all_topk_chapters) >= 5 else all_topk_chapters
        top10_chapters = all_topk_chapters[:10] if len(all_topk_chapters) >= 10 else all_topk_chapters
        
        # Recall@K 从 Top1 开始计算（start_from=1）
        # 直接使用 verify_item 的结果，不需要重新调用 is_valid_match
        def calculate_recall_at_k(items: List[Dict], k: int, start_from: int = 1) -> float:
            """基于 verify_item 的结果计算 Recall@K"""
            start_idx = max(0, start_from - 1)
            end_idx = min(start_idx + k, len(items))
            for item in items[start_idx:end_idx]:
                if item.get('chapter_match', False):
                    return 1.0
            return 0.0
        
        recall_at_3 = calculate_recall_at_k(all_topk_chapters_for_recall, 3, start_from=1)
        recall_at_5 = calculate_recall_at_k(all_topk_chapters_for_recall, 5, start_from=1)
        recall_at_10 = calculate_recall_at_k(all_topk_chapters_for_recall, 10, start_from=1)
        
        # 在日志中显示recall@k的结果（根据verbose模式决定详细程度）
        if verbose:
            # 详细模式：分别显示每项指标
            logger.info(f"{log_prefix} {'-'*80}")
            logger.info(f"{log_prefix} Recall@3: {'是' if recall_at_3 == 1.0 else '否'}, Top3章节: {', '.join(top3_chapters) if top3_chapters else '无'}")
            logger.info(f"{log_prefix} Recall@5: {'是' if recall_at_5 == 1.0 else '否'}, Top5章节: {', '.join(top5_chapters) if top5_chapters else '无'}")
            logger.info(f"{log_prefix} Recall@10: {'是' if recall_at_10 == 1.0 else '否'}, Top10章节: {', '.join(top10_chapters) if top10_chapters else '无'}")
            logger.info(f"{log_prefix} API响应延迟: {latency:.3f}s (单次请求时间)")
        else:
            # 简洁模式：合并为一行
            recall_status = f"R@3:{'✓' if recall_at_3 == 1.0 else '✗'} R@5:{'✓' if recall_at_5 == 1.0 else '✗'} R@10:{'✓' if recall_at_10 == 1.0 else '✗'}"
            matched_info = f"匹配章节: {', '.join(matched_chapters[:3])}" if matched_chapters else "无匹配章节"
            logger.info(f"{log_prefix} {recall_status} | {matched_info} | 延迟: {latency:.2f}s")
            logger.debug(f"{log_prefix} Top3: {', '.join(top3_chapters) if top3_chapters else '无'} | "
                        f"Top5: {', '.join(top5_chapters) if top5_chapters else '无'} | "
                        f"Top10: {', '.join(top10_chapters) if top10_chapters else '无'}")
        
        metrics = {
            "test_index": test_index,  # 保存测试索引，用于排序
            "question": test_case.question,
            # 映射后的章节号（如果identifier匹配成功，使用映射后的章节号；否则使用原始章节号）
            "answer": mapped_chapter if mapped_chapter else (original_chapter if original_chapter else ""),
            # 原始章节号（从API返回的原始章节）
            "original_answer": original_chapter if original_chapter else "",
            # 完整章节标题文本（包含章节编号和标题）
            "answer_title": original_chapter_title if original_chapter_title else "",
            "reference": test_case.reference,
            "type": test_case.type,
            "theme": test_case.theme,
            "retrieved_count": len(validated_chapters),  # 同时满足theme和章节匹配的结果数量
            "latency": latency,
            # Recall@K（从Top1开始，作为召回率指标）
            "recall": recall_at_10,  # 使用 Recall@10 作为召回率指标
            "recall@3": recall_at_3,
            "recall@5": recall_at_5,
            "recall@10": recall_at_10,
            # RAG第一结果的验证状态（保留用于对比）
            "top1_theme_match": top1_theme_match,
            "top1_chapter_match": top1_chapter_match,
            "top1_both_match": top1_theme_match and top1_chapter_match,
            # 保存topk章节信息，用于异常分析（从Top1开始）
            "top3_chapters": top3_chapters,
            "top5_chapters": top5_chapters,
            "top10_chapters": top10_chapters,
        }
        
        return metrics
    
    def _save_anomalies(self, anomalies: List[Dict[str, Any]], output_path: Optional[str] = None):
        """保存异常情况到文件"""
        logger.info('-'*80)
        if output_path is None:
            output_path = str(self.output_base_dir / "anomalies.json")
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
        logger.info("=" * 80)
        logger.info("--- 开始批量评测 ---")
        logger.info(f"开始批量评测，共 {total_cases} 条用例")
        logger.info(f"API响应将保存到目录: {self.api_responses_dir.absolute()}")
        
        if max_workers > 1:
            logger.info(f"使用并发模式，并发线程数: {max_workers}")
            logger.info("提示: 按 Ctrl+C 可以中断执行，已完成的结果会被保存")
            # 全开模式：无论单线程还是多线程都显示详细日志
            self.verbose_logging = True
        else:
            logger.info("使用顺序执行模式")
            logger.info("提示: 按 Ctrl+C 可以中断执行，已完成的结果会被保存")
            # 全开模式：无论单线程还是多线程都显示详细日志
            self.verbose_logging = True
        logger.info("=" * 80)
        logger.info("")
        
        anomalies = []  # 记录异常情况
        anomalies_lock = Lock()  # 异常列表的线程安全锁
        completed_count = 0  # 已完成数量
        completed_lock = Lock()  # 完成计数的线程安全锁
        stop_event = Event()  # 停止事件标志
        
        def run_single_test_with_index(args):
            """包装函数，用于并发执行"""
            idx, test_case = args
            try:
                # 检查是否收到停止信号
                if stop_event.is_set():
                    with self.logger_lock:
                        logger.warning(f"[Test {idx}] 已收到停止信号，跳过执行")
                    return None
                
                # 线程安全的进度日志
                with self.logger_lock:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"[Test {idx}] 开始执行测试 ({idx}/{total_cases})")
                    logger.info(f"{'='*80}\n")
                
                # 执行测试
                result = self.run_single_test(test_case, test_index=idx)
                
                # 再次检查停止信号
                if stop_event.is_set():
                    with self.logger_lock:
                        logger.warning(f"[Test {idx}] 执行过程中收到停止信号")
                    return None
                
                # 测试完成日志
                with self.logger_lock:
                    logger.info(f"[Test {idx}] 测试完成")
                
                # 线程安全地添加结果
                with self.results_lock:
                    self.results.append(result)
                
                # 检查是否为异常情况
                if result.get('recall', 1.0) == 0.0:
                    anomaly = {
                        "test_index": idx,
                        "question": result.get('question', ''),
                        "reference": result.get('reference', ''),
                        "theme": result.get('theme', ''),
                        "type": result.get('type', ''),
                        "answer": result.get('answer', ''),
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
                        logger.warning(f"[Test {idx}] 发现异常情况: 召回率={result.get('recall', 0.0)}")
                
                # 更新完成计数
                with completed_lock:
                    nonlocal completed_count
                    completed_count += 1
                    if completed_count % 10 == 0 or completed_count == total_cases:
                        with self.logger_lock:
                            logger.info(f"[进度] 已完成 {completed_count}/{total_cases} 个测试用例 ({completed_count/total_cases*100:.1f}%)")
                
                # 延迟以避免API限流（每个线程独立延迟）
                if delay_between_requests > 0 and not stop_event.is_set():
                    time.sleep(delay_between_requests)
                
                return result
            except Exception as e:
                with self.logger_lock:
                    logger.error(f"[Test {idx}] 测试用例执行失败: {str(e)}")
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
                    "recall": 0.0,
                }
                with self.results_lock:
                    self.results.append(error_result)
                with completed_lock:
                    completed_count += 1
                return error_result
        
        # 根据 max_workers 决定使用并发还是顺序执行
        start_time = time.time()  # 记录开始时间
        if max_workers > 1:
            # 并发执行
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有任务
                    future_to_index = {
                        executor.submit(run_single_test_with_index, (idx, test_case)): idx
                        for idx, test_case in enumerate(test_cases, 1)
                    }
                    
                    # 等待所有任务完成
                    for future in as_completed(future_to_index):
                        # 检查停止信号
                        if stop_event.is_set():
                            with self.logger_lock:
                                logger.warning("收到停止信号，等待当前任务完成...")
                            # 不再提交新任务，但等待正在执行的任务完成
                            # 继续处理已完成的任务
                        
                        try:
                            result = future.result()  # 获取结果，如果有异常会在这里抛出
                            if result is None:  # 如果返回None，说明被停止了
                                continue
                        except Exception as e:
                            idx = future_to_index.get(future, 'unknown')
                            with self.logger_lock:
                                logger.error(f"[Test {idx}] 任务执行异常: {str(e)}")
                        
                        # 如果已停止，检查是否还有未完成的任务
                        if stop_event.is_set():
                            remaining = sum(1 for f in future_to_index if not f.done())
                            if remaining == 0:
                                break
                
                elapsed_time = time.time() - start_time
                self.total_runtime = elapsed_time  # 保存实际运行时间
                if stop_event.is_set():
                    logger.warning(f"执行被用户中断，已完成部分结果，实际运行时间: {elapsed_time:.2f}秒")
                else:
                    logger.info(f"并发执行完成，实际运行时间: {elapsed_time:.2f}秒")
            except KeyboardInterrupt:
                # 捕获 Ctrl+C
                with self.logger_lock:
                    logger.warning("\n收到中断信号 (Ctrl+C)，正在停止...")
                stop_event.set()
                # 等待当前正在执行的任务完成
                with self.logger_lock:
                    logger.info("等待当前任务完成...")
                time.sleep(2)  # 给一些时间让任务完成
                elapsed_time = time.time() - start_time
                self.total_runtime = elapsed_time  # 保存实际运行时间
                logger.warning(f"执行被用户中断，已完成部分结果，实际运行时间: {elapsed_time:.2f}秒")
        else:
            # 顺序执行（原有逻辑）
            try:
                for idx, test_case in enumerate(test_cases, 1):
                    if stop_event.is_set():
                        logger.warning("收到停止信号，停止顺序执行")
                        break
                    run_single_test_with_index((idx, test_case))
                elapsed_time = time.time() - start_time
                self.total_runtime = elapsed_time  # 保存实际运行时间
                logger.info(f"顺序执行完成，实际运行时间: {elapsed_time:.2f}秒")
            except KeyboardInterrupt:
                logger.warning("\n收到中断信号 (Ctrl+C)，正在停止...")
                stop_event.set()
                elapsed_time = time.time() - start_time
                self.total_runtime = elapsed_time  # 保存实际运行时间
                logger.warning(f"执行被用户中断，已完成部分结果，实际运行时间: {elapsed_time:.2f}秒")
        
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
            output_path = str(self.output_base_dir / "evaluation_dashboard.html")
        
        # 计算latency统计信息
        latency_stats = {}
        if 'latency' in df.columns:
            latency_stats['latency_avg'] = df['latency'].mean()  # 单个请求的平均延迟
            latency_stats['latency_min'] = df['latency'].min()  # 最小延迟
            latency_stats['latency_max'] = df['latency'].max()  # 最大延迟
            latency_stats['latency_sum'] = df['latency'].sum()  # 所有请求延迟的累加（用于对比）
        
        # 使用实际运行时间（从开始到结束的真实时间），而不是所有请求延迟的累加
        # 注意：在多线程环境下，total_runtime 会小于 latency_sum，因为请求是并发执行的
        if self.total_runtime is not None:
            latency_stats['latency_total'] = self.total_runtime  # 实际运行时间（墙钟时间）
        else:
            # 如果没有记录实际运行时间，回退到累加方式（兼容旧代码）
            if 'latency' in df.columns:
                latency_stats['latency_total'] = df['latency'].sum()
                logger.warning("未记录实际运行时间，使用请求延迟累加值（可能不准确）")
        
        # 使用仪表盘生成器
        dashboard = EvaluationDashboard(df, latency_stats)
        summary = dashboard.generate(output_path)
        
        logger.info(f"报告已生成: {output_path}")
        logger.info(f"Latency统计: 平均响应时间={latency_stats.get('latency_avg', 0):.3f}秒, "
                   f"最小={latency_stats.get('latency_min', 0):.3f}秒, "
                   f"最大={latency_stats.get('latency_max', 0):.3f}秒")
        
        latency_total = latency_stats.get('latency_total', 0)
        latency_sum = latency_stats.get('latency_sum', 0)
        latency_total_min = latency_total / 60.0
        latency_sum_min = latency_sum / 60.0
        concurrency_efficiency = latency_sum / max(latency_total, 0.01)
        
        logger.info(f"总运行时间: {latency_total:.2f}秒 ({latency_total_min:.2f}分钟) "
                   f"[延迟累加: {latency_sum:.2f}秒 ({latency_sum_min:.2f}分钟), "
                   f"并发效率: {concurrency_efficiency:.2f}x]")
        
        # 总体指标不再在这里输出，由主函数统一输出（使用print，避免多线程时重复）
        
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
        
        # 创建AB测试输出目录结构
        ab_test_output_dir = output_dir / "ab_test"
        ab_test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # A组输出目录
        output_dir_a = ab_test_output_dir / "a"
        output_dir_a.mkdir(parents=True, exist_ok=True)
        
        # B组输出目录
        output_dir_b = ab_test_output_dir / "b"
        output_dir_b.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"A组结果将保存到: {output_dir_a.absolute()}")
        logger.info(f"B组结果将保存到: {output_dir_b.absolute()}")
        
        # 创建带有独立输出目录的runner
        runner_a = EvaluationRunner(self.client_a, config_a, output_base_dir=output_dir_a)
        runner_b = EvaluationRunner(self.client_b, config_b, output_base_dir=output_dir_b)
        
        df_a = runner_a.run_batch_evaluation(test_cases, max_workers=max_workers, delay_between_requests=delay_between_requests)
        df_b = runner_b.run_batch_evaluation(test_cases, max_workers=max_workers, delay_between_requests=delay_between_requests)
        
        # 将实际运行时间添加到DataFrame中（作为元数据）
        if runner_a.total_runtime is not None:
            df_a['latency_total'] = runner_a.total_runtime
        if runner_b.total_runtime is not None:
            df_b['latency_total'] = runner_b.total_runtime
        
        # 生成对比报告
        comparison = self.generate_comparison(df_a, df_b)
        return comparison, df_a, df_b
    
    def generate_comparison(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
        """生成对比结果"""
        metric_columns = [col for col in df_a.columns if '@' in col or col in ['latency', 'recall']]
        
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
        
        bars_a = plt.bar([i - width/2 for i in x], comparison_df['Plan A'], width, label='Plan A')
        bars_b = plt.bar([i + width/2 for i in x], comparison_df['Plan B'], width, label='Plan B')
        
        # 添加数值标注
        def add_value_labels(bars, values):
            """在柱状图上添加数值标注"""
            for bar, value in zip(bars, values):
                height = bar.get_height()
                # 格式化数值：保留4位小数
                label_text = f'{value:.4f}'
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                        label_text, ha='center', va='bottom', fontsize=8)
        
        add_value_labels(bars_a, comparison_df['Plan A'])
        add_value_labels(bars_b, comparison_df['Plan B'])
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('A/B Test Comparison')
        plt.xticks(x, metrics, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        # 保存到ab_test目录
        ab_test_output_dir = output_dir / "ab_test"
        ab_test_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = ab_test_output_dir / 'ab_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存
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


def create_client_from_config(env: Optional[str] = None) -> RagFlowClient:
    """
    从配置文件创建 RagFlowClient（统一的环境配置函数）
    
    Args:
        env: 环境名称（'prod', 'dev', 'test'），如果为None则从config.DEFAULT_ENV读取
    
    Returns:
        RagFlowClient: 配置好的客户端实例
    
    Raises:
        ValueError: 配置缺失或无效
        ImportError: 找不到config模块
    """
    import config
    import logging
    
    logger = logging.getLogger(__name__)
    
    # 获取环境名称
    if env is None:
        env = getattr(config, 'DEFAULT_ENV', 'test').lower()
    else:
        env = env.lower()
    
    # 根据环境加载配置
    if env == 'prod':
        logger.info("================================================")
        logger.info("使用生产环境（OpenWebUI RAGFlow）")
        logger.info("================================================")
        api_url = getattr(config, 'PROD_API_URL', '')
        api_key = getattr(config, 'PROD_API_KEY', '')
        knowledge_id = getattr(config, 'PROD_KNOWLEDGE_ID', '')
        use_ragflow_format = getattr(config, 'USE_RAGFLOW_FORMAT', True)
        use_ragflow_index = getattr(config, 'USE_RAGFLOW_INDEX', True)
        ragflow_timeout = getattr(config, 'RAGFLOW_TIMEOUT', 30)
    elif env == 'dev':
        logger.info("================================================")
        logger.info("使用开发环境（标准 RagFlow API）")
        logger.info("================================================")
        api_url = getattr(config, 'DEV_API_URL', '')
        api_key = getattr(config, 'DEV_API_KEY', '')
        knowledge_id = None
        use_ragflow_format = False
        use_ragflow_index = False
        ragflow_timeout = 30
    else:  # test
        logger.info("================================================")
        logger.info("使用测试环境（标准 RagFlow API）")
        logger.info("================================================")
        api_url = getattr(config, 'TEST_API_URL', '')
        api_key = getattr(config, 'TEST_API_KEY', '')
        knowledge_id = None
        use_ragflow_format = False
        use_ragflow_index = False
        ragflow_timeout = 30
    
    if not api_url or not api_key:
        raise ValueError(f"配置文件中的 {env.upper()}_API_URL 或 {env.upper()}_API_KEY 为空")
    
    # 根据环境创建客户端
    if env == 'prod':
        # 生产环境：使用 OpenWebUI 接口
        client = RagFlowClient(
            api_url=api_url,
            api_key=api_key,
            knowledge_id=knowledge_id,
            use_ragflow_format=use_ragflow_format,
            use_ragflow_index=use_ragflow_index,
            ragflow_timeout=ragflow_timeout
        )
        logger.info(f"知识库ID: {knowledge_id}")
        logger.info(f"使用 RAGflow 格式: {use_ragflow_format}")
        logger.info(f"使用 RAGFlow 索引: {use_ragflow_index}")
        logger.info(f"超时时间: {ragflow_timeout}秒")
    else:
        # 开发/测试环境：使用标准 RagFlow API
        client = RagFlowClient(api_url=api_url, api_key=api_key)
    
    logger.info("")  # 空行分隔
    return client


if __name__ == "__main__":
    # 配置RagFlow客户端
    # 请注意：不同环境有不同的api_url和api_key，需要根据实际情况选择使用哪个环境

    # 从配置文件加载 API 配置
    try:
        import config
        client = create_client_from_config()
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
    
    # 创建临时runner来加载测试集（规则系统会在EvaluationRunner初始化时自动加载）
    logger.info("--- 规则系统初始化 ---")
    temp_runner = EvaluationRunner(client, RetrievalConfig(dataset_ids=[]))
    logger.info("")
    logger.info("--- 测试集加载 ---")
    test_cases = temp_runner.load_test_set(str(test_set_path))
    logger.info("")
    
    # 限制测试用例数量或范围（如果配置了）
    logger.info("--- 测试用例配置 ---")
    
    # 优先检查测试范围配置
    test_case_range = getattr(config, 'TEST_CASE_RANGE', None)
    if test_case_range and isinstance(test_case_range, tuple) and len(test_case_range) == 2:
        start_idx, end_idx = test_case_range
        # 转换为0-based索引（Excel行号从1开始，但列表索引从0开始）
        start_idx_0based = max(0, start_idx - 1)
        end_idx_0based = min(len(test_cases), end_idx)  # end_idx已经是1-based，直接使用
        
        if start_idx_0based < len(test_cases):
            test_cases = test_cases[start_idx_0based:end_idx_0based]
            logger.info(f"测试用例已限制为索引 {start_idx}-{end_idx}（共 {len(test_cases)} 个）")
        else:
            logger.warning(f"测试范围 {start_idx}-{end_idx} 超出测试集大小，使用全部测试用例")
            logger.info(f"使用全部测试用例（共 {len(test_cases)} 个）")
    else:
        # 如果没有设置范围，检查最大数量限制
        max_test_cases = getattr(config, 'MAX_TEST_CASES', None)
        if max_test_cases and max_test_cases > 0 and len(test_cases) > max_test_cases:
            test_cases = test_cases[:max_test_cases]
            logger.info(f"测试用例已限制为前 {max_test_cases} 个（共 {len(test_cases)} 个）")
        else:
            logger.info(f"使用全部测试用例（共 {len(test_cases)} 个）")
    logger.info("")
    
    # ========== 并发配置 ==========
    logger.info("--- 执行模式配置 ---")
    # 从配置文件读取并发参数，如果没有则使用默认值
    enable_concurrency = getattr(config, 'ENABLE_CONCURRENCY', True)  # 默认启用并发
    max_workers = getattr(config, 'MAX_WORKERS', 5)  # 默认5个线程
    delay_between_requests = getattr(config, 'DELAY_BETWEEN_REQUESTS', 0.5)  # 默认0.5秒延迟
    
    # 如果禁用了并发，强制设置为1（顺序执行）
    if not enable_concurrency:
        max_workers = 1
        logger.info("并发执行已禁用，使用顺序执行模式（单线程）")
    elif max_workers > 1:
        logger.info(f"并发模式已启用: {max_workers} 个并发线程")
        logger.info(f"请求延迟: {delay_between_requests} 秒")
    else:
        logger.info("使用顺序执行模式（单线程）")
    logger.info("")
    
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
            vector_similarity_weight=0.5,  # A组：默认权重
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
            vector_similarity_weight=0.9,  # B组：更高权重
            rerank_id="",  # API默认值
            highlight=False,  # API默认值
            page=1,  # API默认值
            page_size=30  # API默认值
        )
        
        # 创建AB测试输出目录
        ab_test_output_dir = output_dir / "ab_test"
        ab_test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 根据实际配置动态生成README说明文件，只列出A组和B组的差异
        def get_diff_params(config_a: RetrievalConfig, config_b: RetrievalConfig) -> List[str]:
            """找出A组和B组配置的差异参数"""
            diff_params = []
            if config_a.vector_similarity_weight != config_b.vector_similarity_weight:
                diff_params.append(f"vector_similarity_weight: A={config_a.vector_similarity_weight}, B={config_b.vector_similarity_weight}")
            if config_a.top_k != config_b.top_k:
                diff_params.append(f"top_k: A={config_a.top_k}, B={config_b.top_k}")
            if config_a.similarity_threshold != config_b.similarity_threshold:
                diff_params.append(f"similarity_threshold: A={config_a.similarity_threshold}, B={config_b.similarity_threshold}")
            if config_a.rerank_id != config_b.rerank_id:
                diff_params.append(f"rerank_id: A={config_a.rerank_id or 'None'}, B={config_b.rerank_id or 'None'}")
            if config_a.highlight != config_b.highlight:
                diff_params.append(f"highlight: A={config_a.highlight}, B={config_b.highlight}")
            return diff_params
        
        diff_params = get_diff_params(config_a, config_b)
        diff_content = "\n".join([f"- {param}" for param in diff_params]) if diff_params else "- 配置相同（请检查代码）"
        
        readme_content = f"""# AB测试配置说明

## 配置差异

{diff_content}

## 目录结构
- `a/` - A组测试结果
- `b/` - B组测试结果
- `ab_test_comparison.csv` - AB测试对比结果
- `ab_comparison.png` - AB测试对比图表
"""
        readme_path = ab_test_output_dir / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')
        logger.info(f"AB测试配置说明文件已保存到: {readme_path}")
        
        # 创建A/B测试对比器（使用同一个客户端，只是配置不同）
        ab_comparator = ABTestComparator(client, client)
        
        # 运行A/B测试（支持并发）
        comparison_df, results_df_a, results_df_b = ab_comparator.run_ab_test(
            test_cases, config_a, config_b,
            max_workers=max_workers,
            delay_between_requests=delay_between_requests
        )
        
        # 保存A/B测试对比结果到ab_test目录
        comparison_csv_path = ab_test_output_dir / "ab_test_comparison.csv"
        comparison_df.to_csv(comparison_csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"A/B测试对比结果已保存到: {comparison_csv_path}")
        
        # 生成A组和B组的详细报告（使用已有的runner，它们已经有正确的输出目录）
        logger.info("=" * 80)
        logger.info("生成A组详细报告")
        logger.info("=" * 80)
        # A组和B组的runner已经在run_ab_test中创建，但我们需要重新创建以生成报告
        output_dir_a = ab_test_output_dir / "a"
        runner_a = EvaluationRunner(client, config_a, output_base_dir=output_dir_a)
        # 如果DataFrame中有latency_total，将其设置到runner中以便生成报告
        if 'latency_total' in results_df_a.columns and len(results_df_a['latency_total'].dropna()) > 0:
            runner_a.total_runtime = results_df_a['latency_total'].iloc[0]
        summary_a = runner_a.generate_report(results_df_a, str(output_dir_a / "evaluation_results_a.html"))
        csv_path_a = output_dir_a / "evaluation_results_a.csv"
        results_df_a.to_csv(csv_path_a, index=False, encoding='utf-8-sig')
        logger.info(f"A组评测结果已保存到: {csv_path_a}")
        
        logger.info("=" * 80)
        logger.info("生成B组详细报告")
        logger.info("=" * 80)
        output_dir_b = ab_test_output_dir / "b"
        runner_b = EvaluationRunner(client, config_b, output_base_dir=output_dir_b)
        # 如果DataFrame中有latency_total，将其设置到runner中以便生成报告
        if 'latency_total' in results_df_b.columns and len(results_df_b['latency_total'].dropna()) > 0:
            runner_b.total_runtime = results_df_b['latency_total'].iloc[0]
        summary_b = runner_b.generate_report(results_df_b, str(output_dir_b / "evaluation_results_b.html"))
        csv_path_b = output_dir_b / "evaluation_results_b.csv"
        results_df_b.to_csv(csv_path_b, index=False, encoding='utf-8-sig')
        logger.info(f"B组评测结果已保存到: {csv_path_b}")
        
        print("\n" + "=" * 80)
        print("===== A/B测试完成 =====")
        print("=" * 80)
        print("\n对比结果:")
        print(comparison_df.to_string(index=False))
        print("\nA组总体指标:")
        for metric, value in summary_a.items():
            if isinstance(value, (int, float)):
                if metric == 'latency_total':
                    # 总响应时间显示为2位小数
                    print(f"  {metric}: {value:.2f}s")
                elif 'latency' in metric.lower():
                    # 其他latency相关指标显示为3位小数
                    print(f"  {metric}: {value:.3f}s")
                else:
                    print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        print("\nB组总体指标:")
        for metric, value in summary_b.items():
            if isinstance(value, (int, float)):
                if metric == 'latency_total':
                    # 总响应时间显示为2位小数
                    print(f"  {metric}: {value:.2f}s")
                elif 'latency' in metric.lower():
                    # 其他latency相关指标显示为3位小数
                    print(f"  {metric}: {value:.3f}s")
                else:
                    print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        print("\n改进百分比 (B vs A):")
        for _, row in comparison_df.iterrows():
            improvement = row['Improvement']
            print(f"  {row['Metric']}: {improvement:+.2f}%")
    else:

        
        # ========== 单次评测模式 ==========
        print(f"\n将使用 {len(test_cases)} 条测试数据进行单次评测")
        
        # 创建单次测试输出目录
        single_output_dir = output_dir / "single"
        single_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"单次测试结果将保存到: {single_output_dir.absolute()}")
        
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
        
        # 直接使用temp_runner，只需更新配置和输出目录
        temp_runner.config = retrieval_config
        temp_runner.output_base_dir = single_output_dir
        temp_runner.output_base_dir.mkdir(parents=True, exist_ok=True)
        # 更新API响应保存目录
        temp_runner.api_responses_dir = single_output_dir / "api_responses"
        temp_runner.api_responses_dir.mkdir(parents=True, exist_ok=True)
        
        # 运行批量评测（支持并发）
        results_df = temp_runner.run_batch_evaluation(
            test_cases,
            max_workers=max_workers,
            delay_between_requests=delay_between_requests
        )
        
        # 生成报告
        logger.info("=" * 80)
        logger.info("生成评测报告")
        logger.info("=" * 80)
        summary = temp_runner.generate_report(results_df)
        
        # 保存结果到单次测试目录
        csv_path = single_output_dir / "evaluation_results.csv"
        # 如果记录了实际运行时间，将其添加到DataFrame中（作为元数据，每行相同）
        if temp_runner.total_runtime is not None:
            results_df['latency_total'] = temp_runner.total_runtime
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"评测结果已保存到: {csv_path}")
        
        print("\n" + "=" * 80)
        print("===== 评测完成 =====")
        print("=" * 80)
        print("\n总体指标:")
        for metric, value in summary.items():
            if isinstance(value, (int, float)):
                if metric == 'latency_total':
                    # 总响应时间显示为2位小数
                    print(f"  {metric}: {value:.2f}s")
                elif 'latency' in metric.lower():
                    # 其他latency相关指标显示为3位小数
                    print(f"  {metric}: {value:.3f}s")
                else:
                    print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
