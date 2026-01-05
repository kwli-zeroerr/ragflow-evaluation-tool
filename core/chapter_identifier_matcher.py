"""
章节标识符匹配器
基于对象字典编号和关键词进行章节匹配
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class ChapterIdentifierMatcher:
    """章节标识符匹配器 - 基于对象字典编号和关键词进行匹配"""
    
    def __init__(self, identifier_mappings: Dict[str, Dict[str, Any]]):
        """
        初始化标识符匹配器
        
        Args:
            identifier_mappings: 标识符映射规则字典，格式：
                {
                    "8.2.105. 0x6087:00h 扭矩斜坡（Torque slope）": {
                        "object_dict_id": "0X6087",
                        "keywords": ["扭矩斜坡", "Torque slope"],
                        "equivalent_chapters": ["8.2.103"]
                    }
                }
        """
        self.identifier_mappings = identifier_mappings or {}
    
    def match_by_identifier(
        self,
        retrieved_chapter: str,
        reference_chapter: str,
        retrieved_identifiers: Dict[str, Any],
        reference_identifiers: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        基于标识符进行匹配
        
        Args:
            retrieved_chapter: 检索到的章节（完整文本或章节编号）
            reference_chapter: 参考章节（完整文本或章节编号）
            retrieved_identifiers: 检索章节的标识符（包含 object_dict_id 和 keywords）
            reference_identifiers: 参考章节的标识符（包含 object_dict_id 和 keywords）
        
        Returns:
            (是否匹配, 映射后的章节号)
            如果匹配成功但章节号相同，mapped_chapter 为 None
            如果匹配成功但章节号不同，mapped_chapter 为参考章节号
        """
        if not retrieved_chapter or not reference_chapter:
            return (False, None)
        
        retrieved_object_dict_id = retrieved_identifiers.get('object_dict_id')
        reference_object_dict_id = reference_identifiers.get('object_dict_id')
        retrieved_keywords = retrieved_identifiers.get('keywords', [])
        reference_keywords = reference_identifiers.get('keywords', [])
        
        # 方法1：如果两者都有对象字典编号，直接比较
        if retrieved_object_dict_id and reference_object_dict_id:
            if retrieved_object_dict_id.upper() == reference_object_dict_id.upper():
                # 提取章节编号进行比较
                retrieved_chapter_num = self._extract_chapter_number(retrieved_chapter)
                reference_chapter_num = self._extract_chapter_number(reference_chapter)
                
                # 如果章节号相同，返回None（无需映射）
                if retrieved_chapter_num and reference_chapter_num:
                    if self._normalize_chapter_number(retrieved_chapter_num) == self._normalize_chapter_number(reference_chapter_num):
                        return (True, None)  # 匹配成功，章节号相同，无需映射
                
                # 如果章节号不同，返回参考章节号作为映射后的章节号
                if reference_chapter_num:
                    return (True, reference_chapter_num)  # 匹配成功，返回参考章节号作为映射
                
                return (True, None)  # 匹配成功，但无法提取章节号
        
        # 方法2：尝试从映射规则中查找匹配
        # 遍历映射规则，查找匹配的规则
        for mapping_key, mapping_value in self.identifier_mappings.items():
            mapping_object_dict_id = mapping_value.get('object_dict_id')
            mapping_keywords = mapping_value.get('keywords', [])
            equivalent_chapters = mapping_value.get('equivalent_chapters', [])
            
            # 检查对象字典编号是否匹配
            if mapping_object_dict_id:
                if retrieved_object_dict_id and retrieved_object_dict_id.upper() == mapping_object_dict_id.upper():
                    # 如果找到了匹配的映射规则，返回等价章节号
                    if equivalent_chapters:
                        return (True, equivalent_chapters[0])
                
                if reference_object_dict_id and reference_object_dict_id.upper() == mapping_object_dict_id.upper():
                    # 如果参考章节匹配映射规则，提取参考章节号
                    reference_chapter_num = self._extract_chapter_number(reference_chapter)
                    if reference_chapter_num:
                        return (True, reference_chapter_num)
            
            # 检查关键词是否匹配
            if mapping_keywords:
                # 检查检索章节的关键词是否匹配
                if retrieved_keywords:
                    for keyword in mapping_keywords:
                        if any(kw.lower() == keyword.lower() for kw in retrieved_keywords):
                            if equivalent_chapters:
                                return (True, equivalent_chapters[0])
                
                # 检查参考章节的关键词是否匹配
                if reference_keywords:
                    for keyword in mapping_keywords:
                        if any(kw.lower() == keyword.lower() for kw in reference_keywords):
                            reference_chapter_num = self._extract_chapter_number(reference_chapter)
                            if reference_chapter_num:
                                return (True, reference_chapter_num)
        
        # 方法3：直接比较关键词（如果对象字典编号都不存在）
        if not retrieved_object_dict_id and not reference_object_dict_id:
            if retrieved_keywords and reference_keywords:
                # 检查是否有共同的关键词
                retrieved_keywords_lower = [kw.lower() for kw in retrieved_keywords]
                reference_keywords_lower = [kw.lower() for kw in reference_keywords]
                
                common_keywords = set(retrieved_keywords_lower) & set(reference_keywords_lower)
                if common_keywords:
                    # 关键词匹配成功，提取参考章节号
                    reference_chapter_num = self._extract_chapter_number(reference_chapter)
                    if reference_chapter_num:
                        return (True, reference_chapter_num)
        
        return (False, None)
    
    @staticmethod
    def _extract_chapter_number(chapter: str) -> Optional[str]:
        """
        从章节文本中提取章节编号
        
        Args:
            chapter: 章节文本，例如 "8.2.105. 0x6087:00h 扭矩斜坡（Torque slope）" 或 "8.2.105"
        
        Returns:
            章节编号，例如 "8.2.105"，如果无法提取则返回 None
        """
        if not chapter:
            return None
        
        # 提取数字格式的章节编号（如 "1.1"、"1.2"、"10.1"）
        # 匹配模式：数字.数字.数字...（可能后面有空格、点、文字等）
        pattern = r'^(\d+(?:\.\d+)*)'
        match = re.match(pattern, chapter.strip())
        if match:
            return match.group(1)
        
        return None
    
    @staticmethod
    def _normalize_chapter_number(chapter_num: str) -> str:
        """
        标准化章节编号格式（移除末尾的点等）
        
        Args:
            chapter_num: 章节编号，例如 "8.2.105" 或 "8.2.105."
        
        Returns:
            标准化后的章节编号
        """
        if not chapter_num:
            return ""
        
        return chapter_num.strip().rstrip('.')
