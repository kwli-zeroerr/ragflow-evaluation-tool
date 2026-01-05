"""
章节标识符提取器
从章节文本中提取对象字典编号和关键词
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ChapterIdentifierExtractor:
    """章节标识符提取器 - 提取对象字典编号和关键词"""
    
    # 对象字典编号正则表达式：0x[0-9A-Fa-f]+
    OBJECT_DICT_PATTERN = re.compile(r'0x[0-9A-Fa-f]+', re.IGNORECASE)
    
    # 中文关键词提取：括号前的中文部分
    CHINESE_KEYWORD_PATTERN = re.compile(r'([^（(]+)（', re.UNICODE)
    
    # 英文关键词提取：括号内的英文部分
    ENGLISH_KEYWORD_PATTERN = re.compile(r'[（(]([^）)]+)[）)]', re.UNICODE)
    
    @staticmethod
    def extract_object_dict_id(text: str) -> Optional[str]:
        """
        从文本中提取对象字典编号
        
        Args:
            text: 章节文本，例如 "8.2.100. 0x6082:00h 结束速度（End velocity）"
        
        Returns:
            对象字典编号，例如 "0x6082"，如果未找到则返回 None
        """
        if not text:
            return None
        
        match = ChapterIdentifierExtractor.OBJECT_DICT_PATTERN.search(text)
        if match:
            return match.group(0).upper()  # 统一转换为大写
        return None
    
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """
        从文本中提取关键词（中文和英文）
        
        Args:
            text: 章节文本，例如 "8.2.100. 0x6082:00h 结束速度（End velocity）"
        
        Returns:
            关键词列表，例如 ["结束速度", "End velocity"]
        """
        if not text:
            return []
        
        keywords = []
        
        # 提取中文关键词（括号前的中文部分）
        chinese_match = ChapterIdentifierExtractor.CHINESE_KEYWORD_PATTERN.search(text)
        if chinese_match:
            chinese_keyword = chinese_match.group(1).strip()
            # 移除对象字典编号和子索引（如 "0x6082:00h"）
            chinese_keyword = re.sub(r'0x[0-9A-Fa-f]+:?\d*h?\s*', '', chinese_keyword, flags=re.IGNORECASE)
            chinese_keyword = chinese_keyword.strip()
            # 移除章节编号（如 "8.2.100."）
            chinese_keyword = re.sub(r'^\d+(?:\.\d+)*\.\s*', '', chinese_keyword)
            chinese_keyword = chinese_keyword.strip()
            if chinese_keyword:
                keywords.append(chinese_keyword)
        
        # 提取英文关键词（括号内的英文部分）
        english_match = ChapterIdentifierExtractor.ENGLISH_KEYWORD_PATTERN.search(text)
        if english_match:
            english_keyword = english_match.group(1).strip()
            if english_keyword:
                keywords.append(english_keyword)
        
        return keywords
    
    @staticmethod
    def extract_identifiers(text: str) -> Dict[str, any]:
        """
        从文本中提取所有标识符（对象字典编号和关键词）
        
        Args:
            text: 章节文本，例如 "8.2.100. 0x6082:00h 结束速度（End velocity）"
        
        Returns:
            包含 object_dict_id 和 keywords 的字典
        """
        object_dict_id = ChapterIdentifierExtractor.extract_object_dict_id(text)
        keywords = ChapterIdentifierExtractor.extract_keywords(text)
        
        return {
            "object_dict_id": object_dict_id,
            "keywords": keywords
        }
    
    @staticmethod
    def extract_from_important_keywords(important_keywords: List[str]) -> Dict[str, any]:
        """
        从 important_keywords 列表中提取标识符
        
        Args:
            important_keywords: important_keywords 列表，例如：
                ["完整路径 > 章节", "文档名字", "8.2.100. 0x6082:00h 结束速度（End velocity）"]
                或
                ["文档名字", "大章", "小章"]
        
        Returns:
            包含 object_dict_id 和 keywords 的字典
        """
        if not important_keywords:
            return {"object_dict_id": None, "keywords": []}
        
        # 检测格式：新格式（包含 " > "）或旧格式
        is_new_format = len(important_keywords) >= 3 and " > " in str(important_keywords[0])
        
        # 新格式：从第三个元素（索引2）提取章节信息
        # 旧格式：从第二个或第三个元素提取
        if is_new_format and len(important_keywords) >= 3:
            chapter_text = important_keywords[2]
        elif len(important_keywords) >= 2:
            chapter_text = important_keywords[1]  # 可能是大章
            # 如果有第三个元素，也尝试提取
            if len(important_keywords) >= 3:
                minor_chapter_text = important_keywords[2]
                # 如果小章包含更多信息（如对象字典编号），优先使用小章
                if ChapterIdentifierExtractor.extract_object_dict_id(minor_chapter_text):
                    chapter_text = minor_chapter_text
        else:
            chapter_text = important_keywords[0] if important_keywords else ""
        
        return ChapterIdentifierExtractor.extract_identifiers(str(chapter_text))
    
    @staticmethod
    def extract_from_content(content: str) -> Dict[str, any]:
        """
        从内容文本中提取标识符（备用方法）
        
        Args:
            content: 章节内容文本
        
        Returns:
            包含 object_dict_id 和 keywords 的字典
        """
        if not content:
            return {"object_dict_id": None, "keywords": []}
        
        # 尝试从内容的第一行或标题中提取
        lines = content.split('\n')
        for line in lines[:5]:  # 只检查前5行
            if line.strip():
                identifiers = ChapterIdentifierExtractor.extract_identifiers(line)
                if identifiers.get("object_dict_id") or identifiers.get("keywords"):
                    return identifiers
        
        return {"object_dict_id": None, "keywords": []}






