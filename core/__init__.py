"""
核心模块
包含规则管理器、章节映射器、手册映射器、格式提取器、标识符提取器和标识符匹配器
"""

from .rules_manager import RuleManager
from .chapter_mapper import ChapterMapper
from .manual_mapper import ManualMapper
from .format_extractor import FormatExtractor
from .chapter_identifier_extractor import ChapterIdentifierExtractor
from .chapter_identifier_matcher import ChapterIdentifierMatcher

__all__ = [
    'RuleManager', 
    'ChapterMapper', 
    'ManualMapper', 
    'FormatExtractor',
    'ChapterIdentifierExtractor',
    'ChapterIdentifierMatcher'
]

