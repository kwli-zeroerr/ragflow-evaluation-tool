"""
格式提取器
从 important_keywords 中提取章节信息，支持大章（顿号格式）和小章（句号空格格式）
"""

import re
import logging
from typing import Optional
from .rules_manager import RuleManager

logger = logging.getLogger(__name__)


class FormatExtractor:
    """格式提取器 - 从文本中提取章节信息"""
    
    def __init__(self, rule_manager: RuleManager):
        """
        初始化格式提取器
        
        Args:
            rule_manager: 规则管理器实例
        """
        self.rule_manager = rule_manager
        self.major_chapter_pattern: Optional[str] = None
        self.minor_chapter_pattern: Optional[str] = None
        self._load_format_patterns()
    
    def _load_format_patterns(self):
        """加载格式规则"""
        try:
            chapter_formats = self.rule_manager.get_extraction_rule("chapter_formats", {})
            if isinstance(chapter_formats, dict):
                major_config = chapter_formats.get("major_chapter", {})
                minor_config = chapter_formats.get("minor_chapter", {})
                
                self.major_chapter_pattern = major_config.get("pattern") if isinstance(major_config, dict) else None
                self.minor_chapter_pattern = minor_config.get("pattern") if isinstance(minor_config, dict) else None
        except Exception as e:
            logger.warning(f"加载格式规则失败: {str(e)}，将使用默认模式")
            self.major_chapter_pattern = None
            self.minor_chapter_pattern = None
    
    def extract_major_chapter(self, text: str) -> Optional[str]:
        """
        提取大章：匹配 [大章节][顿号] 格式
        例如："二十六、故障排查说明" -> "二十六"
        
        Args:
            text: 包含大章信息的文本
            
        Returns:
            提取的大章编号，如果无法提取则返回 None
        """
        if not text:
            return None
        
        text_str = str(text).strip()
        
        # 使用配置的模式
        if self.major_chapter_pattern:
            try:
                match = re.match(self.major_chapter_pattern, text_str)
                if match:
                    # 提取第一个捕获组（大章编号）
                    chapter = match.group(1) if match.groups() else match.group(0)
                    logger.debug(f"提取大章: '{text_str}' -> '{chapter}'")
                    return chapter.strip()
            except Exception as e:
                logger.warning(f"使用配置模式提取大章失败: {str(e)}")
        
        # 降级方案：使用默认模式
        # 匹配格式：[大章节][顿号]，例如 "二十六、故障排查说明"
        default_pattern = r'^([^、]+)、'
        match = re.match(default_pattern, text_str)
        if match:
            chapter = match.group(1).strip()
            logger.debug(f"提取大章（默认模式）: '{text_str}' -> '{chapter}'")
            return chapter
        
        return None
    
    def extract_minor_chapter(self, text: str) -> Optional[str]:
        """
        提取小章：匹配 [完整章节][句号][空格] 格式
        例如："26.3. 电机堵转报错的排查步骤" -> "26.3"
        
        Args:
            text: 包含小章信息的文本
            
        Returns:
            提取的小章编号，如果无法提取则返回 None
        """
        if not text:
            return None
        
        text_str = str(text).strip()
        
        # 使用配置的模式
        if self.minor_chapter_pattern:
            try:
                match = re.match(self.minor_chapter_pattern, text_str)
                if match:
                    # 提取第一个捕获组（小章编号）
                    chapter = match.group(1) if match.groups() else match.group(0)
                    logger.debug(f"提取小章: '{text_str}' -> '{chapter}'")
                    return chapter.strip()
            except Exception as e:
                logger.warning(f"使用配置模式提取小章失败: {str(e)}")
        
        # 降级方案：使用默认模式
        # 匹配格式：[完整章节][句号][空格]，例如 "26.3. 电机堵转报错的排查步骤"
        default_pattern = r'^(\d+(?:\.\d+)+)\.\s+'
        match = re.match(default_pattern, text_str)
        if match:
            chapter = match.group(1).strip()
            logger.debug(f"提取小章（默认模式）: '{text_str}' -> '{chapter}'")
            return chapter
        
        return None
    
    def extract_chapter(self, text: str, prefer_minor: bool = True) -> Optional[str]:
        """
        提取章节信息（优先小章，如果没有则提取大章）
        
        Args:
            text: 包含章节信息的文本
            prefer_minor: 是否优先提取小章
            
        Returns:
            提取的章节编号
        """
        if not text:
            return None
        
        if prefer_minor:
            # 优先提取小章
            minor_chapter = self.extract_minor_chapter(text)
            if minor_chapter:
                return minor_chapter
            
            # 如果没有小章，提取大章
            major_chapter = self.extract_major_chapter(text)
            if major_chapter:
                return major_chapter
        else:
            # 优先提取大章
            major_chapter = self.extract_major_chapter(text)
            if major_chapter:
                return major_chapter
            
            # 如果没有大章，提取小章
            minor_chapter = self.extract_minor_chapter(text)
            if minor_chapter:
                return minor_chapter
        
        return None

