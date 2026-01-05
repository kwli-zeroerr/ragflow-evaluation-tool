"""
章节映射器
处理旧章节到新章节的映射逻辑，支持双向映射和格式识别
"""

import logging
from typing import Optional, Dict, Any
from functools import lru_cache
from .rules_manager import RuleManager

logger = logging.getLogger(__name__)


class ChapterMapper:
    """章节映射器 - 处理旧章节到新章节的映射，支持双向映射和格式识别"""
    
    def __init__(self, rule_manager: RuleManager):
        """
        初始化章节映射器
        
        Args:
            rule_manager: 规则管理器实例
        """
        self.rule_manager = rule_manager
        self.reverse_mapping: Dict[str, str] = {}
        self.auto_detect: bool = True
        self._load_mapping_config()
        # 清除缓存（如果之前有实例）
        self._clear_cache()
    
    def _load_mapping_config(self):
        """加载映射配置"""
        try:
            mapping_config = self.rule_manager.get_chapter_mapping_config()
            if isinstance(mapping_config, dict):
                # 获取反向映射（如果配置了）
                reverse_mapping_data = mapping_config.get("reverse_mapping", {})
                if isinstance(reverse_mapping_data, dict):
                    self.reverse_mapping = {k: v for k, v in reverse_mapping_data.items() 
                                           if not k.startswith("_")}
                
                # 获取自动识别配置
                self.auto_detect = mapping_config.get("auto_detect", True)
        except Exception as e:
            logger.warning(f"加载章节映射配置失败: {str(e)}，使用默认配置")
            self.reverse_mapping = {}
            self.auto_detect = True
    
    def _clear_cache(self):
        """清除缓存（当映射配置改变时调用）"""
        # 清除实例方法的缓存
        if hasattr(self._map_retrieved_chapter_cached, 'cache_clear'):
            self._map_retrieved_chapter_cached.cache_clear()
        if hasattr(self._map_reference_cached, 'cache_clear'):
            self._map_reference_cached.cache_clear()
    
    @lru_cache(maxsize=1000)
    def _map_retrieved_chapter_cached(self, chapter: str, target_format: str, 
                                     chapter_mapping_tuple: tuple, reverse_mapping_tuple: tuple) -> str:
        """
        缓存的章节映射方法（内部方法）
        
        Args:
            chapter: 章节编号
            target_format: 目标格式
            chapter_mapping_tuple: 章节映射表的元组形式（用于缓存键）
            reverse_mapping_tuple: 反向映射表的元组形式（用于缓存键）
        
        Returns:
            映射后的章节编号
        """
        if not chapter:
            return chapter
        
        # 恢复映射表（从元组转换回字典）
        chapter_mapping = dict(chapter_mapping_tuple)
        reverse_mapping = dict(reverse_mapping_tuple)
        
        # 检查是否需要映射
        detected_format = "unknown"
        if chapter in chapter_mapping:
            detected_format = "old"
        elif chapter in reverse_mapping:
            detected_format = "new"
        elif chapter in chapter_mapping.values():
            detected_format = "new"
        
        needs_mapping = False
        if target_format == "new":
            needs_mapping = detected_format == "old"
        elif target_format == "old":
            needs_mapping = detected_format == "new"
        
        if not needs_mapping:
            return chapter
        
        # 正向映射（旧格式 → 新格式）
        if target_format == "new":
            mapped = chapter_mapping.get(chapter, chapter)
            if mapped != chapter:
                return mapped
        
        # 反向映射（新格式 → 旧格式）
        elif target_format == "old":
            if chapter in reverse_mapping:
                return reverse_mapping[chapter]
            # 尝试从正向映射的值中查找
            for old_chapter, new_chapter in chapter_mapping.items():
                if new_chapter == chapter:
                    return old_chapter
        
        return chapter
    
    @lru_cache(maxsize=1000)
    def _map_reference_cached(self, reference: str, target_format: str,
                             chapter_mapping_tuple: tuple, reverse_mapping_tuple: tuple) -> str:
        """
        缓存的reference映射方法（内部方法）
        
        Args:
            reference: 原始 reference 字段
            target_format: 目标格式
            chapter_mapping_tuple: 章节映射表的元组形式（用于缓存键）
            reverse_mapping_tuple: 反向映射表的元组形式（用于缓存键）
        
        Returns:
            映射后的 reference 字段
        """
        if not reference:
            return reference
        
        # 恢复映射表（从元组转换回字典）
        chapter_mapping = dict(chapter_mapping_tuple)
        reverse_mapping = dict(reverse_mapping_tuple)
        
        # 检查是否需要映射
        detected_format = "unknown"
        if reference in chapter_mapping:
            detected_format = "old"
        elif reference in reverse_mapping:
            detected_format = "new"
        elif reference in chapter_mapping.values():
            detected_format = "new"
        
        needs_mapping = False
        if target_format == "new":
            needs_mapping = detected_format == "old"
        elif target_format == "old":
            needs_mapping = detected_format == "new"
        
        if not needs_mapping:
            return reference
        
        # 正向映射（旧格式 → 新格式）
        if target_format == "new":
            mapped = chapter_mapping.get(reference, reference)
            if mapped != reference:
                return mapped
        
        # 反向映射（新格式 → 旧格式）
        elif target_format == "old":
            if reference in reverse_mapping:
                return reverse_mapping[reference]
            # 尝试从正向映射的值中查找
            for old_chapter, new_chapter in chapter_mapping.items():
                if new_chapter == reference:
                    return old_chapter
        
        return reference
    
    def detect_chapter_format(self, chapter: str) -> str:
        """
        智能识别章节格式（旧格式/新格式）
        
        Args:
            chapter: 章节编号
            
        Returns:
            "old" 或 "new"，如果无法识别则返回 "unknown"
        """
        if not chapter:
            return "unknown"
        
        # 检查是否在正向映射表中（旧格式）
        if chapter in self.rule_manager.chapter_mapping:
            return "old"
        
        # 检查是否在反向映射表中（新格式）
        if chapter in self.reverse_mapping:
            return "new"
        
        # 检查是否在映射值中（可能是新格式）
        if chapter in self.rule_manager.chapter_mapping.values():
            return "new"
        
        # 如果启用了自动识别，尝试根据模式判断
        if self.auto_detect:
            # 这里可以根据实际需求添加更复杂的识别逻辑
            # 例如：检查章节格式是否符合新格式模式
            pass
        
        return "unknown"
    
    def needs_mapping(self, chapter: str, target_format: str = "new") -> bool:
        """
        判断是否需要映射
        
        Args:
            chapter: 章节编号
            target_format: 目标格式（"old" 或 "new"）
            
        Returns:
            如果需要映射则返回 True
        """
        if not chapter:
            return False
        
        detected_format = self.detect_chapter_format(chapter)
        
        if target_format == "new":
            # 目标是新格式，如果检测到是旧格式，需要映射
            return detected_format == "old"
        elif target_format == "old":
            # 目标是旧格式，如果检测到是新格式，需要反向映射
            return detected_format == "new"
        
        return False
    
    def map_reference(self, reference: str, target_format: str = "new") -> str:
        """
        映射测试用例的 reference 字段（标注章节）
        使用LRU缓存优化性能
        
        Args:
            reference: 原始 reference 字段（可能包含旧章节编号或新章节编号）
            target_format: 目标格式（"new" 表示映射到新格式，"old" 表示映射到旧格式）
            
        Returns:
            映射后的 reference 字段
        """
        if not reference:
            return reference
        
        # 将映射表转换为元组（用于缓存键）
        chapter_mapping_tuple = tuple(self.rule_manager.chapter_mapping.items())
        reverse_mapping_tuple = tuple(self.reverse_mapping.items())
        
        # 调用缓存方法
        mapped = self._map_reference_cached(reference, target_format, 
                                           chapter_mapping_tuple, reverse_mapping_tuple)
        
        # 记录映射日志（仅在映射成功时）
        if mapped != reference:
            logger.debug(f"章节映射（reference）: '{reference}' -> '{mapped}'")
        
        return mapped
    
    def map_retrieved_chapter(self, chapter: str, target_format: str = "new") -> str:
        """
        映射检索到的章节
        使用LRU缓存优化性能
        
        Args:
            chapter: 从检索结果中提取的章节编号
            target_format: 目标格式（"new" 表示映射到新格式，"old" 表示映射到旧格式）
            
        Returns:
            映射后的章节编号
        """
        if not chapter:
            return chapter
        
        # 将映射表转换为元组（用于缓存键）
        chapter_mapping_tuple = tuple(self.rule_manager.chapter_mapping.items())
        reverse_mapping_tuple = tuple(self.reverse_mapping.items())
        
        # 调用缓存方法
        mapped = self._map_retrieved_chapter_cached(chapter, target_format,
                                                    chapter_mapping_tuple, reverse_mapping_tuple)
        
        # 记录映射日志（仅在映射成功时）
        if mapped != chapter:
            logger.debug(f"检索章节映射: '{chapter}' -> '{mapped}'")
        
        return mapped
    
    def map_chapter(self, chapter: str, target_format: str = "new") -> str:
        """
        通用章节映射方法（别名方法，方便调用）
        
        Args:
            chapter: 章节编号
            target_format: 目标格式（"new" 表示映射到新格式，"old" 表示映射到旧格式）
            
        Returns:
            映射后的章节编号
        """
        return self.map_retrieved_chapter(chapter, target_format)

