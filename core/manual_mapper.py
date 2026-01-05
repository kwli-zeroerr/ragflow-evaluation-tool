"""
手册映射器
处理手册名称、版本、格式的映射
"""

import re
import logging
from typing import Optional, Dict, Any
from .rules_manager import RuleManager

logger = logging.getLogger(__name__)


class ManualMapper:
    """手册映射器 - 处理手册名称、版本、格式的映射"""
    
    def __init__(self, rule_manager: RuleManager):
        """
        初始化手册映射器
        
        Args:
            rule_manager: 规则管理器实例
        """
        self.rule_manager = rule_manager
        self.manual_mapping: Dict[str, Dict[str, Any]] = {}
        self.reverse_mapping: Dict[str, str] = {}  # 反向映射（新名称 → 旧名称）
        self.format_patterns: Dict[str, str] = {}
        self.auto_detect: bool = True
        self._load_manual_mapping()
    
    def _load_manual_mapping(self):
        """加载手册映射配置"""
        try:
            # 从规则管理器获取手册映射配置
            manual_mapping_data = self.rule_manager.get_manual_mapping()
            if isinstance(manual_mapping_data, dict) and manual_mapping_data:
                self.manual_mapping = manual_mapping_data.get("mappings", {})
                # 移除注释字段
                if isinstance(self.manual_mapping, dict):
                    self.manual_mapping = {k: v for k, v in self.manual_mapping.items() 
                                          if not k.startswith("_")}
                else:
                    self.manual_mapping = {}
                
                # 构建反向映射（新名称 → 旧名称）
                self.reverse_mapping = {}
                for old_key, mapping in self.manual_mapping.items():
                    if isinstance(mapping, dict):
                        new_name = mapping.get("new_name")
                        if new_name:
                            self.reverse_mapping[new_name] = old_key
                
                # 加载反向映射配置（如果配置了）
                reverse_mapping_data = manual_mapping_data.get("reverse_mapping", {})
                if isinstance(reverse_mapping_data, dict):
                    for k, v in reverse_mapping_data.items():
                        if not k.startswith("_"):
                            self.reverse_mapping[k] = v
                
                self.format_patterns = manual_mapping_data.get("format_patterns", {})
                # 移除注释字段
                if isinstance(self.format_patterns, dict):
                    self.format_patterns = {k: v for k, v in self.format_patterns.items() 
                                           if not k.startswith("_")}
                else:
                    self.format_patterns = {}
                
                # 获取自动识别配置
                self.auto_detect = manual_mapping_data.get("auto_detect", True)
                
                logger.info(f"加载手册映射规则: {len(self.manual_mapping)} 条映射，{len(self.reverse_mapping)} 条反向映射")
            else:
                logger.info("手册映射配置为空，将使用默认逻辑")
                self.manual_mapping = {}
                self.reverse_mapping = {}
                self.format_patterns = {}
                self.auto_detect = True
        except Exception as e:
            logger.warning(f"加载手册映射配置失败: {str(e)}，将使用默认逻辑")
            self.manual_mapping = {}
            self.reverse_mapping = {}
            self.format_patterns = {}
            self.auto_detect = True
    
    def detect_theme_format(self, theme: str) -> str:
        """
        智能识别手册格式（旧格式/新格式）
        
        Args:
            theme: 手册名称
            
        Returns:
            "old" 或 "new"，如果无法识别则返回 "unknown"
        """
        if not theme:
            return "unknown"
        
        # 检查是否在正向映射表中（旧格式）
        if theme in self.manual_mapping:
            return "old"
        
        # 检查是否在反向映射表中（新格式）
        if theme in self.reverse_mapping:
            return "new"
        
        # 检查是否在映射值中（可能是新格式）
        for mapping in self.manual_mapping.values():
            if isinstance(mapping, dict):
                if mapping.get("new_name") == theme:
                    return "new"
                aliases = mapping.get("aliases", [])
                if theme in aliases:
                    return "old"  # 别名通常对应旧格式
        
        # 如果启用了自动识别，尝试根据格式后缀判断
        if self.auto_detect:
            if theme.endswith('.md'):
                return "old"  # 假设 .md 是旧格式
            elif theme.endswith('.docx'):
                return "new"  # 假设 .docx 是新格式
        
        return "unknown"
    
    def needs_mapping(self, theme: str, target_format: str = "new") -> bool:
        """
        判断是否需要映射
        
        Args:
            theme: 手册名称
            target_format: 目标格式（"old" 或 "new"）
            
        Returns:
            如果需要映射则返回 True
        """
        if not theme:
            return False
        
        detected_format = self.detect_theme_format(theme)
        
        if target_format == "new":
            # 目标是新格式，如果检测到是旧格式，需要映射
            return detected_format == "old"
        elif target_format == "old":
            # 目标是旧格式，如果检测到是新格式，需要反向映射
            return detected_format == "new"
        
        return False
    
    def map_theme(self, old_theme: str, target_format: str = "new") -> str:
        """
        映射手册名称（支持双向映射）
        
        Args:
            old_theme: 手册名称（可能包含格式后缀，如 .md 或 .docx）
            target_format: 目标格式（"new" 表示映射到新格式，"old" 表示映射到旧格式）
            
        Returns:
            映射后的手册名称，如果映射不存在则返回原值
        """
        if not old_theme:
            return old_theme
        
        # 如果不需要映射，直接返回
        if not self.needs_mapping(old_theme, target_format):
            return old_theme
        
        # 正向映射（旧格式 → 新格式）
        if target_format == "new":
            # 精确匹配
            if old_theme in self.manual_mapping:
                mapping = self.manual_mapping[old_theme]
                new_name = mapping.get("new_name", old_theme)
                logger.debug(f"手册映射（正向）: '{old_theme}' -> '{new_name}'")
                return new_name
            
            # 尝试通过别名匹配
            for old_key, mapping in self.manual_mapping.items():
                aliases = mapping.get("aliases", [])
                if old_theme in aliases:
                    new_name = mapping.get("new_name", old_theme)
                    logger.debug(f"手册映射（通过别名）: '{old_theme}' -> '{new_name}'")
                    return new_name
            
            # 尝试去除格式后缀后匹配
            theme_without_format = self._remove_format_suffix(old_theme)
            if theme_without_format != old_theme and theme_without_format in self.manual_mapping:
                mapping = self.manual_mapping[theme_without_format]
                new_name = mapping.get("new_name", old_theme)
                logger.debug(f"手册映射（去除格式）: '{old_theme}' -> '{new_name}'")
                return new_name
        
        # 反向映射（新格式 → 旧格式）
        elif target_format == "old":
            if old_theme in self.reverse_mapping:
                old_name = self.reverse_mapping[old_theme]
                logger.debug(f"手册映射（反向）: '{old_theme}' -> '{old_name}'")
                return old_name
            
            # 尝试从正向映射的值中查找
            for old_key, mapping in self.manual_mapping.items():
                if isinstance(mapping, dict):
                    new_name = mapping.get("new_name")
                    if new_name == old_theme:
                        logger.debug(f"手册映射（反向查找）: '{old_theme}' -> '{old_key}'")
                        return old_key
        
        # 如果没有映射，返回原值
        return old_theme
    
    def get_format(self, theme: str) -> Optional[str]:
        """
        获取手册格式（markdown/docx）
        
        Args:
            theme: 手册名称（可能包含格式后缀）
            
        Returns:
            格式类型（'markdown' 或 'docx'），如果无法识别则返回 None
        """
        if not theme:
            return None
        
        # 检查映射配置中是否有格式信息
        if theme in self.manual_mapping:
            mapping = self.manual_mapping[theme]
            format_type = mapping.get("format")
            if format_type:
                return format_type
        
        # 通过文件名后缀识别格式
        for format_type, pattern in self.format_patterns.items():
            if re.search(pattern, theme):
                return format_type
        
        # 默认通过后缀判断
        if theme.endswith('.md'):
            return 'markdown'
        elif theme.endswith('.docx'):
            return 'docx'
        
        return None
    
    def normalize_theme(self, theme: str) -> Optional[str]:
        """
        标准化手册名称（去除版本号、格式后缀等）
        
        Args:
            theme: 手册名称
            
        Returns:
            标准化后的手册名称
        """
        if not theme:
            return None
        
        # 去除格式后缀
        normalized = self._remove_format_suffix(theme)
        
        # 去除版本号（如 v1.0, v2.0 等）
        normalized = re.sub(r'\s*v\d+\.\d+.*$', '', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\s*版本\d+.*$', '', normalized)
        
        return normalized.strip()
    
    def _remove_format_suffix(self, theme: str) -> str:
        """移除格式后缀（.md, .docx 等）"""
        # 移除常见的格式后缀
        for suffix in ['.md', '.docx', '.doc', '.txt', '.pdf']:
            if theme.endswith(suffix):
                return theme[:-len(suffix)]
        return theme
    
    def get_aliases(self, theme: str) -> list:
        """
        获取手册的所有别名（包括中英文）
        
        Args:
            theme: 手册名称
            
        Returns:
            别名列表
        """
        if not theme:
            return []
        
        # 查找映射配置
        for old_key, mapping in self.manual_mapping.items():
            if old_key == theme or theme in mapping.get("aliases", []):
                aliases = mapping.get("aliases", [])
                # 包含原始名称和新名称
                result = [old_key, mapping.get("new_name", "")]
                result.extend(aliases)
                # 去重并过滤空值
                return list(set([a for a in result if a]))
        
        return [theme]

