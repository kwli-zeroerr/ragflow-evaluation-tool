"""
规则管理器
统一加载和管理所有规则配置，支持热更新
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RuleManager:
    """规则管理器 - 统一加载和管理所有规则配置"""
    
    def __init__(self, rules_dir: Path = None):
        """
        初始化规则管理器
        
        Args:
            rules_dir: 规则配置目录路径，默认为项目根目录下的 "rules" 目录
        """
        if rules_dir is None:
            # 默认使用项目根目录下的 rules 目录
            # __file__ 在 core/ 目录下，需要向上两级到项目根目录
            rules_dir = Path(__file__).parent.parent / "rules"
        
        self.rules_dir = Path(rules_dir)
        self.rules_dir.mkdir(exist_ok=True)
        
        # 规则数据
        self.chapter_mapping: Dict[str, str] = {}
        self.chapter_mapping_config: Dict[str, Any] = {}  # 完整的章节映射配置（包含反向映射等）
        self.matching_rules: Dict[str, Any] = {}
        self.extraction_rules: Dict[str, Any] = {}
        self.manual_mapping: Dict[str, Any] = {}
        self.identifier_mappings: Dict[str, Dict[str, Any]] = {}  # 标识符映射规则
        
        # 加载所有规则
        self._load_all_rules()
    
    def _load_all_rules(self):
        """加载所有规则配置"""
        try:
            self._load_chapter_mapping()
            self._load_matching_rules()
            self._load_extraction_rules()
            self._load_manual_mapping()
            self._load_identifier_mapping()
        except Exception as e:
            logger.warning(f"规则配置加载失败，将使用默认硬编码逻辑: {str(e)}")
    
    def _load_chapter_mapping(self):
        """加载章节映射规则"""
        mapping_file = self.rules_dir / "chapter_mapping.json"
        
        if not mapping_file.exists():
            logger.info(f"章节映射文件不存在: {mapping_file}，使用空映射")
            self.chapter_mapping = {}
            return
        
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 保存完整配置
                self.chapter_mapping_config = {k: v for k, v in data.items() 
                                              if not k.startswith("_")}
                # 提取正向映射
                self.chapter_mapping = data.get("mappings", {})
                # 移除注释字段
                self.chapter_mapping = {k: v for k, v in self.chapter_mapping.items() 
                                       if not k.startswith("_")}
                # 只在有映射时才输出日志
                if len(self.chapter_mapping) > 0:
                    logger.info(f"加载章节映射规则: {len(self.chapter_mapping)} 条映射")
        except json.JSONDecodeError as e:
            logger.error(f"章节映射文件 JSON 格式错误: {str(e)}")
            self.chapter_mapping = {}
        except Exception as e:
            logger.error(f"加载章节映射规则失败: {str(e)}")
            self.chapter_mapping = {}
    
    def _load_matching_rules(self):
        """加载匹配规则"""
        matching_file = self.rules_dir / "matching_rules.json"
        
        if not matching_file.exists():
            logger.info(f"匹配规则文件不存在: {matching_file}，使用默认规则")
            self.matching_rules = self._get_default_matching_rules()
            return
        
        try:
            with open(matching_file, 'r', encoding='utf-8') as f:
                self.matching_rules = json.load(f)
                # 移除注释字段
                self.matching_rules = {k: v for k, v in self.matching_rules.items() 
                                     if not k.startswith("_")}
        except json.JSONDecodeError as e:
            logger.error(f"匹配规则文件 JSON 格式错误: {str(e)}")
            self.matching_rules = self._get_default_matching_rules()
        except Exception as e:
            logger.error(f"加载匹配规则失败: {str(e)}")
            self.matching_rules = self._get_default_matching_rules()
    
    def _load_extraction_rules(self):
        """加载提取规则"""
        extraction_file = self.rules_dir / "extraction_rules.json"
        
        if not extraction_file.exists():
            logger.info(f"提取规则文件不存在: {extraction_file}，使用默认规则")
            self.extraction_rules = self._get_default_extraction_rules()
            return
        
        try:
            with open(extraction_file, 'r', encoding='utf-8') as f:
                self.extraction_rules = json.load(f)
                # 移除注释字段
                self.extraction_rules = {k: v for k, v in self.extraction_rules.items() 
                                       if not k.startswith("_")}
        except json.JSONDecodeError as e:
            logger.error(f"提取规则文件 JSON 格式错误: {str(e)}")
            self.extraction_rules = self._get_default_extraction_rules()
        except Exception as e:
            logger.error(f"加载提取规则失败: {str(e)}")
            self.extraction_rules = self._get_default_extraction_rules()
    
    def _get_default_matching_rules(self) -> Dict[str, Any]:
        """获取默认匹配规则（降级方案）"""
        return {
            "parent_child_rules": {
                "allow_parent_match_child": True,
                "allow_child_match_parent": False
            },
            "chinese_arabic_mapping": {
                "enabled": True,
                "custom_mappings": {}
            },
            "normalization_rules": {
                "remove_english_text": True,
                "remove_trailing_dots": True
            }
        }
    
    def _get_default_extraction_rules(self) -> Dict[str, Any]:
        """获取默认提取规则（降级方案）"""
        return {
            "patterns": {
                "numeric": r"^\d+(?:\.\d+)*",
                "chinese_chapter": r"第[一二三四五六七八九十\d]+章",
                "chinese_section": r"第[一二三四五六七八九十\d]+节"
            },
            "important_keywords_structure": {
                "book_name_index": 0,
                "major_chapter_index": 1,
                "minor_chapter_index": 2
            },
            "chinese_num_map": {
                "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
                "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
                "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
                "二十一": 21, "二十二": 22, "二十三": 23, "二十四": 24, "二十五": 25,
                "二十六": 26, "二十七": 27, "二十八": 28, "二十九": 29, "三十": 30,
                "百": 100, "千": 1000, "万": 10000
            }
        }
    
    def map_chapter(self, old_chapter: str) -> str:
        """
        映射旧章节到新章节
        
        Args:
            old_chapter: 旧章节编号
            
        Returns:
            新章节编号，如果映射不存在则返回原章节
        """
        if not old_chapter:
            return old_chapter
        
        # 精确匹配
        if old_chapter in self.chapter_mapping:
            return self.chapter_mapping[old_chapter]
        
        # 如果没有映射，返回原章节（fallback: keep_original）
        return old_chapter
    
    def get_chapter_mapping_config(self) -> Dict[str, Any]:
        """
        获取完整的章节映射配置（包含反向映射、自动识别等）
        
        Returns:
            章节映射配置字典
        """
        return self.chapter_mapping_config
    
    def get_matching_rule(self, key: str, default: Any = None) -> Any:
        """
        获取匹配规则配置项
        
        Args:
            key: 规则键，支持点号分隔的嵌套键，如 "parent_child_rules.allow_parent_match_child"
            default: 默认值
            
        Returns:
            规则值
        """
        keys = key.split(".")
        value = self.matching_rules
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_extraction_rule(self, key: str, default: Any = None) -> Any:
        """
        获取提取规则配置项
        
        Args:
            key: 规则键，支持点号分隔的嵌套键
            default: 默认值
            
        Returns:
            规则值
        """
        keys = key.split(".")
        value = self.extraction_rules
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def reload_rules(self):
        """重新加载规则（支持热更新）"""
        logger.info("重新加载规则配置...")
        self._load_all_rules()
        logger.info("规则配置重新加载完成")
    
    def get_chinese_num_map(self) -> Dict[str, int]:
        """获取中文数字映射表"""
        custom_map = self.get_extraction_rule("chinese_arabic_mapping.custom_mappings", {})
        default_map = self.get_extraction_rule("chinese_num_map", {})
        
        # 合并自定义映射和默认映射，自定义映射优先
        result = {**default_map, **custom_map}
        return result
    
    def _load_manual_mapping(self):
        """加载手册映射规则"""
        manual_file = self.rules_dir / "manual_mapping.json"
        
        if not manual_file.exists():
            logger.info(f"手册映射文件不存在: {manual_file}，使用空映射")
            self.manual_mapping = {}
            return
        
        try:
            with open(manual_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 移除注释字段
                self.manual_mapping = {k: v for k, v in data.items() 
                                     if not k.startswith("_")}
                # 同时添加到 extraction_rules 中，方便通过 get_extraction_rule 访问
                if "manual_mapping" not in self.extraction_rules:
                    self.extraction_rules["manual_mapping"] = data
                else:
                    self.extraction_rules["manual_mapping"] = data
                # 只在有映射时才输出日志
                mapping_count = len(data.get('mappings', {}))
                if mapping_count > 0:
                    logger.info(f"加载手册映射规则: {mapping_count} 条映射")
        except json.JSONDecodeError as e:
            logger.error(f"手册映射文件 JSON 格式错误: {str(e)}")
            self.manual_mapping = {}
        except Exception as e:
            logger.error(f"加载手册映射规则失败: {str(e)}")
            self.manual_mapping = {}
    
    def get_manual_mapping(self) -> Dict[str, Any]:
        """获取手册映射配置"""
        return self.manual_mapping
    
    def _load_identifier_mapping(self):
        """加载标识符映射规则"""
        identifier_file = self.rules_dir / "chapter_identifier_mapping.json"
        
        if not identifier_file.exists():
            logger.info(f"标识符映射文件不存在: {identifier_file}，使用空映射")
            self.identifier_mappings = {}
            return
        
        try:
            with open(identifier_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 提取 mappings 字段
                self.identifier_mappings = data.get("mappings", {})
                # 移除注释字段
                self.identifier_mappings = {k: v for k, v in self.identifier_mappings.items() 
                                          if not k.startswith("_")}
                # 只在有映射时才输出日志
                if len(self.identifier_mappings) > 0:
                    logger.info(f"加载标识符映射规则: {len(self.identifier_mappings)} 条映射")
        except json.JSONDecodeError as e:
            logger.error(f"标识符映射文件 JSON 格式错误: {str(e)}")
            self.identifier_mappings = {}
        except Exception as e:
            logger.error(f"加载标识符映射规则失败: {str(e)}")
            self.identifier_mappings = {}
    
    def get_identifier_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        获取标识符映射配置
        
        Returns:
            标识符映射字典，格式：
            {
                "8.2.98": {
                    "object_dict_id": "0x6082",
                    "keywords": ["结束速度", "End velocity"],
                    "equivalent_chapters": ["8.2.100"]
                }
            }
        """
        return self.identifier_mappings

