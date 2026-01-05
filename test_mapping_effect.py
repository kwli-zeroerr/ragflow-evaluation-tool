#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试映射规则对测试结果的影响
"""

import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.rules_manager import RuleManager
from core.chapter_identifier_extractor import ChapterIdentifierExtractor
from core.chapter_identifier_matcher import ChapterIdentifierMatcher
from evaluation_tool import ChapterMatcher


def try_read_csv(file_path: Path) -> Optional[List[Dict]]:
    """尝试使用多种编码读取CSV文件"""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception:
            continue
    
    return None


def extract_chapter_number(chapter_text: str) -> Optional[str]:
    """从章节文本中提取章节编号"""
    if not chapter_text:
        return None
    match = re.match(r'^(\d+(?:\.\d+)*)', chapter_text)
    if match:
        return match.group(1)
    return None


def main():
    """主函数"""
    print("=" * 80)
    print("测试映射规则对测试结果的影响")
    print("=" * 80)
    
    # 1. 加载映射规则
    rule_manager = RuleManager()
    identifier_mappings = rule_manager.get_identifier_mapping()
    identifier_matcher = ChapterIdentifierMatcher(identifier_mappings)
    
    print(f"\n1. 加载映射规则: {len(identifier_mappings)} 条")
    
    # 2. 加载测试结果
    results_file = project_root / "output" / "single" / "evaluation_results.csv"
    if not results_file.exists():
        print(f"\n错误: 未找到测试结果文件: {results_file}")
        return
    
    rows = try_read_csv(results_file)
    if not rows:
        print(f"\n错误: 无法读取测试结果文件")
        return
    
    print(f"2. 加载测试结果: {len(rows)} 条")
    
    # 3. 分析映射效果
    total_cases = 0
    mapped_cases = 0
    mapping_stats = {
        'object_dict_id_match': 0,      # 对象字典编号匹配
        'mapping_rule_match': 0,        # 映射规则匹配
        'keyword_match': 0,             # 关键词匹配
        'no_match': 0                   # 无法匹配
    }
    
    mapped_details = []
    
    print("\n3. 分析映射效果...")
    print("-" * 80)
    
    for i, row in enumerate(rows, 1):  # 测试全部案例
        reference = row.get('reference', '').strip()
        original_answer = row.get('original_answer', '').strip()
        answer_title = row.get('answer_title', '').strip()  # 完整章节标题
        answer = row.get('answer', '').strip()
        recall = float(row.get('recall', 0))
        
        if not reference or not original_answer:
            continue
        
        total_cases += 1
        
        # 使用与 evaluation_tool.py 相同的逻辑
        # 优先使用 answer_title（完整标题），如果没有则使用 original_answer（章节号）
        retrieved_chapter_for_mapping = answer_title if answer_title else original_answer
        
        # 使用 ChapterMatcher.get_mapped_chapter 方法，与 evaluation_tool.py 保持一致
        is_mapped, mapped_chapter_num = ChapterMatcher.get_mapped_chapter(
            retrieved_chapter_for_mapping,
            reference
        )
        
        # 判断是否匹配成功
        is_match = is_mapped
        mapped_chapter = mapped_chapter_num if is_mapped and mapped_chapter_num else None
        
        if is_match:
            mapped_cases += 1
            
            # 判断匹配方式（需要提取标识符用于统计）
            retrieved_identifiers = ChapterIdentifierExtractor.extract_identifiers(retrieved_chapter_for_mapping)
            reference_identifiers = ChapterIdentifierExtractor.extract_identifiers(reference)
            
            retrieved_obj_id = retrieved_identifiers.get('object_dict_id')
            reference_obj_id = reference_identifiers.get('object_dict_id')
            
            if retrieved_obj_id and reference_obj_id and retrieved_obj_id.upper() == reference_obj_id.upper():
                mapping_stats['object_dict_id_match'] += 1
                match_type = "对象字典编号匹配"
            else:
                # 检查是否在映射规则中
                reference_num = extract_chapter_number(reference)
                original_num = extract_chapter_number(original_answer)
                in_mapping = False
                for mapping_key, mapping_value in identifier_mappings.items():
                    mapping_obj_id = mapping_value.get('object_dict_id')
                    if mapping_obj_id:
                        if (retrieved_obj_id and retrieved_obj_id.upper() == mapping_obj_id.upper()) or \
                           (reference_obj_id and reference_obj_id.upper() == mapping_obj_id.upper()):
                            in_mapping = True
                            break
                
                if in_mapping:
                    mapping_stats['mapping_rule_match'] += 1
                    match_type = "映射规则匹配"
                else:
                    mapping_stats['keyword_match'] += 1
                    match_type = "关键词匹配"
            
            # 记录详细信息
            mapped_details.append({
                'test_index': row.get('test_index', ''),
                'reference': reference,
                'original_answer': original_answer,
                'mapped_chapter': mapped_chapter,
                'match_type': match_type,
                'recall': recall,
                'object_dict_id': retrieved_obj_id or reference_obj_id
            })
        else:
            mapping_stats['no_match'] += 1
    
    # 4. 显示统计结果
    print("\n" + "=" * 80)
    print("映射效果统计")
    print("=" * 80)
    print(f"\n总测试案例: {total_cases} 条")
    print(f"成功映射: {mapped_cases} 条 ({mapped_cases/total_cases*100:.1f}%)")
    print(f"无法映射: {mapping_stats['no_match']} 条 ({mapping_stats['no_match']/total_cases*100:.1f}%)")
    
    print(f"\n匹配方式统计:")
    print(f"  - 对象字典编号匹配: {mapping_stats['object_dict_id_match']} 条 ({mapping_stats['object_dict_id_match']/mapped_cases*100:.1f}%)" if mapped_cases > 0 else "  - 对象字典编号匹配: 0 条")
    print(f"  - 映射规则匹配: {mapping_stats['mapping_rule_match']} 条 ({mapping_stats['mapping_rule_match']/mapped_cases*100:.1f}%)" if mapped_cases > 0 else "  - 映射规则匹配: 0 条")
    print(f"  - 关键词匹配: {mapping_stats['keyword_match']} 条 ({mapping_stats['keyword_match']/mapped_cases*100:.1f}%)" if mapped_cases > 0 else "  - 关键词匹配: 0 条")
    
    # 5. 显示映射后的效果
    if mapped_details:
        print(f"\n映射后的章节变化（前10条）:")
        print("-" * 80)
        for detail in mapped_details[:10]:
            print(f"Test {detail['test_index']}:")
            print(f"  参考章节: {detail['reference']}")
            print(f"  原始答案: {detail['original_answer']}")
            print(f"  映射后: {detail['mapped_chapter']}")
            print(f"  匹配方式: {detail['match_type']}")
            print(f"  对象字典编号: {detail['object_dict_id'] or 'N/A'}")
            print(f"  Recall: {detail['recall']}")
            print()
    
    # 6. 分析映射前后的Recall变化
    if mapped_details:
        print("\n" + "=" * 80)
        print("映射前后Recall对比")
        print("=" * 80)
        
        recall_before_mapping = [float(row.get('recall', 0)) for row in rows if row.get('original_answer')]
        recall_after_mapping = []
        
        # 假设映射后，如果映射成功且mapped_chapter与reference匹配，则recall=1.0
        for detail in mapped_details:
            if detail['mapped_chapter']:
                reference_num = extract_chapter_number(detail['reference'])
                mapped_num = detail['mapped_chapter']
                if reference_num == mapped_num:
                    recall_after_mapping.append(1.0)
                else:
                    recall_after_mapping.append(0.0)
            else:
                recall_after_mapping.append(detail['recall'])
        
        if recall_before_mapping:
            avg_recall_before = sum(recall_before_mapping) / len(recall_before_mapping)
            print(f"\n映射前平均Recall: {avg_recall_before:.4f}")
        
        if recall_after_mapping:
            avg_recall_after = sum(recall_after_mapping) / len(recall_after_mapping)
            print(f"映射后平均Recall: {avg_recall_after:.4f}")
            
            if recall_before_mapping:
                improvement = avg_recall_after - avg_recall_before
                print(f"Recall提升: {improvement:+.4f} ({improvement/avg_recall_before*100:+.1f}%)" if avg_recall_before > 0 else "Recall提升: N/A")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

