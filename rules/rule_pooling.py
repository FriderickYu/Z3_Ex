from typing import Dict, List, Optional, Set
from collections import defaultdict
import random
from .base.rule import BaseRule, RuleType, LogicalFormula

# 修复相对导入问题
try:
    from ..utils.logger_utils import ARNGLogger
except ValueError:
    # 如果相对导入失败，使用绝对导入
    from utils.logger_utils import ARNGLogger


class TierConfig:
    """层级配置"""

    def __init__(self, tier: int, max_rules: int, complexity_range: tuple):
        self.tier = tier
        self.max_rules = max_rules
        self.complexity_range = complexity_range
        self.weight = 1.0  # 选择权重


class StratifiedRulePool:
    """
    分层规则池 - 实现双层架构中的规则管理
    支持按层级组织规则，控制复杂度递增
    """

    def __init__(self):
        self.logger = ARNGLogger("RulePool")

        # 分层存储规则
        self.rules_by_tier: Dict[int, List[BaseRule]] = defaultdict(list)
        self.rules_by_type: Dict[RuleType, List[BaseRule]] = defaultdict(list)
        self.rules_by_id: Dict[str, BaseRule] = {}

        # 层级配置
        self.tier_configs = {
            1: TierConfig(1, 10, (1, 3)),  # 基础公理
            2: TierConfig(2, 15, (2, 5)),  # 基础推理
            3: TierConfig(3, 20, (3, 7)),  # 复合规则
            4: TierConfig(4, 12, (4, 8)),  # 量词逻辑
            5: TierConfig(5, 8, (5, 10))  # 算术逻辑
        }

        # 选择策略
        self.selection_strategy = "balanced"  # balanced, complexity_driven, random

        self.logger.info("分层规则池初始化完成")

    def register_rule(self, rule: BaseRule) -> bool:
        """注册规则到规则池"""
        if rule.rule_id in self.rules_by_id:
            self.logger.warning(f"规则 {rule.rule_id} 已存在，跳过注册")
            return False

        # 检查层级限制
        tier_config = self.tier_configs.get(rule.tier)
        if not tier_config:
            self.logger.error(f"无效的规则层级: {rule.tier}")
            return False

        if len(self.rules_by_tier[rule.tier]) >= tier_config.max_rules:
            self.logger.warning(f"层级 {rule.tier} 已达到最大规则数 {tier_config.max_rules}")
            return False

        # 注册规则
        self.rules_by_tier[rule.tier].append(rule)
        self.rules_by_type[rule.rule_type].append(rule)
        self.rules_by_id[rule.rule_id] = rule

        self.logger.info(f"规则注册成功: {rule.rule_id} (层级: {rule.tier})")
        return True

    def get_applicable_rules(self, premises: List[LogicalFormula],
                             tier_limit: Optional[int] = None) -> List[BaseRule]:
        """获取可应用于给定前提的规则"""
        applicable_rules = []
        max_tier = tier_limit or max(self.tier_configs.keys())

        for tier in range(1, max_tier + 1):
            for rule in self.rules_by_tier[tier]:
                try:
                    if rule.can_apply(premises):
                        applicable_rules.append(rule)
                except Exception as e:
                    self.logger.error(f"检查规则 {rule.rule_id} 适用性时出错: {e}")

        self.logger.debug(f"找到 {len(applicable_rules)} 个可应用规则")
        return applicable_rules

    def select_rule(self, applicable_rules: List[BaseRule],
                    target_complexity: Optional[int] = None) -> Optional[BaseRule]:
        """根据策略选择规则"""
        if not applicable_rules:
            return None

        if self.selection_strategy == "random":
            return random.choice(applicable_rules)

        elif self.selection_strategy == "complexity_driven":
            if target_complexity:
                # 选择最接近目标复杂度的规则
                def complexity_distance(rule):
                    return abs(rule.get_complexity_score() - target_complexity)

                return min(applicable_rules, key=complexity_distance)
            else:
                # 选择最高复杂度的规则
                return max(applicable_rules, key=lambda r: r.get_complexity_score())

        elif self.selection_strategy == "balanced":
            # 平衡选择：考虑复杂度和成功率
            def balanced_score(rule):
                complexity_score = rule.get_complexity_score() / 100  # 归一化
                success_score = rule.success_rate if rule.usage_count > 0 else 0.5
                return complexity_score * 0.7 + success_score * 0.3

            return max(applicable_rules, key=balanced_score)

        return random.choice(applicable_rules)

    def get_rules_by_tier(self, tier: int) -> List[BaseRule]:
        """获取指定层级的所有规则"""
        return self.rules_by_tier.get(tier, [])

    def get_rules_by_type(self, rule_type: RuleType) -> List[BaseRule]:
        """获取指定类型的所有规则"""
        return self.rules_by_type.get(rule_type, [])

    def get_rule_by_id(self, rule_id: str) -> Optional[BaseRule]:
        """通过ID获取规则"""
        return self.rules_by_id.get(rule_id)

    def get_statistics(self) -> Dict[str, any]:
        """获取规则池统计信息"""
        stats = {
            "total_rules": len(self.rules_by_id),
            "rules_by_tier": {tier: len(rules) for tier, rules in self.rules_by_tier.items()},
            "rules_by_type": {rule_type.value: len(rules) for rule_type, rules in self.rules_by_type.items()},
            "average_success_rate": 0.0,
            "most_used_rule": None,
            "least_used_rule": None
        }

        if self.rules_by_id:
            total_success = sum(rule.success_rate for rule in self.rules_by_id.values())
            stats["average_success_rate"] = total_success / len(self.rules_by_id)

            used_rules = [rule for rule in self.rules_by_id.values() if rule.usage_count > 0]
            if used_rules:
                stats["most_used_rule"] = max(used_rules, key=lambda r: r.usage_count).rule_id
                stats["least_used_rule"] = min(used_rules, key=lambda r: r.usage_count).rule_id

        return stats

    def set_selection_strategy(self, strategy: str):
        """设置规则选择策略"""
        valid_strategies = ["balanced", "complexity_driven", "random"]
        if strategy in valid_strategies:
            self.selection_strategy = strategy
            self.logger.info(f"规则选择策略设置为: {strategy}")
        else:
            self.logger.error(f"无效的选择策略: {strategy}. 有效选项: {valid_strategies}")