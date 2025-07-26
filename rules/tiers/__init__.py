"""
分层规则实现模块
包含不同层级的推理规则实现
"""

from rules.tiers.tier1_axioms import (
    ModusPonensRule,
    ModusTollensRule,
    HypotheticalSyllogismRule,
    DisjunctiveSyllogismRule,
    ConjunctionIntroductionRule,
    ConjunctionEliminationRule
)

# Tier 1 规则注册器
TIER1_RULES = [
    ModusPonensRule,
    ModusTollensRule,
    HypotheticalSyllogismRule,
    DisjunctiveSyllogismRule,
    ConjunctionIntroductionRule,
    ConjunctionEliminationRule
]


def get_tier1_rules():
    """获取所有Tier 1规则实例"""
    return [rule_class() for rule_class in TIER1_RULES]


__all__ = [
    'ModusPonensRule',
    'ModusTollensRule',
    'HypotheticalSyllogismRule',
    'DisjunctiveSyllogismRule',
    'ConjunctionIntroductionRule',
    'ConjunctionEliminationRule',
    'TIER1_RULES',
    'get_tier1_rules'
]
