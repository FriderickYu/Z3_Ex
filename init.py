"""
ARNG_Generator_v3 - Advanced Reasoning Graph Generator Version 3
基于演绎推理的双层DAG生成系统
"""

__version__ = "3.0.0"
__author__ = "YU TIANQI"

from core import (
    DualLayerDAGGenerator,
    DAGGenerationMode,
    ComplexityController,
    ComplexityProfile
)

from rules import (
    StratifiedRulePool,
    BaseRule,
    RuleType,
    LogicalFormula,
    get_tier1_rules
)

from utils import ARNGLogger


def create_default_generator():
    """创建默认配置的生成器"""
    rule_pool = StratifiedRulePool()

    tier1_rules = get_tier1_rules()
    for rule in tier1_rules:
        rule_pool.register_rule(rule)

    dag_generator = DualLayerDAGGenerator(rule_pool)
    complexity_controller = ComplexityController()

    return dag_generator, complexity_controller, rule_pool


def create_logger(name: str = "ARNG"):
    """创建日志器"""
    return ARNGLogger(name)


__all__ = [
    'DualLayerDAGGenerator',
    'DAGGenerationMode',
    'ComplexityController',
    'ComplexityProfile',
    'StratifiedRulePool',
    'BaseRule',
    'RuleType',
    'LogicalFormula',
    'ARNGLogger',
    'get_tier1_rules',
    'create_default_generator',
    'create_logger'
]