# 文件：rules/rules_pool.py
# 说明：规则池模块，包含多个规则模板，支持复杂 DAG 构建

import random

from .balanced_and_rule import BalancedAndRule
from .multi_branch_and_rule import MultiBranchAndRule
from .simple_and_rule import SimpleAndRule
from .nested_and_rule import NestedAndRule


class RulePool:
    def __init__(self):
        # 注册所有可用规则
        self.rules = [
            SimpleAndRule(),  # 基础二元合取规则 A ∧ B => And(A, B)
            NestedAndRule(),  # 嵌套多元合取规则 A ∧ B ∧ ... => 嵌套 And(...)
            BalancedAndRule(),
            MultiBranchAndRule(max_branching=4),
        ]

    def sample_rule_for_conclusion(self, conclusion_expr=None, goal_only=False):
        """
        给定一个结论表达式，从规则池中选一条规则用于反向展开
        如果 goal_only=True 则忽略 conclusion_expr，直接采样一个规则用于构造初始目标
        """
        if goal_only or conclusion_expr is None:
            return random.choice(self.rules)

        # 将来可扩展为根据 expr 类型选择规则
        return random.choice(self.rules)

    def sample_rules(self, max_rules=None):
        """
        从规则池中随机采样若干规则（用于 DAG 多规则组合构建）
        """
        if max_rules is None:
            max_rules = len(self.rules)
        num_rules = random.randint(1, max_rules)
        return random.sample(self.rules, num_rules)

    def sample_rule(self):
        """
        从规则池中随机抽取一个规则（用于构造初始目标表达式）
        """
        return random.choice(self.rules)


# 创建默认规则池实例供外部使用
rule_pool = RulePool()
