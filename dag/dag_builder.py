# 文件：dag/dag_builder.py（更新版）
# 说明：构建支持多规则组合的逻辑推理 DAG，使用统一变量命名系统

import random
import logging
from copy import deepcopy
from collections import deque
from typing import List, Dict, Optional, Tuple

import z3
from rules.rules_pool import rule_pool
from utils.variable_manager import variable_manager, EnhancedVariableExtractor


class DAGNode:
    def __init__(self, z3_expr, rule=None, rule_name=None, depth=0):
        self.z3_expr = z3_expr
        self.rule = rule
        self.rule_name = rule_name or (rule.name if rule else None)
        self.children = []
        self.depth = depth

    def add_child(self, child_node):
        self.children.append(child_node)


class ShortChainDAGBuilder:
    """短链条DAG构建器（2-5步）：使用统一变量命名"""

    def __init__(self, max_depth=5, min_depth=2):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.node_counter = 0
        self.logger = logging.getLogger("short_chain_dag_builder")
        self.processing_expressions = set()
        self.max_recursion_depth = 6
        self.current_recursion_depth = 0

    def build(self):
        """构建短推理链"""
        try:
            self.logger.info(f"构建短推理链，深度范围: {self.min_depth}-{self.max_depth}")

            # 重置变量管理器
            variable_manager.reset()
            self._reset_state()

            # 选择最终规则
            final_rule = self._select_goal_rule()
            self.logger.info(f"选择最终规则: {final_rule.name}")

            # 限制目标深度
            target_depth = min(random.randint(self.min_depth, self.max_depth), 4)

            root = self._build_reasoning_chain(final_rule, target_depth)

            if root is None:
                self.logger.warning("构建失败，使用回退方案")
                return self._build_simple_fallback()

            actual_depth = self._calculate_depth(root)
            self.logger.info(f"成功构建短链条，实际深度: {actual_depth}")

            return root

        except RecursionError as e:
            self.logger.error(f"递归深度超限: {e}")
            return self._build_simple_fallback()
        except Exception as e:
            self.logger.error(f"构建短链条时出错: {e}")
            return self._build_simple_fallback()

    def _reset_state(self):
        """重置构建状态"""
        self.processing_expressions.clear()
        self.current_recursion_depth = 0

    def _select_goal_rule(self):
        """选择目标规则"""
        preferred_goal_rules = [
            "HypotheticalSyllogism",
            "ConjunctionIntroduction",
            "BiconditionalElimination"
        ]

        for _ in range(5):
            rule = rule_pool.sample_rule()
            if rule.name in preferred_goal_rules:
                return rule

        return rule_pool.sample_rule_for_conclusion(goal_only=True)

    def _build_reasoning_chain(self, final_rule, depth):
        """构建推理链"""
        try:
            num_final_premises = final_rule.num_premises()

            # 创建最终前提变量（使用统一命名）
            final_premises = []
            for i in range(num_final_premises):
                var = variable_manager.create_variable("Goal")
                final_premises.append(var)

            # 构造最终结论
            try:
                final_conclusion = final_rule.construct_conclusion(final_premises)
            except Exception as e:
                self.logger.debug(f"构造最终结论失败: {e}")
                final_conclusion = variable_manager.create_variable("FinalConclusion")

            # 创建根节点
            root = DAGNode(final_conclusion, final_rule, final_rule.name, 0)

            # 为每个前提构建子推理链
            for premise in final_premises:
                child_chain = self._build_premise_chain(premise, depth - 1)
                if child_chain:
                    root.add_child(child_chain)
                else:
                    leaf = DAGNode(premise, depth=depth)
                    root.add_child(leaf)

            return root

        except Exception as e:
            self.logger.error(f"构建推理链失败: {e}")
            return None

    def _build_premise_chain(self, target_premise, remaining_depth):
        """构建前提链（带递归保护）"""
        # 递归深度保护
        if (remaining_depth <= 0 or
                self.current_recursion_depth >= self.max_recursion_depth):
            return DAGNode(target_premise, depth=remaining_depth)

        # 循环检测保护
        expr_str = str(target_premise)
        if expr_str in self.processing_expressions:
            self.logger.debug(f"检测到循环引用: {expr_str}")
            return DAGNode(target_premise, depth=remaining_depth)

        self.processing_expressions.add(expr_str)
        self.current_recursion_depth += 1

        try:
            rule = self._select_rule_for_depth(remaining_depth)

            try:
                sub_premises = rule.generate_premises(target_premise)
            except Exception:
                # 使用统一变量命名创建前提
                sub_premises = []
                for _ in range(rule.num_premises()):
                    sub_premises.append(variable_manager.create_variable("SubPremise"))

            # 限制子前提数量
            if len(sub_premises) > 3:
                sub_premises = sub_premises[:3]

            if not self._can_apply_rule_safely(rule, sub_premises):
                return DAGNode(target_premise, depth=remaining_depth)

            try:
                intermediate_conclusion = rule.construct_conclusion(sub_premises)
                current_node = DAGNode(intermediate_conclusion, rule, rule.name, remaining_depth)
            except Exception:
                return DAGNode(target_premise, depth=remaining_depth)

            # 递归构建子前提
            for sub_premise in sub_premises:
                sub_expr_str = str(sub_premise)
                if sub_expr_str not in self.processing_expressions:
                    child_chain = self._build_premise_chain(sub_premise, remaining_depth - 1)
                    if child_chain:
                        current_node.add_child(child_chain)
                else:
                    leaf = DAGNode(sub_premise, depth=remaining_depth + 1)
                    current_node.add_child(leaf)

            return current_node

        except Exception as e:
            self.logger.debug(f"构建前提链失败: {e}")
            return DAGNode(target_premise, depth=remaining_depth)

        finally:
            self.processing_expressions.discard(expr_str)
            self.current_recursion_depth -= 1

    def _select_rule_for_depth(self, depth):
        """根据深度选择规则"""
        if depth <= 1:
            simple_rules = ["ModusPonens", "ConjunctionElimination", "UniversalInstantiation"]
        elif depth == 2:
            simple_rules = ["HypotheticalSyllogism", "TransitivityRule", "ModusPonens"]
        else:
            return rule_pool.sample_rule()

        for _ in range(5):
            rule = rule_pool.sample_rule()
            if rule.name in simple_rules:
                return rule
        return rule_pool.sample_rule()

    def _can_apply_rule_safely(self, rule, premises):
        """安全检查规则应用性"""
        try:
            if hasattr(rule, 'can_apply'):
                return rule.can_apply(premises)
            return True
        except Exception:
            return True

    def _calculate_depth(self, node):
        """计算DAG深度"""
        if not node.children:
            return 1
        max_child_depth = max(self._calculate_depth(child) for child in node.children)
        return max_child_depth + 1

    def _build_simple_fallback(self):
        """简单回退方案"""
        self.logger.info("使用简单回退方案")

        try:
            # 使用统一变量命名
            premise = variable_manager.create_variable("FallbackP")
            conclusion = variable_manager.create_variable("FallbackQ")

            from rules.modus_ponens import ModusPonensRule
            rule = ModusPonensRule()

            implication = z3.Implies(premise, conclusion)

            root = DAGNode(conclusion, rule, rule.name, 0)
            child1 = DAGNode(premise, depth=1)
            child2 = DAGNode(implication, depth=1)

            root.add_child(child1)
            root.add_child(child2)

            return root

        except Exception as e:
            self.logger.error(f"回退方案失败: {e}")
            var = variable_manager.create_variable("Emergency")
            return DAGNode(var)


class LongChainDAGBuilder:
    """长链条DAG构建器（5+步）：使用统一变量命名"""

    def __init__(self, max_depth=15, min_depth=5):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.logger = logging.getLogger("long_chain_dag_builder")
        self.used_expressions = set()

    def build_long_chain(self) -> Optional[DAGNode]:
        """构建长逻辑链条"""
        try:
            # 重置变量管理器
            variable_manager.reset()

            target_depth = random.randint(self.min_depth, self.max_depth)
            self.logger.info(f"构建长逻辑链，目标深度: {target_depth}")

            # 构建线性链条
            chain = self._build_linear_chain(target_depth)

            if not chain:
                return self._build_fallback_chain()

            # 转换为DAG结构
            root = self._convert_chain_to_dag(chain)

            actual_depth = len(chain)
            self.logger.info(f"成功构建长链条，实际深度: {actual_depth}")

            return root

        except Exception as e:
            self.logger.error(f"构建长链条失败: {e}")
            return self._build_fallback_chain()

    def _build_linear_chain(self, target_depth: int) -> List[Dict]:
        """构建线性逻辑链条"""
        chain = []
        current_conclusion = variable_manager.create_variable("FinalConclusion")

        for step in range(target_depth):
            rule = self._select_rule_for_step(step, target_depth)
            premises = self._generate_premises_for_rule(rule, current_conclusion, step)

            if not premises:
                self.logger.warning(f"第{step}步无法生成前提，停止构建")
                break

            chain_node = {
                'step': step,
                'rule': rule,
                'rule_name': rule.name,
                'conclusion': current_conclusion,
                'premises': premises,
                'depth': target_depth - step
            }

            chain.append(chain_node)

            # 选择下一个目标
            if len(premises) > 1:
                current_conclusion = self._select_next_conclusion(premises)
            else:
                current_conclusion = premises[0]

        return chain

    def _select_rule_for_step(self, step: int, total_depth: int):
        """为步骤选择规则"""
        if step == 0:  # 最终步骤
            preferred = ["HypotheticalSyllogism", "ModusPonens", "ConjunctionElimination"]
        elif step < total_depth // 3:  # 前期
            preferred = ["ConjunctionIntroduction", "DisjunctionIntroduction", "BiconditionalElimination"]
        elif step < 2 * total_depth // 3:  # 中期
            preferred = ["HypotheticalSyllogism", "TransitivityRule", "ModusPonens"]
        else:  # 后期
            preferred = ["UniversalInstantiation", "ConjunctionElimination", "ModusPonens"]

        # 尝试获取偏好规则
        for _ in range(5):
            rule = rule_pool.sample_rule()
            if rule.name in preferred:
                return rule

        return rule_pool.sample_rule()

    def _generate_premises_for_rule(self, rule, conclusion, step: int) -> List[z3.ExprRef]:
        """为规则生成前提"""
        try:
            if hasattr(rule, 'generate_premises'):
                premises = rule.generate_premises(conclusion)
                if premises and len(premises) <= 4:
                    return premises

            # 回退：使用统一变量命名生成新前提
            num_premises = min(rule.num_premises(), 3)
            premises = []

            for i in range(num_premises):
                var = variable_manager.create_variable(f"Step{step}_Premise{i}")
                premises.append(var)

            return premises

        except Exception as e:
            self.logger.debug(f"生成前提失败: {e}")
            return [variable_manager.create_variable(f"Step{step}_SimplePremise")]

    def _select_next_conclusion(self, premises: List[z3.ExprRef]) -> z3.ExprRef:
        """选择下一个结论"""
        complex_premises = []
        simple_premises = []

        for premise in premises:
            premise_str = str(premise)
            if any(op in premise_str.lower() for op in ['and', 'or', 'implies']):
                complex_premises.append(premise)
            else:
                simple_premises.append(premise)

        if complex_premises:
            return random.choice(complex_premises)
        else:
            return random.choice(premises)

    def _convert_chain_to_dag(self, chain: List[Dict]) -> DAGNode:
        """将线性链转换为DAG结构"""
        if not chain:
            return self._create_simple_node()

        nodes = {}

        # 从最后一步开始构建
        for i, chain_node in enumerate(reversed(chain)):
            depth = chain_node['depth']
            conclusion = chain_node['conclusion']
            premises = chain_node['premises']
            rule = chain_node['rule']

            # 创建结论节点
            conclusion_node = DAGNode(
                z3_expr=conclusion,
                rule=rule,
                rule_name=rule.name,
                depth=depth
            )

            # 为每个前提创建子节点
            for premise in premises:
                premise_key = str(premise)

                if premise_key in nodes:
                    premise_node = nodes[premise_key]
                else:
                    premise_node = DAGNode(
                        z3_expr=premise,
                        depth=depth + 1
                    )
                    nodes[premise_key] = premise_node

                conclusion_node.add_child(premise_node)

            conclusion_key = str(conclusion)
            nodes[conclusion_key] = conclusion_node

        # 返回最顶层节点
        if chain:
            root_conclusion = chain[0]['conclusion']
            return nodes[str(root_conclusion)]

        return self._create_simple_node()

    def _create_simple_node(self) -> DAGNode:
        """创建简单节点"""
        var = variable_manager.create_variable("Simple")
        return DAGNode(var)

    def _build_fallback_chain(self) -> DAGNode:
        """构建回退链条"""
        self.logger.info("使用回退方案构建短链条")

        try:
            # 使用统一变量命名
            p = variable_manager.create_variable("FallbackP")
            q = variable_manager.create_variable("FallbackQ")
            r = variable_manager.create_variable("FallbackR")

            from rules.hypothetical_syllogism import HypotheticalSyllogismRule
            rule = HypotheticalSyllogismRule()

            conclusion = z3.Implies(p, r)
            root = DAGNode(conclusion, rule, rule.name, depth=0)

            premise1 = DAGNode(z3.Implies(p, q), depth=1)
            premise2 = DAGNode(z3.Implies(q, r), depth=1)

            root.add_child(premise1)
            root.add_child(premise2)

            # 添加更深的子节点
            leaf1 = DAGNode(p, depth=2)
            leaf2 = DAGNode(q, depth=2)
            leaf3 = DAGNode(r, depth=2)

            premise1.add_child(leaf1)
            premise1.add_child(leaf2)
            premise2.add_child(leaf2)
            premise2.add_child(leaf3)

            return root

        except Exception as e:
            self.logger.error(f"回退方案失败: {e}")
            return self._create_simple_node()


def extract_logical_steps(root_node: DAGNode) -> List[Dict]:
    """
    迭代式提取逻辑步骤，使用增强的变量提取器
    """
    steps = []
    visited = set()

    # 使用队列进行层次遍历
    queue = deque([(root_node, 0)])

    while queue:
        node, depth = queue.popleft()
        node_id = id(node)

        if node_id in visited:
            continue

        visited.add(node_id)

        # 添加子节点到队列
        for child in node.children:
            if id(child) not in visited:
                queue.append((child, depth + 1))

        # 如果不是叶子节点，构造推理步骤
        if node.children:
            try:
                premises_expr = [c.z3_expr for c in node.children]
                premises_str = [str(c.z3_expr) for c in node.children]

                if len(premises_expr) == 1:
                    antecedent_str = premises_str[0]
                else:
                    antecedent_str = f"And({', '.join(premises_str)})"

                step = {
                    "rule": node.rule_name or "Unknown",
                    "conclusion": str(node.z3_expr),
                    "conclusion_expr": node.z3_expr,
                    "premises": premises_str,
                    "premises_expr": premises_expr,
                    "antecedent": antecedent_str,
                    "description": f"{node.rule_name or 'Unknown'} 推理得到 {str(node.z3_expr)}",
                    "depth": depth,
                    "premise_count": len(node.children),
                    "rule_type": _get_rule_type(node.rule_name) if node.rule_name else "unknown"
                }
                steps.append(step)

            except Exception as e:
                logging.getLogger("extract_logical_steps").debug(f"提取步骤失败: {e}")

    # 按深度排序，确保逻辑顺序
    steps.sort(key=lambda x: x.get('depth', 0))

    return steps


def _get_rule_type(rule_name):
    """获取规则类型"""
    rule_types = {
        "ModusPonens": "modus_ponens",
        "HypotheticalSyllogism": "hypothetical_syllogism",
        "ConjunctionIntroduction": "conjunction_intro",
        "ConjunctionElimination": "conjunction_elim",
        "DisjunctionIntroduction": "disjunction_intro",
        "UniversalInstantiation": "universal_instantiation",
        "TransitivityRule": "transitivity",
        "BiconditionalElimination": "biconditional_elim"
    }
    return rule_types.get(rule_name, "unknown")


# 统一接口函数
def build_reasoning_dag(max_depth=5, min_depth=2) -> Tuple[Optional[DAGNode], List[Dict]]:
    """
    构建推理DAG的统一接口
    使用统一变量命名系统
    """
    try:
        if max_depth <= 5:
            # 短链条：使用递归方式
            builder = ShortChainDAGBuilder(max_depth=max_depth, min_depth=min_depth)
            root_node = builder.build()
        else:
            # 长链条：使用迭代方式
            builder = LongChainDAGBuilder(max_depth=max_depth, min_depth=min_depth)
            root_node = builder.build_long_chain()

        if root_node is None:
            logging.getLogger("build_reasoning_dag").error("构建DAG失败")
            return None, []

        # 提取逻辑步骤
        steps = extract_logical_steps(root_node)

        logging.getLogger("build_reasoning_dag").info(
            f"成功构建DAG: 步骤数={len(steps)}, 类型={'短链条' if max_depth <= 5 else '长链条'}"
        )

        return root_node, steps

    except Exception as e:
        logging.getLogger("build_reasoning_dag").error(f"构建推理DAG时出错: {e}")
        return None, []


# 向后兼容的别名
def extract_logical_steps_improved(root_node):
    """向后兼容的接口"""
    return extract_logical_steps(root_node)


def build_long_reasoning_chain(max_depth=15, min_depth=5):
    """专用长链条接口"""
    return build_reasoning_dag(max_depth=max_depth, min_depth=min_depth)