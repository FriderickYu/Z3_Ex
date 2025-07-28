# 文件：utils/safe_variable_extractor.py
# 说明：安全的Z3变量提取工具，避免布尔转换错误

import re
import z3
from typing import List, Set, Dict, Any
import logging


class SafeVariableExtractor:
    """安全的变量提取器，避免Z3表达式布尔转换错误"""

    def __init__(self):
        self.logger = logging.getLogger("safe_variable_extractor")

    def extract_from_dag(self, root_node) -> List[str]:
        """从DAG中安全地提取所有变量名"""
        variables = set()

        def safe_traverse(node):
            try:
                # 检查节点是否有z3_expr属性
                if not hasattr(node, 'z3_expr'):
                    return

                expr = node.z3_expr
                if expr is None:
                    return

                # 安全地转换为字符串并提取变量
                try:
                    expr_str = str(expr)
                    vars_in_expr = re.findall(r'Var_\d+', expr_str)
                    variables.update(vars_in_expr)
                except Exception as e:
                    self.logger.debug(f"提取表达式字符串时出错: {e}")

                # 安全地遍历子节点
                if hasattr(node, 'children') and node.children:
                    for child in node.children:
                        if child is not None:
                            safe_traverse(child)

            except Exception as e:
                self.logger.debug(f"遍历节点时出错: {e}")

        safe_traverse(root_node)
        return sorted(list(variables))

    def extract_from_steps(self, logical_steps: List[Dict]) -> List[str]:
        """从逻辑步骤中提取变量"""
        variables = set()

        for step in logical_steps:
            # 从前提中提取
            premises = step.get('premises', [])
            if premises:
                for premise in premises:
                    if isinstance(premise, str):
                        vars_found = re.findall(r'Var_\d+', premise)
                        variables.update(vars_found)

            # 从结论中提取
            conclusion = step.get('conclusion', '')
            if isinstance(conclusion, str):
                vars_found = re.findall(r'Var_\d+', conclusion)
                variables.update(vars_found)

        return sorted(list(variables))

    def extract_from_expression(self, expr) -> Set[str]:
        """从单个Z3表达式中提取变量"""
        variables = set()

        try:
            if expr is None:
                return variables

            # 安全地转换为字符串
            expr_str = str(expr)
            vars_found = re.findall(r'Var_\d+', expr_str)
            variables.update(vars_found)

        except Exception as e:
            self.logger.debug(f"从表达式提取变量时出错: {e}")

        return variables

    def create_safe_bool_vars(self, var_names: List[str], max_count: int = 10) -> List[z3.BoolRef]:
        """安全地创建Z3布尔变量"""
        safe_vars = []

        for var_name in var_names[:max_count]:
            try:
                bool_var = z3.Bool(var_name)
                safe_vars.append(bool_var)
            except Exception as e:
                self.logger.debug(f"创建布尔变量 {var_name} 时出错: {e}")
                continue

        return safe_vars

    def validate_expressions(self, expressions: List[Any]) -> List[Any]:
        """验证表达式列表，过滤掉可能有问题的表达式"""
        valid_expressions = []

        for expr in expressions:
            try:
                if expr is None:
                    continue

                # 尝试安全地访问表达式
                _ = str(expr)

                # 如果是Z3表达式，检查是否为布尔类型
                if hasattr(expr, 'sort'):
                    if z3.is_bool(expr):
                        valid_expressions.append(expr)
                else:
                    # 如果不是Z3表达式，也保留
                    valid_expressions.append(expr)

            except Exception as e:
                self.logger.debug(f"验证表达式时出错: {e}")
                continue

        return valid_expressions