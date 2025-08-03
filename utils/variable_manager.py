# 文件：utils/variable_manager.py
# 说明：改进的变量提取器，专注于控制变量数量和质量

import re
import random
import threading
from typing import Set, List, Dict, Optional, Tuple
import z3
from collections import defaultdict, Counter


class VariableManager:
    """
    统一变量命名管理器
    确保整个系统使用一致的变量命名模式：LogicVar_<ID>_<Type>
    """

    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()
        self._created_variables = set()

    def create_variable(self, var_type: str = "General", prefix: str = "LogicVar") -> z3.BoolRef:
        """创建统一格式的变量：LogicVar_<ID>_<Type>"""
        with self._lock:
            self._counter += 1
            var_name = f"{prefix}_{self._counter}_{var_type}"

            # 确保变量名唯一
            while var_name in self._created_variables:
                self._counter += 1
                var_name = f"{prefix}_{self._counter}_{var_type}"

            self._created_variables.add(var_name)
            return z3.Bool(var_name)

    def create_variables(self, count: int, var_type: str = "General") -> List[z3.BoolRef]:
        """批量创建变量"""
        return [self.create_variable(var_type) for _ in range(count)]

    def get_variable_info(self, var_name: str) -> Dict[str, str]:
        """解析变量名获取信息"""
        match = re.match(r'(\w+)_(\d+)_(\w+)', var_name)
        if match:
            return {
                "prefix": match.group(1),
                "id": match.group(2),
                "type": match.group(3),
                "full_name": var_name
            }
        return {"full_name": var_name, "type": "Unknown"}

    def reset(self):
        """重置计数器和变量集合"""
        with self._lock:
            self._counter = 0
            self._created_variables.clear()


# 全局变量管理器实例
variable_manager = VariableManager()


class VariableFilter:
    """
    变量过滤器：识别和过滤无意义的变量
    """

    def __init__(self):
        self.core_variable_patterns = [
            # 目标和结论相关
            r'.*Goal.*', r'.*Conclusion.*', r'.*Target.*', r'.*Final.*',
            # 核心前提
            r'.*Premise_[12].*', r'.*Primary.*', r'.*Main.*',
            # 中间关键节点
            r'.*Intermediate.*', r'.*Key.*', r'.*Critical.*'
        ]

        self.redundant_patterns = [
            # 重复或冗余变量
            r'.*_duplicate.*', r'.*_copy.*', r'.*_temp.*', r'.*_aux.*',
            # 过度细分的变量
            r'.*Sub\d+.*', r'.*Detail\d+.*', r'.*Minor\d+.*'
        ]

    def is_core_variable(self, var_name: str) -> bool:
        """判断是否为核心变量"""
        for pattern in self.core_variable_patterns:
            if re.match(pattern, var_name, re.IGNORECASE):
                return True
        return False

    def is_redundant_variable(self, var_name: str) -> bool:
        """判断是否为冗余变量"""
        for pattern in self.redundant_patterns:
            if re.match(pattern, var_name, re.IGNORECASE):
                return True
        return False

    def calculate_variable_importance(self, var_name: str, usage_count: int,
                                      context_info: Dict) -> float:
        """
        计算变量重要性分数 (0-1)

        Args:
            var_name: 变量名
            usage_count: 在表达式中的使用次数
            context_info: 上下文信息（如深度、规则类型等）
        """
        score = 0.0

        # 基础分数
        score += 0.2

        # 使用频率分数
        score += min(usage_count * 0.1, 0.3)

        # 核心变量加分
        if self.is_core_variable(var_name):
            score += 0.3

        # 变量类型分数
        var_info = variable_manager.get_variable_info(var_name)
        var_type = var_info.get("type", "").lower()

        type_scores = {
            "goal": 0.3,
            "conclusion": 0.25,
            "premise": 0.2,
            "intermediate": 0.15,
            "general": 0.1
        }
        score += type_scores.get(var_type, 0.1)

        # 冗余变量减分
        if self.is_redundant_variable(var_name):
            score -= 0.4

        # 上下文相关性分数
        depth = context_info.get("depth", 0)
        if depth <= 2:  # 浅层变量更重要
            score += 0.1

        return min(max(score, 0.0), 1.0)


class EnhancedVariableExtractor:
    """
    增强的变量提取器：支持多种变量命名模式，智能过滤和数量控制
    """

    def __init__(self, max_variables: int = 10, min_variables: int = 3):
        """
        Args:
            max_variables: 最大变量数量
            min_variables: 最小变量数量
        """
        self.max_variables = max_variables
        self.min_variables = min_variables
        self.filter = VariableFilter()

        # 扩展的变量匹配模式
        self.variable_patterns = [
            r'LogicVar_\d+_\w+',  # 新的统一格式
            r'Var_\d+',  # 原有格式
            r'\w+_\d+',  # 通用格式：前缀_数字
            r'[A-Z][a-zA-Z]*_\d+',  # 首字母大写格式
        ]

        # 编译正则表达式以提高性能
        self.compiled_patterns = [re.compile(pattern) for pattern in self.variable_patterns]

    def extract_from_dag(self, root_node) -> List[str]:
        """从DAG中提取核心变量，控制数量和质量"""
        raw_variables = self._extract_raw_variables_from_dag(root_node)

        if not raw_variables:
            return []

        # 分析变量使用情况
        variable_analysis = self._analyze_variable_usage(root_node, raw_variables)

        # 过滤和排序变量
        filtered_variables = self._filter_and_rank_variables(variable_analysis)

        # 应用数量限制
        final_variables = self._apply_quantity_limits(filtered_variables)

        return final_variables

    def _extract_raw_variables_from_dag(self, root_node) -> Set[str]:
        """提取DAG中的所有原始变量"""
        variables = set()

        def safe_traverse(node):
            try:
                if not hasattr(node, 'z3_expr') or node.z3_expr is None:
                    return

                # 提取当前节点的变量
                expr_vars = self._extract_from_single_expression(node.z3_expr)
                variables.update(expr_vars)

                # 递归处理子节点
                if hasattr(node, 'children') and node.children:
                    for child in node.children:
                        if child is not None:
                            safe_traverse(child)

            except Exception as e:
                print(f"[DEBUG] 遍历节点时出错: {e}")

        safe_traverse(root_node)
        return variables

    def _extract_from_single_expression(self, expr) -> Set[str]:
        """从单个表达式中提取变量"""
        variables = set()

        try:
            if expr is None:
                return variables

            # 方法1：字符串匹配
            expr_str = str(expr)
            for pattern in self.compiled_patterns:
                matches = pattern.findall(expr_str)
                variables.update(matches)

            # 方法2：递归遍历Z3表达式结构
            self._recursive_extract(expr, variables)

        except Exception as e:
            print(f"[DEBUG] 从表达式提取变量时出错: {e}")

        return variables

    def _recursive_extract(self, expr, variables: Set[str]):
        """递归提取Z3表达式中的变量"""
        try:
            if z3.is_const(expr):
                if z3.is_bool(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                    var_name = str(expr)
                    if self._is_valid_variable_name(var_name):
                        variables.add(var_name)
            elif hasattr(expr, 'children'):
                for child in expr.children():
                    self._recursive_extract(child, variables)

        except Exception as e:
            print(f"[DEBUG] 递归提取时出错: {e}")

    def _is_valid_variable_name(self, name: str) -> bool:
        """检查是否为有效的变量名"""
        for pattern in self.compiled_patterns:
            if pattern.fullmatch(name):
                return True
        return False

    def _analyze_variable_usage(self, root_node, variables: Set[str]) -> Dict[str, Dict]:
        """分析变量在DAG中的使用情况"""
        analysis = {}

        for var in variables:
            analysis[var] = {
                "usage_count": 0,
                "depth_appearances": [],
                "expression_types": [],
                "context_info": {}
            }

        def analyze_node(node, depth=0):
            try:
                if not hasattr(node, 'z3_expr') or node.z3_expr is None:
                    return

                expr_str = str(node.z3_expr)

                for var in variables:
                    if var in expr_str:
                        analysis[var]["usage_count"] += 1
                        analysis[var]["depth_appearances"].append(depth)

                        # 记录表达式类型
                        if hasattr(node, 'rule_name') and node.rule_name:
                            analysis[var]["expression_types"].append(node.rule_name)

                # 递归分析子节点
                if hasattr(node, 'children') and node.children:
                    for child in node.children:
                        if child is not None:
                            analyze_node(child, depth + 1)

            except Exception as e:
                print(f"[DEBUG] 分析节点时出错: {e}")

        analyze_node(root_node)

        # 计算上下文信息
        for var, info in analysis.items():
            info["context_info"] = {
                "depth": min(info["depth_appearances"]) if info["depth_appearances"] else 0,
                "depth_span": max(info["depth_appearances"]) - min(info["depth_appearances"]) if len(
                    info["depth_appearances"]) > 1 else 0,
                "rule_diversity": len(set(info["expression_types"]))
            }

        return analysis

    def _filter_and_rank_variables(self, variable_analysis: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """过滤和排序变量"""
        variable_scores = []

        for var_name, analysis in variable_analysis.items():
            # 基础过滤：排除明显无用的变量
            if analysis["usage_count"] == 0:
                continue

            if self.filter.is_redundant_variable(var_name):
                continue

            # 计算重要性分数
            importance_score = self.filter.calculate_variable_importance(
                var_name,
                analysis["usage_count"],
                analysis["context_info"]
            )

            variable_scores.append((var_name, importance_score))

        # 按重要性分数排序
        variable_scores.sort(key=lambda x: x[1], reverse=True)

        return variable_scores

    def _apply_quantity_limits(self, ranked_variables: List[Tuple[str, float]]) -> List[str]:
        """应用数量限制"""
        if not ranked_variables:
            return []

        # 确保至少有最小数量的变量
        min_count = min(self.min_variables, len(ranked_variables))

        # 应用最大数量限制
        max_count = min(self.max_variables, len(ranked_variables))

        # 智能选择：确保包含最重要的变量
        selected_variables = []

        # 首先选择高分变量
        high_score_vars = [var for var, score in ranked_variables if score >= 0.7]
        selected_variables.extend(high_score_vars[:max_count])

        # 如果数量不足，添加中等分数的变量
        if len(selected_variables) < min_count:
            remaining_count = min_count - len(selected_variables)
            medium_score_vars = [var for var, score in ranked_variables
                                 if 0.4 <= score < 0.7 and var not in selected_variables]
            selected_variables.extend(medium_score_vars[:remaining_count])

        # 如果还是不足，添加任何可用的变量
        if len(selected_variables) < min_count:
            remaining_count = min_count - len(selected_variables)
            any_vars = [var for var, _ in ranked_variables
                        if var not in selected_variables]
            selected_variables.extend(any_vars[:remaining_count])

        # 最终限制在最大数量内
        final_variables = selected_variables[:self.max_variables]

        print(f"[DEBUG] 变量筛选结果: {len(ranked_variables)} -> {len(final_variables)}")

        return final_variables

    def extract_from_steps(self, logical_steps: List[Dict]) -> List[str]:
        """从逻辑步骤中提取变量（备用方法）"""
        raw_variables = set()

        for step in logical_steps:
            try:
                # 从前提表达式中提取
                premises_expr = step.get('premises_expr', [])
                for premise in premises_expr:
                    if premise is not None:
                        expr_vars = self._extract_from_single_expression(premise)
                        raw_variables.update(expr_vars)

                # 从结论表达式中提取
                conclusion_expr = step.get('conclusion_expr')
                if conclusion_expr is not None:
                    expr_vars = self._extract_from_single_expression(conclusion_expr)
                    raw_variables.update(expr_vars)

            except Exception as e:
                print(f"[DEBUG] 从步骤提取变量时出错: {e}")

        # 简化的变量分析（基于使用频率）
        variable_counts = Counter()
        for step in logical_steps:
            try:
                premises_str = step.get('premises', [])
                conclusion_str = step.get('conclusion', '')
                full_text = ' '.join(premises_str) + ' ' + conclusion_str

                for var in raw_variables:
                    if var in full_text:
                        variable_counts[var] += 1

            except Exception:
                continue

        # 按使用频率排序并应用数量限制
        sorted_vars = sorted(variable_counts.items(), key=lambda x: x[1], reverse=True)

        # 应用数量控制
        max_count = min(self.max_variables, len(sorted_vars))
        min_count = min(self.min_variables, len(sorted_vars))

        selected_count = max(min_count, min(max_count, len(sorted_vars)))
        final_variables = [var for var, _ in sorted_vars[:selected_count]]

        print(f"[DEBUG] 从步骤提取变量: {len(raw_variables)} -> {len(final_variables)}")

        return final_variables

    def create_safe_bool_vars(self, var_names: List[str], max_count: int = None) -> List[z3.BoolRef]:
        """安全地创建Z3布尔变量"""
        if max_count is None:
            max_count = self.max_variables

        safe_vars = []

        for var_name in var_names[:max_count]:
            try:
                if self._is_valid_variable_name(var_name):
                    bool_var = z3.Bool(var_name)
                    safe_vars.append(bool_var)
                else:
                    print(f"[DEBUG] 跳过无效变量名: {var_name}")

            except Exception as e:
                print(f"[DEBUG] 创建布尔变量 {var_name} 时出错: {e}")

        return safe_vars

    def normalize_variable_names(self, variables: List[str]) -> List[str]:
        """规范化变量名（转换为统一格式）"""
        normalized = []

        for var in variables:
            try:
                # 如果已经是统一格式，直接使用
                if re.match(r'LogicVar_\d+_\w+', var):
                    normalized.append(var)
                # 否则转换为统一格式
                else:
                    # 提取数字部分作为ID
                    numbers = re.findall(r'\d+', var)
                    if numbers:
                        var_id = numbers[0]
                        # 推断类型
                        if 'goal' in var.lower() or 'conclusion' in var.lower():
                            var_type = 'Goal'
                        elif 'premise' in var.lower():
                            var_type = 'Premise'
                        else:
                            var_type = 'General'

                        normalized_name = f"LogicVar_{var_id}_{var_type}"
                        normalized.append(normalized_name)
                    else:
                        # 如果无法提取数字，使用原名
                        normalized.append(var)

            except Exception as e:
                print(f"[DEBUG] 规范化变量名 {var} 时出错: {e}")
                normalized.append(var)

        return normalized

    def get_variable_statistics(self, variables: List[str]) -> Dict:
        """获取变量统计信息"""
        stats = {
            "total_count": len(variables),
            "by_type": {},
            "by_prefix": {},
            "unified_format_count": 0,
            "legacy_format_count": 0,
            "quality_info": {
                "within_limits": len(variables) <= self.max_variables,
                "meets_minimum": len(variables) >= self.min_variables,
                "recommended_range": f"{self.min_variables}-{self.max_variables}"
            }
        }

        for var in variables:
            var_info = variable_manager.get_variable_info(var)
            var_type = var_info.get("type", "Unknown")
            var_prefix = var_info.get("prefix", "Unknown")

            stats["by_type"][var_type] = stats["by_type"].get(var_type, 0) + 1
            stats["by_prefix"][var_prefix] = stats["by_prefix"].get(var_prefix, 0) + 1

            if re.match(r'LogicVar_\d+_\w+', var):
                stats["unified_format_count"] += 1
            else:
                stats["legacy_format_count"] += 1

        return stats


class RuleVariableMixin:
    """
    规则变量混入类
    为所有规则提供统一的变量创建方法
    """

    def create_variable(self, var_type: str = "General") -> z3.BoolRef:
        """创建统一格式的变量"""
        return variable_manager.create_variable(var_type)

    def create_goal_variable(self) -> z3.BoolRef:
        """创建目标变量"""
        return variable_manager.create_variable("Goal")

    def create_premise_variable(self) -> z3.BoolRef:
        """创建前提变量"""
        return variable_manager.create_variable("Premise")

    def create_conclusion_variable(self) -> z3.BoolRef:
        """创建结论变量"""
        return variable_manager.create_variable("Conclusion")

    def create_intermediate_variable(self) -> z3.BoolRef:
        """创建中间变量"""
        return variable_manager.create_variable("Intermediate")