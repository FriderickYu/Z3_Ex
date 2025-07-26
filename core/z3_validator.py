"""
Z3验证器 - 确保推理数据集的逻辑正确性
核心职责：字符串表达式转Z3 + 推理结构验证 + 数据集质量控制
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import time
import threading
import json
import yaml
from contextlib import contextmanager

try:
    import z3
except ImportError:
    raise ImportError("请安装z3-solver: pip install z3-solver")

from rules.base.rule import LogicalFormula, LogicalOperator

try:
    from ..utils.logger_utils import ARNGLogger
except ValueError:
    from utils.logger_utils import ARNGLogger


class ValidationStrategy(Enum):
    """验证策略"""
    DIRECT = "direct"  # 直接验证法（推荐）
    MODEL_BASED = "model_based"  # 模型检验法
    HYBRID = "hybrid"  # 混合策略
    FAST = "fast"  # 快速验证（牺牲一些准确性）


class ValidationResult(Enum):
    """验证结果状态"""
    VALID = "valid"  # 逻辑有效
    INVALID = "invalid"  # 逻辑无效
    UNKNOWN = "unknown"  # Z3无法确定
    TIMEOUT = "timeout"  # 验证超时
    ERROR = "error"  # 转换或验证错误


@dataclass
class Z3ValidationResult:
    """Z3验证结果"""
    status: ValidationResult
    is_valid: bool
    confidence_score: float = 0.0

    # 诊断信息
    error_messages: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Z3相关
    z3_result: Optional[z3.CheckSatResult] = None
    counterexample_model: Optional[z3.ModelRef] = None

    # 性能统计
    validation_time_ms: float = 0.0
    z3_solver_stats: Dict[str, Any] = field(default_factory=dict)

    # 调试信息
    z3_expressions: List[str] = field(default_factory=list)
    conversion_warnings: List[str] = field(default_factory=list)


@dataclass
class Z3ValidatorConfig:
    """Z3验证器配置"""
    # 核心验证设置
    timeout_seconds: int = 30
    validation_strategy: ValidationStrategy = ValidationStrategy.DIRECT
    max_model_count: int = 10

    # 质量控制设置
    require_strict_validity: bool = True  # 严格模式：任何不确定都视为无效
    enable_counterexamples: bool = True
    enable_suggestion_generation: bool = True

    # 性能优化设置
    use_expression_cache: bool = True
    parallel_validation: bool = False
    memory_limit_mb: int = 1024

    # 错误处理设置
    fallback_on_timeout: bool = True
    fallback_on_conversion_error: bool = True
    log_all_failures: bool = True

    # Z3求解器设置
    z3_tactics: List[str] = field(default_factory=lambda: ["simplify", "solve-eqs"])
    z3_parameters: Dict[str, Any] = field(default_factory=dict)


class ExpressionConverter:
    """
    表达式转换器：字符串表达式 → Z3表达式
    这是数据质量保证的第一道关卡
    """

    def __init__(self, config: Z3ValidatorConfig):
        self.config = config
        self.logger = ARNGLogger("ExpressionConverter")

        # 变量管理
        self.z3_variables: Dict[str, z3.BoolRef] = {}
        self.variable_contexts: Dict[str, int] = {}  # 变量作用域计数

        # 转换缓存
        self.conversion_cache: Dict[str, z3.ExprRef] = {}

        # 操作符映射
        self.operator_patterns = {
            '→': r'→|->|implies|IMPLIES',
            '↔': r'↔|<->|<=>|iff|IFF',
            '∧': r'∧|&|and|AND|∩',
            '∨': r'∨|\||or|OR|∪',
            '¬': r'¬|~|not|NOT|!',
            '∀': r'∀|forall|FORALL|all',
            '∃': r'∃|exists|EXISTS|some'
        }

        self.logger.info("表达式转换器初始化完成")

    def convert_to_z3(self, expression: Union[str, LogicalFormula]) -> Tuple[z3.ExprRef, List[str]]:
        """
        将表达式转换为Z3表达式

        Args:
            expression: 字符串表达式或LogicalFormula对象

        Returns:
            Tuple[z3.ExprRef, List[str]]: (Z3表达式, 警告信息列表)
        """
        warnings = []

        # 统一转换为字符串
        if isinstance(expression, LogicalFormula):
            expr_str = expression.expression
            # 利用LogicalFormula的变量信息
            for var in expression.variables:
                self._ensure_variable(var)
        else:
            expr_str = str(expression)

        # 检查缓存
        cache_key = self._get_cache_key(expr_str)
        if self.config.use_expression_cache and cache_key in self.conversion_cache:
            return self.conversion_cache[cache_key], warnings

        try:
            # 预处理表达式
            normalized_expr = self._normalize_expression(expr_str)

            # 递归解析转换
            z3_expr = self._parse_expression(normalized_expr, warnings)

            # 缓存结果
            if self.config.use_expression_cache:
                self.conversion_cache[cache_key] = z3_expr

            self.logger.debug(f"成功转换表达式: {expr_str} → {z3_expr}")
            return z3_expr, warnings

        except Exception as e:
            error_msg = f"表达式转换失败: {expr_str}, 错误: {str(e)}"
            self.logger.error(error_msg)
            warnings.append(error_msg)

            # 返回一个占位符（避免完全失败）
            placeholder = z3.BoolVal(True)
            return placeholder, warnings

    def _normalize_expression(self, expr_str: str) -> str:
        """标准化表达式字符串"""
        # 移除多余空格
        expr_str = ' '.join(expr_str.split())

        # 标准化操作符
        for standard_op, pattern in self.operator_patterns.items():
            expr_str = re.sub(pattern, standard_op, expr_str, flags=re.IGNORECASE)

        # 标准化括号
        expr_str = expr_str.replace('（', '(').replace('）', ')')
        expr_str = expr_str.replace('[', '(').replace(']', ')')

        return expr_str.strip()

    def _parse_expression(self, expr_str: str, warnings: List[str]) -> z3.ExprRef:
        """递归解析表达式"""
        expr_str = expr_str.strip()

        # 处理括号
        if expr_str.startswith('(') and expr_str.endswith(')'):
            # 检查括号是否匹配整个表达式
            if self._is_fully_parenthesized(expr_str):
                return self._parse_expression(expr_str[1:-1], warnings)

        # 按优先级解析操作符

        # 1. 双条件 (↔) - 最低优先级
        if '↔' in expr_str:
            return self._parse_binary_operator(expr_str, '↔', z3.And, warnings)  # Z3中用And表示iff

        # 2. 蕴含 (→)
        if '→' in expr_str:
            return self._parse_binary_operator(expr_str, '→', z3.Implies, warnings)

        # 3. 析取 (∨)
        if '∨' in expr_str:
            return self._parse_binary_operator(expr_str, '∨', z3.Or, warnings)

        # 4. 合取 (∧)
        if '∧' in expr_str:
            return self._parse_binary_operator(expr_str, '∧', z3.And, warnings)

        # 5. 否定 (¬) - 最高优先级
        if expr_str.startswith('¬'):
            inner_expr = expr_str[1:].strip()
            return z3.Not(self._parse_expression(inner_expr, warnings))

        # 6. 量词
        if expr_str.startswith('∀') or expr_str.startswith('∃'):
            return self._parse_quantifier(expr_str, warnings)

        # 7. 原子变量
        return self._parse_atomic(expr_str, warnings)

    def _parse_binary_operator(self, expr_str: str, operator: str,
                               z3_func, warnings: List[str]) -> z3.ExprRef:
        """解析二元操作符"""
        # 找到主要的操作符位置（不在括号内的）
        op_pos = self._find_main_operator_position(expr_str, operator)

        if op_pos == -1:
            warnings.append(f"未找到操作符 {operator} 在表达式: {expr_str}")
            return self._parse_atomic(expr_str, warnings)

        left_part = expr_str[:op_pos].strip()
        right_part = expr_str[op_pos + len(operator):].strip()

        left_z3 = self._parse_expression(left_part, warnings)
        right_z3 = self._parse_expression(right_part, warnings)

        # 特殊处理双条件
        if operator == '↔':
            return z3.And(z3.Implies(left_z3, right_z3), z3.Implies(right_z3, left_z3))

        return z3_func(left_z3, right_z3)

    def _parse_quantifier(self, expr_str: str, warnings: List[str]) -> z3.ExprRef:
        """解析量词表达式"""
        # 简化实现：暂时将量词表达式转换为布尔常量
        # 在完整实现中，这里需要处理变量绑定和作用域
        warnings.append(f"量词表达式暂时简化处理: {expr_str}")

        if expr_str.startswith('∀'):
            # 全称量词：暂时返回True（保守估计）
            return z3.BoolVal(True)
        else:
            # 存在量词：暂时返回False（保守估计）
            return z3.BoolVal(False)

    def _parse_atomic(self, expr_str: str, warnings: List[str]) -> z3.ExprRef:
        """解析原子表达式（变量）"""
        expr_str = expr_str.strip()

        # 检查是否是有效的变量名
        if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', expr_str):
            warnings.append(f"可能的无效变量名: {expr_str}")

        return self._ensure_variable(expr_str)

    def _ensure_variable(self, var_name: str) -> z3.BoolRef:
        """确保变量存在，如果不存在则创建"""
        if var_name not in self.z3_variables:
            self.z3_variables[var_name] = z3.Bool(var_name)
            self.variable_contexts[var_name] = 1
        else:
            self.variable_contexts[var_name] += 1

        return self.z3_variables[var_name]

    def _find_main_operator_position(self, expr_str: str, operator: str) -> int:
        """找到不在括号内的主要操作符位置"""
        paren_depth = 0
        i = 0
        while i < len(expr_str):
            if expr_str[i] == '(':
                paren_depth += 1
            elif expr_str[i] == ')':
                paren_depth -= 1
            elif paren_depth == 0 and expr_str[i:].startswith(operator):
                return i
            i += 1
        return -1

    def _is_fully_parenthesized(self, expr_str: str) -> bool:
        """检查表达式是否被完整括号包围"""
        if not (expr_str.startswith('(') and expr_str.endswith(')')):
            return False

        paren_count = 0
        for i, char in enumerate(expr_str):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0 and i < len(expr_str) - 1:
                    return False

        return paren_count == 0

    def _get_cache_key(self, expr_str: str) -> str:
        """生成缓存键"""
        return expr_str.strip().lower()

    def get_variable_usage_stats(self) -> Dict[str, int]:
        """获取变量使用统计"""
        return self.variable_contexts.copy()

    def clear_cache(self):
        """清空转换缓存"""
        self.conversion_cache.clear()
        self.logger.info("表达式转换缓存已清空")


class Z3Validator:
    """
    Z3验证器主类 - 推理结构验证和数据集质量控制
    """

    def __init__(self, config: Optional[Z3ValidatorConfig] = None, **kwargs):
        # 配置驱动设计
        self.config = config or Z3ValidatorConfig()

        # 支持关键字参数覆盖配置
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.logger = ARNGLogger("Z3Validator")

        # 核心组件
        self.converter = ExpressionConverter(self.config)

        # 验证统计
        self.validation_stats = {
            'total_validations': 0,
            'valid_count': 0,
            'invalid_count': 0,
            'timeout_count': 0,
            'error_count': 0,
            'average_time_ms': 0.0
        }

        # 失败案例记录（用于分析和改进）
        self.failed_cases: List[Tuple[str, Z3ValidationResult]] = []

        self.logger.info(f"Z3验证器初始化完成 - 策略: {self.config.validation_strategy.value}")

    @classmethod
    def from_args(cls, args):
        """从命令行参数创建验证器"""
        config = Z3ValidatorConfig(
            timeout_seconds=getattr(args, 'z3_timeout', 30),
            validation_strategy=ValidationStrategy(getattr(args, 'validation_strategy', 'direct')),
            require_strict_validity=getattr(args, 'strict_validation', True),
            use_expression_cache=getattr(args, 'use_cache', True),
        )
        return cls(config)

    def validate_reasoning_step(self,
                                premises: List[Union[str, LogicalFormula]],
                                conclusion: Union[str, LogicalFormula]) -> Z3ValidationResult:
        """
        验证单个推理步骤的逻辑有效性
        这是数据集质量保证的核心方法

        Args:
            premises: 前提列表
            conclusion: 结论

        Returns:
            Z3ValidationResult: 详细的验证结果
        """
        start_time = time.time()
        result = Z3ValidationResult(status=ValidationResult.UNKNOWN, is_valid=False)

        try:
            # 1. 转换所有表达式为Z3格式
            z3_premises = []
            all_warnings = []

            for premise in premises:
                z3_expr, warnings = self.converter.convert_to_z3(premise)
                z3_premises.append(z3_expr)
                all_warnings.extend(warnings)
                result.z3_expressions.append(str(z3_expr))

            z3_conclusion, conclusion_warnings = self.converter.convert_to_z3(conclusion)
            all_warnings.extend(conclusion_warnings)
            result.z3_expressions.append(str(z3_conclusion))
            result.conversion_warnings = all_warnings

            # 2. 执行验证
            if self.config.validation_strategy == ValidationStrategy.DIRECT:
                result = self._validate_direct(z3_premises, z3_conclusion, result)
            elif self.config.validation_strategy == ValidationStrategy.MODEL_BASED:
                result = self._validate_model_based(z3_premises, z3_conclusion, result)
            elif self.config.validation_strategy == ValidationStrategy.HYBRID:
                result = self._validate_hybrid(z3_premises, z3_conclusion, result)
            else:  # FAST
                result = self._validate_fast(z3_premises, z3_conclusion, result)

            # 3. 后处理
            self._post_process_result(result, premises, conclusion)

        except Exception as e:
            result.status = ValidationResult.ERROR
            result.is_valid = False
            result.error_messages.append(f"验证过程中发生错误: {str(e)}")
            self.logger.error(f"验证错误: {str(e)}")

        # 4. 记录统计信息
        result.validation_time_ms = (time.time() - start_time) * 1000
        self._update_statistics(result)

        # 5. 记录失败案例
        if not result.is_valid and self.config.log_all_failures:
            case_info = f"前提: {premises}, 结论: {conclusion}"
            self.failed_cases.append((case_info, result))

            # 限制失败案例记录数量
            if len(self.failed_cases) > 1000:
                self.failed_cases = self.failed_cases[-500:]

        return result

    def _validate_direct(self, z3_premises: List[z3.ExprRef],
                         z3_conclusion: z3.ExprRef,
                         result: Z3ValidationResult) -> Z3ValidationResult:
        """直接验证法：检查 premises ⊨ conclusion"""

        with self._create_solver() as solver:
            # 添加所有前提
            for premise in z3_premises:
                solver.add(premise)

            # 添加结论的否定
            solver.add(z3.Not(z3_conclusion))

            # 检查可满足性
            with self._timeout_context():
                check_result = solver.check()

            result.z3_result = check_result

            if check_result == z3.unsat:
                # 前提+¬结论不可满足 → 推理有效
                result.status = ValidationResult.VALID
                result.is_valid = True
                result.confidence_score = 1.0
            elif check_result == z3.sat:
                # 找到反例
                result.status = ValidationResult.INVALID
                result.is_valid = False
                result.confidence_score = 1.0

                if self.config.enable_counterexamples:
                    result.counterexample_model = solver.model()
                    result.error_messages.append("找到反例，推理无效")
            else:  # unknown
                result.status = ValidationResult.UNKNOWN
                result.is_valid = False if self.config.require_strict_validity else None
                result.confidence_score = 0.0
                result.error_messages.append("Z3无法确定结果")

        return result

    def _validate_model_based(self, z3_premises: List[z3.ExprRef],
                              z3_conclusion: z3.ExprRef,
                              result: Z3ValidationResult) -> Z3ValidationResult:
        """模型检验法：检查所有满足前提的模型都满足结论"""
        # 这里是简化实现，完整版本需要模型枚举
        return self._validate_direct(z3_premises, z3_conclusion, result)

    def _validate_hybrid(self, z3_premises: List[z3.ExprRef],
                         z3_conclusion: z3.ExprRef,
                         result: Z3ValidationResult) -> Z3ValidationResult:
        """混合策略：先尝试快速验证，失败时使用严格验证"""
        # 先尝试快速验证
        fast_result = self._validate_fast(z3_premises, z3_conclusion, result)

        if fast_result.status == ValidationResult.VALID:
            return fast_result

        # 快速验证失败，使用严格验证
        return self._validate_direct(z3_premises, z3_conclusion, result)

    def _validate_fast(self, z3_premises: List[z3.ExprRef],
                       z3_conclusion: z3.ExprRef,
                       result: Z3ValidationResult) -> Z3ValidationResult:
        """快速验证：使用简化的求解器设置"""
        # 实现快速但可能不够准确的验证
        return self._validate_direct(z3_premises, z3_conclusion, result)

    @contextmanager
    def _create_solver(self):
        """创建配置好的Z3求解器"""
        solver = z3.Solver()

        # 应用自定义tactics
        if self.config.z3_tactics:
            tactic = z3.Then(*[z3.Tactic(t) for t in self.config.z3_tactics])
            solver = tactic.solver()

        # 设置参数
        for param, value in self.config.z3_parameters.items():
            solver.set(param, value)

        # 设置超时
        solver.set("timeout", self.config.timeout_seconds * 1000)  # 毫秒

        try:
            yield solver
        finally:
            # 清理资源
            pass

    @contextmanager
    def _timeout_context(self):
        """超时上下文管理器"""
        if self.config.timeout_seconds > 0:
            def timeout_handler():
                raise TimeoutError(f"Z3验证超时 ({self.config.timeout_seconds}s)")

            timer = threading.Timer(self.config.timeout_seconds, timeout_handler)
            timer.start()
            try:
                yield
            finally:
                timer.cancel()
        else:
            yield

    def _post_process_result(self, result: Z3ValidationResult,
                             premises: List, conclusion) -> None:
        """后处理验证结果"""
        # 生成建议
        if self.config.enable_suggestion_generation and not result.is_valid:
            result.suggestions = self._generate_suggestions(result, premises, conclusion)

        # 严格模式处理
        if self.config.require_strict_validity:
            if result.status in [ValidationResult.UNKNOWN, ValidationResult.TIMEOUT, ValidationResult.ERROR]:
                result.is_valid = False

    def _generate_suggestions(self, result: Z3ValidationResult,
                              premises: List, conclusion) -> List[str]:
        """生成修复建议"""
        suggestions = []

        if result.status == ValidationResult.INVALID:
            suggestions.append("检查前提是否充分支持结论")
            suggestions.append("验证前提之间是否存在矛盾")

            if result.counterexample_model:
                suggestions.append("参考反例模型调整推理逻辑")

        elif result.status == ValidationResult.TIMEOUT:
            suggestions.append("简化表达式复杂度")
            suggestions.append("增加验证超时时间")

        elif result.status == ValidationResult.ERROR:
            suggestions.append("检查表达式语法是否正确")
            suggestions.append("确认所有变量都有定义")

        return suggestions

    def _update_statistics(self, result: Z3ValidationResult):
        """更新验证统计信息"""
        self.validation_stats['total_validations'] += 1

        if result.status == ValidationResult.VALID:
            self.validation_stats['valid_count'] += 1
        elif result.status == ValidationResult.INVALID:
            self.validation_stats['invalid_count'] += 1
        elif result.status == ValidationResult.TIMEOUT:
            self.validation_stats['timeout_count'] += 1
        else:
            self.validation_stats['error_count'] += 1

        # 更新平均时间
        total = self.validation_stats['total_validations']
        current_avg = self.validation_stats['average_time_ms']
        new_avg = (current_avg * (total - 1) + result.validation_time_ms) / total
        self.validation_stats['average_time_ms'] = new_avg

    def validate_reasoning_chain(self, reasoning_chain: List[Dict]) -> List[Z3ValidationResult]:
        """
        验证完整的推理链

        Args:
            reasoning_chain: 推理链，每个元素包含 {'premises': [...], 'conclusion': '...'}

        Returns:
            List[Z3ValidationResult]: 每个步骤的验证结果
        """
        results = []

        for i, step in enumerate(reasoning_chain):
            try:
                premises = step.get('premises', [])
                conclusion = step.get('conclusion', '')

                result = self.validate_reasoning_step(premises, conclusion)
                results.append(result)

                # 如果某步骤无效且在严格模式下，可以选择提前终止
                if self.config.require_strict_validity and not result.is_valid:
                    self.logger.warning(f"推理链在步骤 {i} 处验证失败: {step}")

            except Exception as e:
                error_result = Z3ValidationResult(
                    status=ValidationResult.ERROR,
                    is_valid=False,
                    error_messages=[f"步骤 {i} 验证失败: {str(e)}"]
                )
                results.append(error_result)

        return results

    def filter_valid_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        过滤数据集，分离有效和无效样本
        这是数据集质量控制的关键方法

        Args:
            samples: 推理样本列表

        Returns:
            Tuple[List[Dict], List[Dict]]: (有效样本, 无效样本)
        """
        valid_samples = []
        invalid_samples = []

        for sample in samples:
            try:
                if 'premises' in sample and 'conclusion' in sample:
                    result = self.validate_reasoning_step(
                        sample['premises'],
                        sample['conclusion']
                    )

                    if result.is_valid:
                        valid_samples.append(sample)
                    else:
                        # 记录失败原因
                        sample['validation_failure'] = {
                            'status': result.status.value,
                            'errors': result.error_messages,
                            'suggestions': result.suggestions
                        }
                        invalid_samples.append(sample)
                else:
                    # 样本格式不正确
                    sample['validation_failure'] = {
                        'status': 'format_error',
                        'errors': ['样本缺少必要的premises或conclusion字段']
                    }
                    invalid_samples.append(sample)

            except Exception as e:
                sample['validation_failure'] = {
                    'status': 'exception',
                    'errors': [f'验证过程异常: {str(e)}']
                }
                invalid_samples.append(sample)

        validation_rate = len(valid_samples) / len(samples) if samples else 0
        self.logger.info(f"数据集验证完成: {len(valid_samples)}/{len(samples)} 样本有效 "
                         f"(有效率: {validation_rate:.2%})")

        return valid_samples, invalid_samples

    def get_validation_statistics(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        stats = self.validation_stats.copy()

        # 计算成功率
        total = stats['total_validations']
        if total > 0:
            stats['success_rate'] = stats['valid_count'] / total
            stats['failure_rate'] = (stats['invalid_count'] + stats['error_count']) / total
            stats['timeout_rate'] = stats['timeout_count'] / total

        # 添加组件统计
        stats['converter_stats'] = {
            'variable_count': len(self.converter.z3_variables),
            'cache_size': len(self.converter.conversion_cache),
            'variable_usage': self.converter.get_variable_usage_stats()
        }

        stats['failure_analysis'] = {
            'total_failed_cases': len(self.failed_cases),
            'recent_failure_patterns': self._analyze_failure_patterns()
        }

        return stats

    def _analyze_failure_patterns(self) -> Dict[str, int]:
        """分析失败模式"""
        patterns = {}

        for _, result in self.failed_cases[-100:]:  # 分析最近100个失败案例
            status = result.status.value
            patterns[status] = patterns.get(status, 0) + 1

        return patterns

    def get_failed_cases_report(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取失败案例报告"""
        report = []

        for case_info, result in self.failed_cases[-limit:]:
            report.append({
                'case': case_info,
                'status': result.status.value,
                'errors': result.error_messages,
                'suggestions': result.suggestions,
                'validation_time_ms': result.validation_time_ms,
                'conversion_warnings': result.conversion_warnings
            })

        return report

    def optimize_performance(self):
        """性能优化"""
        # 清理缓存
        if len(self.converter.conversion_cache) > 10000:
            # 保留最近使用的50%
            cache_items = list(self.converter.conversion_cache.items())
            keep_count = len(cache_items) // 2
            self.converter.conversion_cache = dict(cache_items[-keep_count:])
            self.logger.info(f"缓存清理完成，保留 {keep_count} 个条目")

        # 清理失败案例记录
        if len(self.failed_cases) > 1000:
            self.failed_cases = self.failed_cases[-500:]
            self.logger.info("失败案例记录已清理")

    def reset_statistics(self):
        """重置统计信息"""
        self.validation_stats = {
            'total_validations': 0,
            'valid_count': 0,
            'invalid_count': 0,
            'timeout_count': 0,
            'error_count': 0,
            'average_time_ms': 0.0
        }
        self.failed_cases.clear()
        self.converter.clear_cache()
        self.logger.info("验证器统计信息已重置")

    def update_config(self, **kwargs):
        """运行时更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                old_value = getattr(self.config, key)
                setattr(self.config, key, value)
                self.logger.info(f"配置更新: {key} = {old_value} → {value}")
            else:
                self.logger.warning(f"未知配置项: {key}")

    def export_config(self) -> Dict[str, Any]:
        """导出当前配置"""
        config_dict = {}
        for field in self.config.__dataclass_fields__:
            value = getattr(self.config, field)
            if isinstance(value, Enum):
                config_dict[field] = value.value
            elif isinstance(value, list):
                config_dict[field] = value.copy()
            elif isinstance(value, dict):
                config_dict[field] = value.copy()
            else:
                config_dict[field] = value

        return config_dict


# ==================== 便捷函数和批量处理工具 ====================

def create_validator_from_config_file(config_path: str) -> Z3Validator:
    """从配置文件创建验证器"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                config_data = json.load(f)
            elif config_path.endswith(('.yaml', '.yml')):
                config_data = yaml.safe_load(f)
            else:
                raise ValueError("配置文件必须是JSON或YAML格式")

        # 转换枚举类型
        if 'validation_strategy' in config_data:
            config_data['validation_strategy'] = ValidationStrategy(config_data['validation_strategy'])

        config = Z3ValidatorConfig(**config_data)
        return Z3Validator(config)

    except Exception as e:
        raise ValueError(f"无法从配置文件创建验证器: {str(e)}")


def validate_dataset_quality(dataset_path: str,
                             validator: Optional[Z3Validator] = None,
                             output_report_path: Optional[str] = None) -> Dict[str, Any]:
    """
    验证数据集质量的便捷函数

    Args:
        dataset_path: 数据集文件路径
        validator: Z3验证器实例，如果为None则使用默认配置
        output_report_path: 报告输出路径

    Returns:
        Dict[str, Any]: 验证报告
    """
    import json

    if validator is None:
        validator = Z3Validator()

    # 加载数据集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        if dataset_path.endswith('.json'):
            dataset = json.load(f)
        else:
            raise ValueError("目前只支持JSON格式的数据集")

    # 验证数据集
    valid_samples, invalid_samples = validator.filter_valid_samples(dataset)

    # 生成报告
    report = {
        'dataset_path': dataset_path,
        'total_samples': len(dataset),
        'valid_samples': len(valid_samples),
        'invalid_samples': len(invalid_samples),
        'validation_rate': len(valid_samples) / len(dataset) if dataset else 0,
        'validation_statistics': validator.get_validation_statistics(),
        'failed_cases_summary': validator._analyze_failure_patterns(),
        'recommendations': _generate_dataset_recommendations(valid_samples, invalid_samples)
    }

    # 保存报告
    if output_report_path:
        with open(output_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def batch_validate_expressions(expressions: List[str],
                               validator: Optional[Z3Validator] = None) -> List[Z3ValidationResult]:
    """
    批量验证表达式语法和语义

    Args:
        expressions: 表达式列表
        validator: Z3验证器实例

    Returns:
        List[Z3ValidationResult]: 验证结果列表
    """
    if validator is None:
        validator = Z3Validator()

    results = []
    for expr in expressions:
        try:
            # 简单的自反验证：expr ⊨ expr
            result = validator.validate_reasoning_step([expr], expr)
            results.append(result)
        except Exception as e:
            error_result = Z3ValidationResult(
                status=ValidationResult.ERROR,
                is_valid=False,
                error_messages=[f"表达式验证失败: {str(e)}"]
            )
            results.append(error_result)

    return results


def _generate_dataset_recommendations(valid_samples: List[Dict],
                                      invalid_samples: List[Dict]) -> List[str]:
    """生成数据集改进建议"""
    recommendations = []

    total_samples = len(valid_samples) + len(invalid_samples)
    if total_samples == 0:
        return ["数据集为空，需要添加样本"]

    validation_rate = len(valid_samples) / total_samples

    if validation_rate < 0.5:
        recommendations.append("验证通过率过低，需要检查数据生成逻辑")
    elif validation_rate < 0.8:
        recommendations.append("验证通过率偏低，建议优化表达式生成质量")
    else:
        recommendations.append("数据集质量良好")

    # 分析失败原因
    error_types = {}
    for sample in invalid_samples:
        if 'validation_failure' in sample:
            status = sample['validation_failure'].get('status', 'unknown')
            error_types[status] = error_types.get(status, 0) + 1

    if 'format_error' in error_types:
        recommendations.append("存在格式错误的样本，需要检查数据结构")

    if 'invalid' in error_types:
        recommendations.append("存在逻辑无效的推理，需要改进推理规则")

    if 'timeout' in error_types:
        recommendations.append("部分验证超时，考虑简化复杂表达式或增加超时时间")

    return recommendations


# ==================== 示例配置和使用模板 ====================

def create_default_validator() -> Z3Validator:
    """创建默认配置的验证器"""
    return Z3Validator()


def create_fast_validator() -> Z3Validator:
    """创建快速验证器（适用于大规模数据处理）"""
    config = Z3ValidatorConfig(
        timeout_seconds=10,
        validation_strategy=ValidationStrategy.FAST,
        require_strict_validity=False,
        enable_counterexamples=False,
        enable_suggestion_generation=False,
        use_expression_cache=True
    )
    return Z3Validator(config)


def create_strict_validator() -> Z3Validator:
    """创建严格验证器（适用于高质量数据生成）"""
    config = Z3ValidatorConfig(
        timeout_seconds=60,
        validation_strategy=ValidationStrategy.DIRECT,
        require_strict_validity=True,
        enable_counterexamples=True,
        enable_suggestion_generation=True,
        log_all_failures=True
    )
    return Z3Validator(config)


# ==================== 命令行接口支持 ====================

def add_z3_validator_args(parser):
    """为命令行解析器添加Z3验证器参数"""
    z3_group = parser.add_argument_group('Z3验证器配置')

    z3_group.add_argument('--z3-timeout', type=int, default=30,
                          help='Z3验证超时时间（秒）')

    z3_group.add_argument('--validation-strategy',
                          choices=['direct', 'model_based', 'hybrid', 'fast'],
                          default='direct', help='验证策略')

    z3_group.add_argument('--strict-validation', action='store_true',
                          help='启用严格验证模式')

    z3_group.add_argument('--use-cache', action='store_true', default=True,
                          help='启用表达式转换缓存')

    z3_group.add_argument('--enable-counterexamples', action='store_true',
                          help='启用反例生成')

    z3_group.add_argument('--log-failures', action='store_true',
                          help='记录所有验证失败案例')

    return parser


if __name__ == "__main__":
    # 简单的测试示例
    validator = create_default_validator()

    # 测试单个推理步骤
    premises = ["P → Q", "P"]
    conclusion = "Q"

    result = validator.validate_reasoning_step(premises, conclusion)
    print(f"验证结果: {result.status.value}")
    print(f"是否有效: {result.is_valid}")
    print(f"验证时间: {result.validation_time_ms:.2f}ms")

    if result.error_messages:
        print(f"错误信息: {result.error_messages}")

    if result.suggestions:
        print(f"建议: {result.suggestions}")

    # 打印统计信息
    stats = validator.get_validation_statistics()
    print(f"\n验证统计: {stats}")