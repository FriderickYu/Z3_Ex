# 文件：utils/consistency_validator.py
# 说明：双向约束验证器，确保Z3逻辑、变量绑定和自然语言的一致性

import re
import json
from typing import Dict, List, Set, Tuple, Optional
import logging


class ConsistencyValidator:
    """双向约束验证器：确保逻辑结构与自然语言描述的一致性"""

    def __init__(self, strictness_level: str = "medium"):
        """
        初始化约束验证器

        Args:
            strictness_level: 验证严格程度
                - "strict": 严格验证，所有问题都报告
                - "medium": 中等验证，只报告明显问题
                - "lenient": 宽松验证，只报告严重问题
        """
        self.logger = logging.getLogger("consistency_validator")
        self.strictness = strictness_level

    def validate_sample(self, sample: Dict) -> Tuple[bool, List[str]]:
        """
        验证样本的一致性
        返回：(是否通过, 违规原因列表)
        """
        violations = []

        # 1. 基础结构验证（所有严格程度都执行）
        basic_violations = self._validate_basic_structure(sample)
        violations.extend(basic_violations)

        # 2. 变量完整性验证（所有严格程度都执行）
        variable_violations = self._validate_variable_completeness(sample)
        violations.extend(variable_violations)

        # 3. 语义一致性验证（根据严格程度调整）
        if self.strictness in ["strict", "medium"]:
            semantic_violations = self._validate_semantic_consistency(sample)
            violations.extend(semantic_violations)

        # 4. 逻辑结构对应验证（strict模式才执行）
        if self.strictness == "strict":
            logic_violations = self._validate_logic_correspondence(sample)
            violations.extend(logic_violations)

        is_valid = len(violations) == 0
        return is_valid, violations

    def _validate_basic_structure(self, sample: Dict) -> List[str]:
        """验证基础JSON结构"""
        violations = []
        required_fields = ["context", "question", "answers", "label", "z3"]

        for field in required_fields:
            if field not in sample:
                violations.append(f"缺少必需字段: {field}")

        # 验证answers数量
        if "answers" in sample and len(sample["answers"]) != 4:
            violations.append(f"答案选项数量错误: {len(sample['answers'])}, 应为4")

        return violations

    def _validate_variable_completeness(self, sample: Dict) -> List[str]:
        """验证变量使用的完整性"""
        violations = []

        try:
            # 从Z3表达式中提取变量
            z3_vars = self._extract_variables_from_z3(sample.get("z3", []))

            # 从自然语言中提取变量引用
            context = sample.get("context", "")
            question = sample.get("question", "")
            answers = " ".join(sample.get("answers", []))
            nl_content = f"{context} {question} {answers}"

            nl_vars = self._extract_variables_from_text(nl_content)

            # 检查是否所有Z3变量都在自然语言中被提及
            missing_vars = z3_vars - nl_vars
            if missing_vars:
                violations.append(f"自然语言中未提及的变量: {sorted(missing_vars)}")

            # 检查自然语言中是否有多余的变量引用
            extra_vars = nl_vars - z3_vars
            if extra_vars:
                violations.append(f"Z3中未定义的变量引用: {sorted(extra_vars)}")

        except Exception as e:
            violations.append(f"变量完整性验证失败: {e}")

        return violations

    def _validate_semantic_consistency(self, sample: Dict) -> List[str]:
        """验证语义一致性：检查变量绑定是否合理"""
        violations = []

        try:
            context = sample.get("context", "").lower()

            # 扩展语义域关键词，包含更多常见词汇
            semantic_domains = {
                "legal": ["court", "evidence", "witness", "contract", "case", "judge", "law", "trial", "legal",
                          "attorney", "plaintiff", "defendant", "lawsuit", "ruling"],
                "academic": ["student", "exam", "assignment", "grade", "course", "teacher", "class", "school",
                             "university", "study", "homework", "test", "education", "learning"],
                "business": ["project", "budget", "deadline", "client", "company", "meeting", "business", "office",
                             "manager", "employee", "contract", "sales", "profit"],
                "medical": ["patient", "diagnosis", "treatment", "doctor", "hospital", "medicine", "health", "clinic",
                            "nurse", "therapy", "medical", "symptom"],
                "general": ["condition", "requirement", "process", "system", "procedure", "rule", "policy", "standard",
                            "criteria"]  # 新增通用域
            }

            # 检测主要语义域
            domain_scores = {}
            for domain, keywords in semantic_domains.items():
                score = sum(1 for keyword in keywords if keyword in context)
                if score > 0:
                    domain_scores[domain] = score

            # 放宽语义域检查：如果有general域或任何明确域，都视为可接受
            if not domain_scores:
                violations.append("无法识别明确的语义域，内容可能过于抽象")
            elif len(domain_scores) > 1:
                # 排除general域后检查冲突
                non_general_domains = {k: v for k, v in domain_scores.items() if k != "general"}
                if len(non_general_domains) > 1:
                    max_score = max(non_general_domains.values())
                    high_score_domains = [d for d, s in non_general_domains.items() if s >= max_score * 0.8]
                    if len(high_score_domains) > 1:
                        violations.append(f"语义域混淆，同时涉及: {high_score_domains}")

        except Exception as e:
            violations.append(f"语义一致性验证失败: {e}")

        return violations

    def _validate_logic_correspondence(self, sample: Dict) -> List[str]:
        """验证逻辑结构对应关系"""
        violations = []

        try:
            context = sample.get("context", "")
            question = sample.get("question", "")
            answers = " ".join(sample.get("answers", []))
            full_text = f"{context} {question} {answers}".lower()

            z3_exprs = sample.get("z3", [])

            # 更宽松的逻辑词汇模式
            logic_patterns = {
                "implication": [
                    r"if\s+.*\s+then", r"when\s+.*\s+will", r"implies?", r"therefore",
                    r"thus", r"consequently", r"as\s+a\s+result", r"so\s+", r"can\s+be\s+concluded",
                    r"leads?\s+to", r"results?\s+in", r"causes?", r"ensures?", r"guarantees?"
                ],
                "conjunction": [
                    r"\s+and\s+", r"both\s+.*\s+and", r"as\s+well\s+as", r"together\s+with",
                    r"along\s+with", r"in\s+addition\s+to", r"combined\s+with"
                ],
                "disjunction": [
                    r"\s+or\s+", r"either\s+.*\s+or", r"alternatively", r"otherwise"
                ],
                "negation": [
                    r"\s+not\s+", r"\s+no\s+", r"never", r"cannot", r"won't", r"isn't",
                    r"doesn't", r"haven't", r"hasn't", r"wouldn't", r"shouldn't"
                ]
            }

            # 统计Z3中的逻辑操作符
            z3_text = " ".join(z3_exprs).lower()
            z3_logic_ops = {
                "implication": z3_text.count("implies"),
                "conjunction": z3_text.count("and("),
                "disjunction": z3_text.count("or("),
                "negation": z3_text.count("not(")
            }

            # 统计自然语言中的逻辑词汇
            nl_logic_ops = {}
            for op_type, patterns in logic_patterns.items():
                count = 0
                for pattern in patterns:
                    matches = re.findall(pattern, full_text)
                    count += len(matches)
                nl_logic_ops[op_type] = count

            # 更宽松的检查：只在明显缺失时报告问题
            for op_type, z3_count in z3_logic_ops.items():
                if z3_count > 1 and nl_logic_ops.get(op_type, 0) == 0:  # 只有当Z3中有多个该操作时才要求自然语言体现
                    violations.append(f"Z3中包含多个{op_type}操作，但自然语言中完全未体现")

        except Exception as e:
            violations.append(f"逻辑对应验证失败: {e}")

        return violations

    def _extract_variables_from_z3(self, z3_exprs: List[str]) -> Set[str]:
        """从Z3表达式中提取变量名"""
        variables = set()
        for expr in z3_exprs:
            # 匹配变量定义：Var_N = Bool('Var_N')
            var_matches = re.findall(r'Var_\d+', expr)
            variables.update(var_matches)
        return variables

    def _extract_variables_from_text(self, text: str) -> Set[str]:
        """从自然语言文本中提取变量引用"""
        # 查找所有Var_N格式的引用
        variables = set(re.findall(r'Var_\d+', text))
        return variables

    def suggest_fixes(self, violations: List[str]) -> Dict[str, str]:
        """基于违规情况建议修复方案"""
        suggestions = {}

        for violation in violations:
            if "未提及的变量" in violation:
                suggestions["variable_completeness"] = "需要在prompt中强调使用所有变量"
            elif "语义域混淆" in violation:
                suggestions["semantic_consistency"] = "需要选择单一的场景主题"
            elif "逻辑对应" in violation:
                suggestions["logic_correspondence"] = "需要在自然语言中明确表达逻辑关系"

        return suggestions