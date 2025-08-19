# 文件：utils/enhanced_prompt_builder.py
# 说明：增强的Prompt构建器，添加约束指导以确保一致性

from typing import Dict, List
import re


class EnhancedPromptBuilder:
    """增强的Prompt构建器，添加双向约束指导"""

    def __init__(self, base_template_path: str):
        with open(base_template_path, 'r', encoding='utf-8') as f:
            self.base_template = f.read()

    def build_constrained_prompt(
            self,
            z3_exprs: List[str],
            var_bindings: Dict[str, str],
            logical_steps: List[Dict],
            previous_violations: List[str] = None
    ) -> str:
        """
        构建包含约束指导的prompt

        Args:
            z3_exprs: Z3表达式列表
            var_bindings: 变量绑定字典
            logical_steps: 逻辑步骤列表
            previous_violations: 之前生成中的违规情况（用于指导改进）
        """

        # 1. 分析Z3结构，提取约束信息
        constraints = self._analyze_constraints(z3_exprs, var_bindings, logical_steps)

        # 2. 基于历史违规情况添加特定指导
        specific_guidance = self._build_specific_guidance(previous_violations)

        # 3. 构建增强的变量描述
        enhanced_var_desc = self._build_enhanced_variable_description(var_bindings, constraints)

        # 4. 构建逻辑约束描述
        logic_constraints = self._build_logic_constraints(constraints)

        # 5. 组装最终prompt
        enhanced_prompt = self._assemble_enhanced_prompt(
            z3_exprs, enhanced_var_desc, logical_steps,
            logic_constraints, specific_guidance
        )

        return enhanced_prompt

    def _analyze_constraints(self, z3_exprs: List[str], var_bindings: Dict[str, str],
                             logical_steps: List[Dict]) -> Dict:
        """分析生成约束"""
        constraints = {
            "variables_count": len(var_bindings),
            "must_use_all_variables": True,
            "logic_operations": [],
            "inference_chain_length": len(logical_steps),
            "requires_implication": False,
            "requires_conjunction": False
        }

        # 分析Z3表达式中的逻辑操作
        z3_text = " ".join(z3_exprs).lower()
        if "implies" in z3_text:
            constraints["logic_operations"].append("implication")
            constraints["requires_implication"] = True
        if "and(" in z3_text:
            constraints["logic_operations"].append("conjunction")
            constraints["requires_conjunction"] = True
        if "or(" in z3_text:
            constraints["logic_operations"].append("disjunction")

        return constraints

    def _build_specific_guidance(self, previous_violations: List[str]) -> str:
        """基于历史违规情况构建特定指导"""
        if not previous_violations:
            return ""

        guidance_parts = ["\n**特别注意（基于之前的生成问题）：**"]

        for violation in previous_violations:
            if "未提及的变量" in violation:
                guidance_parts.append("- 确保在场景描述中明确提到每一个变量的具体含义和作用")
            elif "语义域混淆" in violation:
                guidance_parts.append("- 选择一个统一的场景主题，所有变量都应该属于同一个语义域")
            elif "逻辑对应" in violation:
                guidance_parts.append("- 在自然语言中清楚地表达逻辑关系，如'如果...那么...'、'当...并且...'等")

        return "\n".join(guidance_parts)

    def _build_enhanced_variable_description(self, var_bindings: Dict[str, str], constraints: Dict) -> str:
        """构建增强的变量描述"""
        desc_parts = []

        desc_parts.append("**变量约束要求：**")
        desc_parts.append(f"- 必须使用所有 {constraints['variables_count']} 个变量")
        desc_parts.append("- 所有变量必须属于同一个场景域（如法律、教育、商业等）")
        desc_parts.append("- 每个变量都需要在场景中有明确、自然的体现")

        desc_parts.append("\n**变量及其语义角色：**")
        for var, semantic in var_bindings.items():
            desc_parts.append(f"- {var}: {semantic}")

        return "\n".join(desc_parts)

    def _build_logic_constraints(self, constraints: Dict) -> str:
        """Build high-level logical constraints aligned with rule library and distractor logic.
        Provides only directional and consistency requirements, without specifying concrete rules,
        fixed sentence patterns, or distractor strategies.
        """
        steps = constraints.get("inference_chain_length", 0)

        parts = []
        parts.append("**Logical Structure Requirements (aligned with rule library and distractors):**")
        parts.append(f"- The reasoning chain should contain around {steps} steps (±1 variation allowed).")
        parts.append(
            "- The reasoning process should unfold step by step: each conclusion depends only on the given premises or prior conclusions, with no jumps.")
        parts.append(
            "- Multiple logical relations may be used; do not restrict to any specific rule or fixed expression.")
        parts.append(
            "- Natural language should clearly reflect the reasoning process, but without enforcing fixed templates or repetitive phrasing.")
        parts.append(
            "- Variables and terms must remain consistent with the provided bindings; no undefined symbols are allowed.")
        parts.append(
            "- Exactly one option must be strictly derivable under the premises and rules; all other options should be non-derivable yet plausible.")
        parts.append(
            "- Distractors should achieve non-derivability through minimal necessary differences (the exact strategies are handled by the generator).")
        parts.append(
            "- Explanations must avoid revealing hints like 'correct' or 'incorrect'; maintain neutrality and variety in expression.")

        return "\n".join(parts)

    def _assemble_enhanced_prompt(
            self,
            z3_exprs: List[str],
            enhanced_var_desc: str,
            logical_steps: List[Dict],
            logic_constraints: str,
            specific_guidance: str
    ) -> str:
        """组装最终的增强prompt"""

        # 格式化逻辑步骤
        steps_desc = self._format_logical_steps(logical_steps)

        # 使用基础模板，但添加约束指导
        enhanced_prompt = self.base_template.format(
            z3_exprs="\n".join(z3_exprs),
            var_bindings=enhanced_var_desc,
            logical_steps=steps_desc
        )

        # 在Instructions部分之后插入约束指导
        constraints_section = f"""

{logic_constraints}

{specific_guidance}

**一致性检查清单：**
- [ ] 所有变量都在场景中被使用
- [ ] 场景属于单一语义域
- [ ] 逻辑关系在自然语言中清晰表达
- [ ] 推理链完整且连贯"""

        # 在现有Instructions后插入约束部分
        enhanced_prompt = enhanced_prompt.replace(
            "Instructions:",
            f"Instructions:{constraints_section}\n\n原始要求:"
        )

        return enhanced_prompt

    def _format_logical_steps(self, logical_steps: List[Dict]) -> str:
        """格式化逻辑步骤描述"""
        formatted_steps = []
        for i, step in enumerate(logical_steps, 1):
            rule = step.get('rule', 'Unknown')
            conclusion = step.get('conclusion', '')
            premises = step.get('premises', [])

            if len(premises) == 1:
                premise_desc = premises[0]
            else:
                premise_desc = f"({', '.join(premises)})"

            formatted_steps.append(
                f"Step {i}: 基于 {premise_desc} 通过 {rule} 推出 {conclusion}"
            )

        return "\n".join(formatted_steps)