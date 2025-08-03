# 文件：dataset_generator.py
# 说明：基于真实逻辑规则的LSAT风格数据集生成器（重构版）

import json
import logging
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

from dag.dag_builder import build_reasoning_dag
from dag.validator import validate_logical_steps
from distractor.generator import DistractorGenerator
from api_key.llm_dispatcher import LLMDispatcher
from utils.consistency_validator import ConsistencyValidator
from utils.enhanced_prompt_builder import EnhancedPromptBuilder
from utils.safe_variable_extractor import SafeVariableExtractor


class DatasetGenerator:
    """
    改进的数据集生成器：基于真实逻辑规则生成高质量LSAT风格题目
    """

    def __init__(self, llm_dispatcher: LLMDispatcher, prompt_template_path: str):
        """
        初始化数据集生成器

        :param llm_dispatcher: LLM调度器实例
        :param prompt_template_path: prompt模板文件路径
        """
        self.llm = llm_dispatcher
        self.logger = logging.getLogger("improved_dataset_generator")
        self.extractor = SafeVariableExtractor()
        self.validator = ConsistencyValidator(strictness_level="medium")
        self.prompt_builder = EnhancedPromptBuilder(prompt_template_path)

        # 质量控制参数
        self.max_retry_attempts = 5
        self.min_valid_steps = 2
        self.max_valid_steps = 6

    def _extract_variables_from_dag(self, root_node) -> List[str]:
        """从DAG中提取所有变量名"""
        return self.extractor.extract_from_dag(root_node)

    def _generate_semantic_bindings(self, variables: List[str]) -> Dict[str, str]:
        """为变量生成语义绑定，确保语义域一致性"""

        # 扩展的语义域，每个域都包含完整的逻辑场景
        semantic_domains = {
            "academic_evaluation": {
                "domain_name": "学术评估",
                "variables": [
                    "passed_midterm_exam", "submitted_research_paper", "attended_seminars",
                    "completed_assignments", "received_recommendation", "qualified_for_thesis",
                    "defended_thesis", "earned_degree", "published_paper", "won_scholarship"
                ],
                "context_template": "学术评估系统中的学生表现评价"
            },
            "business_workflow": {
                "domain_name": "商业流程",
                "variables": [
                    "project_approved", "budget_allocated", "team_assembled",
                    "milestone_completed", "client_satisfied", "contract_signed",
                    "payment_received", "quality_assured", "deadline_met", "profit_achieved"
                ],
                "context_template": "企业项目管理和业务流程"
            },
            "legal_procedure": {
                "domain_name": "法律程序",
                "variables": [
                    "evidence_submitted", "witness_testified", "case_filed",
                    "hearing_scheduled", "motion_granted", "settlement_reached",
                    "judgment_rendered", "appeal_filed", "precedent_cited", "verdict_delivered"
                ],
                "context_template": "法律诉讼程序和案件处理"
            },
            "medical_diagnosis": {
                "domain_name": "医疗诊断",
                "variables": [
                    "symptoms_observed", "tests_conducted", "results_analyzed",
                    "diagnosis_confirmed", "treatment_prescribed", "patient_responded",
                    "recovery_noted", "followup_scheduled", "clearance_given", "discharge_approved"
                ],
                "context_template": "医疗诊断和治疗流程"
            },
            "certification_process": {
                "domain_name": "认证流程",
                "variables": [
                    "training_completed", "exam_passed", "experience_verified",
                    "application_submitted", "review_conducted", "interview_passed",
                    "certification_granted", "license_issued", "renewal_required", "compliance_met"
                ],
                "context_template": "专业认证和资质获取流程"
            }
        }

        # 随机选择一个语义域
        domain_key = random.choice(list(semantic_domains.keys()))
        domain = semantic_domains[domain_key]

        self.logger.info(f"选择语义域: {domain['domain_name']}")

        bindings = {}
        domain_vars = domain["variables"]

        # 为每个变量分配语义绑定
        for i, var in enumerate(variables[:len(domain_vars)]):
            bindings[var] = domain_vars[i]

        # 如果变量太多，使用通用命名
        if len(variables) > len(domain_vars):
            for i, var in enumerate(variables[len(domain_vars):], 1):
                bindings[var] = f"{domain['domain_name']}_additional_condition_{i}"

        return bindings

    def _format_z3_expressions_improved(self, logical_steps: List[Dict], var_bindings: Dict[str, str]) -> List[str]:
        """改进的Z3表达式格式化"""
        z3_exprs = []

        try:
            # 1. 变量声明（使用语义化变量名）
            for var, semantic in var_bindings.items():
                z3_exprs.append(f"{semantic} = Bool('{semantic}')")

            # 2. 规则表达式（改进错误处理）
            for step in logical_steps:
                try:
                    premises_expr = step.get('premises_expr', [])
                    conclusion_expr = step.get('conclusion_expr')
                    rule_name = step.get('rule', 'Unknown')

                    if not premises_expr or conclusion_expr is None:
                        continue

                    # 安全地转换表达式为字符串
                    premise_strs = []
                    for premise in premises_expr:
                        try:
                            # 将Z3变量名替换为语义名
                            premise_str = str(premise)
                            for var, semantic in var_bindings.items():
                                premise_str = premise_str.replace(var, semantic)
                            premise_strs.append(premise_str)
                        except Exception as e:
                            self.logger.debug(f"转换前提表达式失败: {e}")
                            continue

                    if not premise_strs:
                        continue

                    # 转换结论表达式
                    try:
                        conclusion_str = str(conclusion_expr)
                        for var, semantic in var_bindings.items():
                            conclusion_str = conclusion_str.replace(var, semantic)
                    except Exception as e:
                        self.logger.debug(f"转换结论表达式失败: {e}")
                        continue

                    # 构造蕴含关系
                    if len(premise_strs) == 1:
                        z3_exprs.append(f"Implies({premise_strs[0]}, {conclusion_str})")
                    else:
                        premises_conjunction = f"And({', '.join(premise_strs)})"
                        z3_exprs.append(f"Implies({premises_conjunction}, {conclusion_str})")

                except Exception as e:
                    self.logger.debug(f"处理逻辑步骤时出错: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"格式化Z3表达式时出错: {e}")

        return z3_exprs

    def _create_enhanced_distractors(self, logical_steps: List[Dict], var_bindings: Dict[str, str]) -> List[str]:
        """创建增强的干扰项"""
        try:
            # 创建安全的布尔变量
            var_names = list(var_bindings.keys())
            safe_vars = self.extractor.create_safe_bool_vars(var_names)

            if not safe_vars:
                self.logger.warning("无法创建有效的布尔变量，跳过干扰项生成")
                return self._create_fallback_distractors()

            # 生成干扰项
            distractor_gen = DistractorGenerator(
                available_vars=safe_vars,
                enabled_strategies=["illogical_reasoning", "adversarial_structure", "reversed_implication"]
            )

            distractors = distractor_gen.generate_all(logical_steps, num_per_strategy=2)

            # 转换为自然语言描述
            distractor_descriptions = []
            for d in distractors[:3]:  # 最多3个干扰项
                desc = d.get('description', f"基于{d.get('strategy', 'unknown')}策略的干扰项")
                distractor_descriptions.append(desc)

            return distractor_descriptions

        except Exception as e:
            self.logger.warning(f"生成干扰项失败: {e}")
            return self._create_fallback_distractors()

    def _create_fallback_distractors(self) -> List[str]:
        """创建备用干扰项"""
        return [
            "基于不完整前提的错误推理",
            "逻辑方向颠倒的错误结论",
            "无关条件的干扰性推断"
        ]

    def generate_single_sample(self, max_depth: int = 3) -> Optional[Dict[str, Any]]:
        """生成单个数据样本（支持任意长度链条）"""
        for attempt in range(self.max_retry_attempts):
            try:
                # 1. 构建推理DAG（自动选择短链条或长链条）
                self.logger.info(f"构建推理DAG，深度={max_depth}，尝试{attempt + 1}")
                root, logical_steps = build_reasoning_dag(max_depth=max_depth, min_depth=max(max_depth // 3, 2))

                if not logical_steps:
                    self.logger.warning("未能生成逻辑步骤，重试")
                    continue

                # 2. 验证逻辑步骤
                valid_steps, failed_steps = validate_logical_steps(logical_steps)

                if len(valid_steps) < self.min_valid_steps:
                    self.logger.warning(f"有效步骤太少 ({len(valid_steps)} < {self.min_valid_steps})，重试")
                    continue

                if len(valid_steps) > self.max_valid_steps:
                    valid_steps = valid_steps[:self.max_valid_steps]

                self.logger.info(f"成功验证 {len(valid_steps)} 个逻辑步骤")

                # 3. 提取变量和生成语义绑定
                variables = self._extract_variables_from_dag(root)
                if not variables:
                    self.logger.warning("未能提取到变量，重试")
                    continue

                var_bindings = self._generate_semantic_bindings(variables)
                self.logger.info(f"生成 {len(var_bindings)} 个变量绑定")

                # 4. 构建增强prompt
                enhanced_prompt = self.prompt_builder.build_constrained_prompt(
                    z3_exprs=self._format_z3_expressions_improved(valid_steps, var_bindings),
                    var_bindings=var_bindings,
                    logical_steps=valid_steps,
                    previous_violations=[]
                )

                # 5. 调用LLM
                self.logger.info("调用LLM生成题目...")
                response = self.llm.call(enhanced_prompt)

                if not response:
                    self.logger.error("LLM响应为空，重试")
                    continue

                # 6. 解析响应
                sample = self._parse_llm_response_improved(response, valid_steps, var_bindings)
                if not sample:
                    self.logger.error("响应解析失败，重试")
                    continue

                # 7. 质量验证
                is_valid, violations = self.validator.validate_sample(sample)

                if is_valid:
                    self.logger.info("✅ 样本通过质量验证")
                    return self._add_enhanced_metadata(sample, valid_steps, var_bindings)
                else:
                    self.logger.warning(f"⚠️ 质量验证失败: {violations}")
                    # 在最后一次尝试时，返回部分合格的样本
                    if attempt == self.max_retry_attempts - 1:
                        sample['validation_warnings'] = violations
                        return self._add_enhanced_metadata(sample, valid_steps, var_bindings)

            except Exception as e:
                self.logger.error(f"生成样本时出错 (尝试 {attempt + 1}): {e}")
                continue

        self.logger.error("所有重试都失败")
        return None

    def _parse_llm_response_improved(self, response: str, valid_steps: List[Dict], var_bindings: Dict[str, str]) -> \
    Optional[Dict]:
        """改进的LLM响应解析"""
        try:
            # 提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end <= json_start:
                self.logger.error("响应中未找到有效JSON")
                return None

            json_str = response[json_start:json_end]

            # 解析JSON
            result = json.loads(json_str)

            # 验证必需字段
            required_fields = ["context", "question", "answers", "label", "z3"]
            for field in required_fields:
                if field not in result:
                    self.logger.error(f"响应缺少必需字段: {field}")
                    return None

            # 验证answers格式
            if not isinstance(result["answers"], list) or len(result["answers"]) != 4:
                self.logger.error(f"答案选项格式错误: {result.get('answers')}")
                return None

            # 验证label格式
            if result["label"] not in ["A", "B", "C", "D"]:
                self.logger.error(f"标签格式错误: {result.get('label')}")
                return None

            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}")
            # 尝试修复常见的JSON错误
            return self._try_fix_json(json_str)
        except Exception as e:
            self.logger.error(f"解析响应时出错: {e}")
            return None

    def _try_fix_json(self, json_str: str) -> Optional[Dict]:
        """尝试修复常见的JSON错误"""
        try:
            # 修复常见问题：多余的逗号、引号问题等
            fixed_json = json_str.replace(',]', ']').replace(',}', '}')
            # 尝试再次解析
            return json.loads(fixed_json)
        except:
            return None

    def _add_enhanced_metadata(self, sample: Dict, valid_steps: List[Dict], var_bindings: Dict[str, str]) -> Dict:
        """添加增强的元数据"""
        sample['metadata'] = {
            'reasoning_depth': len(valid_steps),
            'variables_count': len(var_bindings),
            'rules_used': [step.get('rule', 'Unknown') for step in valid_steps],
            'semantic_domain': self._infer_semantic_domain(var_bindings),
            'logical_complexity': self._calculate_complexity(valid_steps),
            'generation_version': 'improved_v2'
        }

        # 添加质量分数
        sample['quality_score'] = self._calculate_quality_score(sample)

        return sample

    def _infer_semantic_domain(self, var_bindings: Dict[str, str]) -> str:
        """推断语义域"""
        all_bindings = " ".join(var_bindings.values()).lower()

        domain_keywords = {
            "academic": ["exam", "assignment", "grade", "course", "student"],
            "business": ["project", "budget", "client", "contract", "profit"],
            "legal": ["evidence", "witness", "case", "court", "judgment"],
            "medical": ["diagnosis", "treatment", "patient", "symptoms"],
            "certification": ["training", "certification", "license", "compliance"]
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in all_bindings for keyword in keywords):
                return domain

        return "general"

    def _calculate_complexity(self, valid_steps: List[Dict]) -> str:
        """计算逻辑复杂度"""
        step_count = len(valid_steps)

        if step_count <= 2:
            return "simple"
        elif step_count <= 4:
            return "medium"
        else:
            return "complex"

    def _calculate_quality_score(self, sample: Dict) -> float:
        """计算质量分数 (0-1)"""
        score = 0.0

        # 基础分数
        score += 0.3

        # 语义一致性
        if 'validation_warnings' not in sample:
            score += 0.3

        # 逻辑深度
        depth = sample.get('metadata', {}).get('reasoning_depth', 0)
        score += min(depth / 6, 0.2)  # 最多0.2分

        # 变量使用
        var_count = sample.get('metadata', {}).get('variables_count', 0)
        score += min(var_count / 8, 0.2)  # 最多0.2分

        return min(score, 1.0)

    def generate_dataset(self, num_samples: int, output_path: str, max_depth_range: tuple = (5, 12)) -> None:
        """生成完整数据集（改进版）"""
        self.logger.info(f"开始生成 {num_samples} 个样本的改进数据集")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        successful_samples = []
        attempts = 0
        max_attempts = num_samples * 4  # 增加最大尝试次数

        # 统计信息
        stats = {
            "total_attempts": 0,
            "successful": 0,
            "failed": 0,
            "quality_scores": [],
            "semantic_domains": {},
            "complexity_levels": {}
        }

        while len(successful_samples) < num_samples and attempts < max_attempts:
            attempts += 1
            stats["total_attempts"] += 1

            depth = random.randint(*max_depth_range)

            self.logger.info(f"生成样本 {len(successful_samples) + 1}/{num_samples} (尝试 {attempts})")

            sample = self.generate_single_sample(max_depth=depth)

            if sample:
                successful_samples.append(sample)
                stats["successful"] += 1

                # 收集统计信息
                quality_score = sample.get('quality_score', 0)
                stats["quality_scores"].append(quality_score)

                domain = sample.get('metadata', {}).get('semantic_domain', 'unknown')
                stats["semantic_domains"][domain] = stats["semantic_domains"].get(domain, 0) + 1

                complexity = sample.get('metadata', {}).get('logical_complexity', 'unknown')
                stats["complexity_levels"][complexity] = stats["complexity_levels"].get(complexity, 0) + 1

                self.logger.info(f"✅ 成功生成样本 {len(successful_samples)} (质量分数: {quality_score:.2f})")
            else:
                stats["failed"] += 1
                self.logger.warning(f"❌ 样本生成失败 (尝试 {attempts})")

        # 保存数据集
        self._save_dataset_with_stats(successful_samples, stats, output_path)

        # 输出最终统计
        self._print_final_statistics(stats)

    def _save_dataset_with_stats(self, samples: List[Dict], stats: Dict, output_path: str):
        """保存数据集并包含统计信息"""
        # 保存数据集
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # 保存统计信息
        stats_path = output_path.replace('.jsonl', '_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self.logger.info(f"数据集已保存: {output_path}")
        self.logger.info(f"统计信息已保存: {stats_path}")

    def _print_final_statistics(self, stats: Dict):
        """打印最终统计信息"""
        success_rate = stats["successful"] / stats["total_attempts"] * 100 if stats["total_attempts"] > 0 else 0
        avg_quality = sum(stats["quality_scores"]) / len(stats["quality_scores"]) if stats["quality_scores"] else 0

        self.logger.info("=" * 50)
        self.logger.info("📊 数据集生成统计")
        self.logger.info(f"成功率: {success_rate:.1f}% ({stats['successful']}/{stats['total_attempts']})")
        self.logger.info(f"平均质量分数: {avg_quality:.3f}")
        self.logger.info(f"语义域分布: {stats['semantic_domains']}")
        self.logger.info(f"复杂度分布: {stats['complexity_levels']}")
        self.logger.info("=" * 50)


def main():
    """主函数示例"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 初始化LLM调度器
    llm = LLMDispatcher(
        model_name="deepseek-chat",
        api_key_path="api_key/ds-api_key.txt",
        retries=3
    )

    # 初始化改进的数据集生成器
    generator = DatasetGenerator(
        llm_dispatcher=llm,
        prompt_template_path="prompt/lsat_prompt.txt"
    )

    # 生成数据集
    generator.generate_dataset(
        num_samples=1,
        output_path="output/unified_lsat_dataset.jsonl",
        max_depth_range=(2, 5)  # 支持6-12步的推理链
    )


if __name__ == "__main__":
    main()