# 文件：dataset_generator.py
# 说明：基于真实逻辑规则的LSAT风格数据集生成器（集成matplotlib可视化）

import json
import logging
import random
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from dag.dag_builder import build_reasoning_dag
from dag.validator import validate_logical_steps
from dag.visualizer import visualize_dag
from distractor.generator import DistractorGenerator
from api_key.llm_dispatcher import LLMDispatcher
from utils.consistency_validator import ConsistencyValidator
from utils.enhanced_prompt_builder import EnhancedPromptBuilder
from utils.variable_manager import EnhancedVariableExtractor


class DatasetGenerator:
    """
    改进的数据集生成器：严格控制变量数量，确保高质量的推理题目
    集成：matplotlib + networkx 可视化，无外部依赖
    """

    def __init__(self, llm_dispatcher: LLMDispatcher, prompt_template_path: str,
                 max_variables: int = 8, min_variables: int = 4,
                 enable_visualization: bool = True, viz_output_dir: str = "output/dag_visualizations"):
        """
        初始化数据集生成器

        :param llm_dispatcher: LLM调度器实例
        :param prompt_template_path: prompt模板文件路径
        :param max_variables: 最大变量数量
        :param min_variables: 最小变量数量
        :param enable_visualization: 是否启用可视化
        :param viz_output_dir: 可视化图片输出目录
        """
        self.llm = llm_dispatcher
        self.logger = logging.getLogger("dataset_generator")

        # 使用改进的变量提取器（带数量控制）
        self.extractor = EnhancedVariableExtractor(
            max_variables=max_variables,
            min_variables=min_variables
        )
        self.validator = ConsistencyValidator(strictness_level="medium")
        self.prompt_builder = EnhancedPromptBuilder(prompt_template_path)

        # 质量控制参数
        self.max_retry_attempts = 5
        self.min_valid_steps = 2
        self.max_valid_steps = 6

        # 变量控制参数
        self.max_variables = max_variables
        self.min_variables = min_variables

        # 可视化参数
        self.enable_visualization = enable_visualization
        self.viz_output_dir = viz_output_dir

        # 创建可视化输出目录
        if self.enable_visualization:
            Path(self.viz_output_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"✅ 可视化功能已启用，输出目录: {self.viz_output_dir}")

    def _generate_dag_visualization(self, root_node, sample_id: str, metadata: Dict = None) -> Optional[str]:
        """
        为DAG生成可视化图片

        :param root_node: DAG根节点
        :param sample_id: 样本唯一标识
        :param metadata: 样本元数据（用于文件命名）
        :return: 生成的图片文件路径，失败时返回None
        """
        if not self.enable_visualization or root_node is None:
            return None

        try:
            # 构造文件名
            if metadata:
                depth = metadata.get('reasoning_depth', 0)
                var_count = metadata.get('variables_count', 0)
                filename = f"dag_sample_{sample_id}_depth{depth}_vars{var_count}"
            else:
                filename = f"dag_sample_{sample_id}"

            # 完整输出路径（不包含扩展名，visualize_dag会自动添加）
            output_path = os.path.join(self.viz_output_dir, filename)

            # 调用可视化函数（使用现代风格）
            visualize_dag(root_node, filename=output_path, format="png", style="modern")

            final_path = f"{output_path}.png"
            self.logger.info(f"🎨 DAG可视化已生成: {final_path}")

            return final_path

        except Exception as e:
            self.logger.warning(f"⚠️ 生成DAG可视化失败: {e}")
            return None

    def _extract_variables_from_dag(self, root_node) -> List[str]:
        """从DAG中提取变量（带数量控制）"""
        try:
            variables = self.extractor.extract_from_dag(root_node)

            if variables:
                stats = self.extractor.get_variable_statistics(variables)
                self.logger.info(f"变量提取统计: {stats['total_count']} 个变量")
                self.logger.debug(f"详细统计: {stats}")

                # 检查变量数量是否在合理范围内
                if stats['total_count'] > self.max_variables:
                    self.logger.warning(f"变量数量过多 ({stats['total_count']} > {self.max_variables})")
                elif stats['total_count'] < self.min_variables:
                    self.logger.warning(f"变量数量太少 ({stats['total_count']} < {self.min_variables})")
                else:
                    self.logger.info(f"✅ 变量数量合适: {stats['total_count']} 个")

                # 规范化变量名
                normalized_vars = self.extractor.normalize_variable_names(variables)
                self.logger.debug(f"提取到变量: {normalized_vars}")

                return normalized_vars
            else:
                self.logger.warning("未能从DAG中提取到任何变量")
                return []

        except Exception as e:
            self.logger.error(f"变量提取失败: {e}")
            return []

    def _extract_variables_from_steps(self, logical_steps: List[Dict]) -> List[str]:
        """从逻辑步骤中提取变量（备用方法，带数量控制）"""
        try:
            variables = self.extractor.extract_from_steps(logical_steps)
            if variables:
                stats = self.extractor.get_variable_statistics(variables)
                self.logger.info(f"从步骤中提取 {stats['total_count']} 个变量")

                normalized_vars = self.extractor.normalize_variable_names(variables)
                self.logger.debug(f"从步骤中提取到变量: {normalized_vars}")
                return normalized_vars
            return []
        except Exception as e:
            self.logger.error(f"从步骤提取变量失败: {e}")
            return []

    def _generate_semantic_bindings(self, variables: List[str]) -> Dict[str, str]:
        """
        为变量生成增强的语义绑定，确保语义域一致性
        特别针对变量数量控制后的情况优化
        """

        # 扩展的语义域，提供更多高质量的语义术语
        semantic_domains = {
            "academic_evaluation": {
                "domain_name": "学术评估",
                "variables": [
                    "completed_coursework", "passed_examinations", "submitted_thesis",
                    "attended_seminars", "received_approval", "qualified_for_degree",
                    "defended_research", "earned_certification", "published_work",
                    "won_recognition", "met_requirements", "achieved_standards"
                ],
                "context_template": "学术评估系统中的学生表现评价"
            },
            "business_workflow": {
                "domain_name": "商业流程",
                "variables": [
                    "project_initiated", "budget_approved", "team_assembled",
                    "milestone_achieved", "client_satisfied", "contract_finalized",
                    "payment_processed", "quality_verified", "deadline_met",
                    "profit_realized", "objectives_completed", "standards_exceeded"
                ],
                "context_template": "企业项目管理和业务流程"
            },
            "legal_procedure": {
                "domain_name": "法律程序",
                "variables": [
                    "evidence_presented", "witness_examined", "case_documented",
                    "hearing_conducted", "motion_approved", "settlement_negotiated",
                    "judgment_issued", "appeal_processed", "precedent_established",
                    "verdict_finalized", "ruling_upheld", "procedure_followed"
                ],
                "context_template": "法律诉讼程序和案件处理"
            },
            "medical_diagnosis": {
                "domain_name": "医疗诊断",
                "variables": [
                    "symptoms_documented", "tests_performed", "results_interpreted",
                    "diagnosis_established", "treatment_prescribed", "patient_improved",
                    "recovery_achieved", "followup_completed", "clearance_obtained",
                    "discharge_authorized", "medication_effective", "vitals_stable"
                ],
                "context_template": "医疗诊断和治疗流程"
            },
            "certification_process": {
                "domain_name": "认证流程",
                "variables": [
                    "training_finished", "examination_passed", "experience_documented",
                    "application_processed", "review_completed", "interview_cleared",
                    "certification_awarded", "license_granted", "renewal_scheduled",
                    "compliance_verified", "standards_met", "credentials_validated"
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

        # 为每个变量分配语义绑定（现在变量数量已控制在合理范围内）
        for i, var in enumerate(variables):
            if i < len(domain_vars):
                # 直接使用域内的语义术语
                bindings[var] = domain_vars[i]
            else:
                # 如果变量数量超过域内术语，使用更具体的命名而不是通用格式
                extra_terms = [
                    f"additional_{domain['domain_name'].lower()}_requirement",
                    f"supplementary_{domain['domain_name'].lower()}_condition",
                    f"extended_{domain['domain_name'].lower()}_criterion",
                    f"further_{domain['domain_name'].lower()}_standard"
                ]
                extra_index = (i - len(domain_vars)) % len(extra_terms)
                bindings[var] = f"{extra_terms[extra_index]}_{i - len(domain_vars) + 1}"

        return bindings

    def _format_z3_expressions(self, logical_steps: List[Dict], var_bindings: Dict[str, str]) -> List[str]:
        """改进的Z3表达式格式化（优化变量数量控制后的情况）"""
        z3_exprs = []

        try:
            # 1. 变量声明（使用语义化变量名）
            for var, semantic in var_bindings.items():
                z3_exprs.append(f"{semantic} = Bool('{semantic}')")

            # 2. 规则表达式（优化处理）
            processed_expressions = set()  # 避免重复表达式

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
                            premise_str = str(premise)
                            # 替换变量名为语义名
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
                        implication = f"Implies({premise_strs[0]}, {conclusion_str})"
                    else:
                        premises_conjunction = f"And({', '.join(premise_strs)})"
                        implication = f"Implies({premises_conjunction}, {conclusion_str})"

                    # 避免重复和恒等式
                    if implication not in processed_expressions and not self._is_tautology(implication):
                        z3_exprs.append(implication)
                        processed_expressions.add(implication)

                except Exception as e:
                    self.logger.debug(f"处理逻辑步骤时出错: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"格式化Z3表达式时出错: {e}")

        return z3_exprs

    def _is_tautology(self, implication: str) -> bool:
        """检测是否为恒等式"""
        try:
            # 简单的恒等式检测
            if "Implies(" in implication and implication.count(",") >= 1:
                # 提取前提和结论部分
                start = implication.find("Implies(") + 8
                parts = implication[start:-1].split(", ", 1)
                if len(parts) == 2:
                    premise_part = parts[0].strip()
                    conclusion_part = parts[1].strip()

                    # 检查是否为 A -> A 或 A∧B∧C -> A 类型的恒等式
                    if premise_part == conclusion_part:
                        return True

                    # 检查 And(A,B,C) -> A 类型的恒等式
                    if premise_part.startswith("And(") and conclusion_part in premise_part:
                        return True

            return False
        except:
            return False

    def _create_distractors(self, logical_steps: List[Dict], var_bindings: Dict[str, str]) -> List[str]:
        """创建增强的干扰项（适配变量数量控制）"""
        try:
            # 创建安全的布尔变量
            var_names = list(var_bindings.keys())
            safe_vars = self.extractor.create_safe_bool_vars(var_names)

            if not safe_vars:
                self.logger.warning("无法创建有效的布尔变量，使用回退干扰项")
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

    def generate_single_sample(self, max_depth: int = 3, sample_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        生成单个数据样本（严格控制变量数量 + 集成可视化）

        :param max_depth: 最大推理深度
        :param sample_id: 样本唯一标识符（用于文件命名）
        :return: 生成的样本数据
        """
        # 如果没有提供sample_id，自动生成
        if sample_id is None:
            sample_id = f"{random.randint(10000, 99999)}"

        for attempt in range(self.max_retry_attempts):
            try:
                # 1. 构建推理DAG
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

                # 3. 提取变量和生成语义绑定（严格控制数量）
                variables = self._extract_variables_from_dag(root)

                # 如果DAG提取失败，尝试从步骤中提取
                if not variables:
                    self.logger.info("尝试从逻辑步骤中提取变量...")
                    variables = self._extract_variables_from_steps(valid_steps)

                if not variables:
                    self.logger.warning("未能提取到变量，重试")
                    continue

                # 再次检查变量数量
                if len(variables) > self.max_variables:
                    self.logger.warning(f"变量数量仍然过多 ({len(variables)} > {self.max_variables})，重试")
                    continue

                if len(variables) < self.min_variables:
                    self.logger.warning(f"变量数量太少 ({len(variables)} < {self.min_variables})，重试")
                    continue

                var_bindings = self._generate_semantic_bindings(variables)
                self.logger.info(f"✅ 生成 {len(var_bindings)} 个变量绑定（在合理范围内）")

                # 4. 构建增强prompt
                enhanced_prompt = self.prompt_builder.build_constrained_prompt(
                    z3_exprs=self._format_z3_expressions(valid_steps, var_bindings),
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
                sample = self._parse_llm_response(response, valid_steps, var_bindings)
                if not sample:
                    self.logger.error("响应解析失败，重试")
                    continue

                # 7. 质量验证
                is_valid, violations = self.validator.validate_sample(sample)

                if is_valid:
                    self.logger.info("✅ 样本通过质量验证")

                    # 8. 添加元数据
                    final_sample = self._add_metadata(sample, valid_steps, var_bindings)

                    # 9. 🎨 生成DAG可视化图片
                    if root is not None:
                        viz_path = self._generate_dag_visualization(
                            root_node=root,
                            sample_id=sample_id,
                            metadata=final_sample.get('metadata', {})
                        )

                        # 将可视化路径添加到样本数据中
                        if viz_path:
                            final_sample['visualization_path'] = viz_path

                    return final_sample
                else:
                    self.logger.warning(f"⚠️ 质量验证失败: {violations}")
                    # 在最后一次尝试时，返回部分合格的样本
                    if attempt == self.max_retry_attempts - 1:
                        sample['validation_warnings'] = violations
                        final_sample = self._add_metadata(sample, valid_steps, var_bindings)

                        # 即使验证失败，也生成可视化
                        if root is not None:
                            viz_path = self._generate_dag_visualization(
                                root_node=root,
                                sample_id=f"{sample_id}_partial",
                                metadata=final_sample.get('metadata', {})
                            )
                            if viz_path:
                                final_sample['visualization_path'] = viz_path

                        return final_sample

            except Exception as e:
                self.logger.error(f"生成样本时出错 (尝试 {attempt + 1}): {e}")
                continue

        self.logger.error("所有重试都失败")
        return None

    def _parse_llm_response(self, response: str, valid_steps: List[Dict], var_bindings: Dict[str, str]) -> \
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

    def _add_metadata(self, sample: Dict, valid_steps: List[Dict], var_bindings: Dict[str, str]) -> Dict:
        """添加增强的元数据"""
        sample['metadata'] = {
            'reasoning_depth': len(valid_steps),
            'variables_count': len(var_bindings),
            'rules_used': [step.get('rule', 'Unknown') for step in valid_steps],
            'semantic_domain': self._infer_semantic_domain(var_bindings),
            'logical_complexity': self._calculate_complexity(valid_steps),
            'generation_version': 'variable_controlled_v2_with_matplotlib_viz',
            'variable_extraction_method': 'enhanced_extractor_with_control',
            'variable_control': {
                'max_variables': self.max_variables,
                'min_variables': self.min_variables,
                'actual_variables': len(var_bindings),
                'within_limits': self.min_variables <= len(var_bindings) <= self.max_variables
            },
            'visualization_enabled': self.enable_visualization
        }

        # 添加质量分数
        sample['quality_score'] = self._calculate_quality_score(sample)

        return sample

    def _infer_semantic_domain(self, var_bindings: Dict[str, str]) -> str:
        """推断语义域"""
        all_bindings = " ".join(var_bindings.values()).lower()

        domain_keywords = {
            "academic": ["coursework", "examination", "thesis", "seminar", "degree", "research"],
            "business": ["project", "budget", "team", "milestone", "client", "contract"],
            "legal": ["evidence", "witness", "case", "hearing", "motion", "settlement"],
            "medical": ["symptoms", "tests", "diagnosis", "treatment", "patient", "recovery"],
            "certification": ["training", "examination", "experience", "application", "certification"]
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

        # 变量使用（现在更重要了）
        var_control = sample.get('metadata', {}).get('variable_control', {})
        if var_control.get('within_limits', False):
            score += 0.2  # 变量数量在合理范围内

        return min(score, 1.0)

    def generate_dataset(self, num_samples: int, output_path: str, max_depth_range: tuple = (5, 8)) -> None:
        """
        生成完整数据集（变量数量控制版 + matplotlib可视化）

        :param num_samples: 生成样本数量
        :param output_path: 数据集输出路径
        :param max_depth_range: 推理深度范围
        """
        self.logger.info(
            f"开始生成 {num_samples} 个样本的数据集（变量数量控制: {self.min_variables}-{self.max_variables}）")
        self.logger.info(f"可视化功能: {'✅ 启用 (matplotlib)' if self.enable_visualization else '❌ 禁用'}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        successful_samples = []
        attempts = 0
        max_attempts = num_samples * 4

        # 统计信息
        stats = {
            "total_attempts": 0,
            "successful": 0,
            "failed": 0,
            "quality_scores": [],
            "semantic_domains": {},
            "complexity_levels": {},
            "variable_control_stats": {
                "avg_variables": 0,
                "within_limits_count": 0,
                "over_limit_count": 0,
                "under_limit_count": 0
            },
            "visualization_stats": {
                "enabled": self.enable_visualization,
                "generated_count": 0,
                "failed_count": 0,
                "output_directory": self.viz_output_dir if self.enable_visualization else None,
                "visualization_engine": "matplotlib + networkx"
            }
        }

        while len(successful_samples) < num_samples and attempts < max_attempts:
            attempts += 1
            stats["total_attempts"] += 1

            # 对于变量数量控制，适当降低深度范围
            depth = random.randint(*max_depth_range)

            self.logger.info(f"生成样本 {len(successful_samples) + 1}/{num_samples} (尝试 {attempts})")

            # 生成唯一的样本ID
            sample_id = f"sample_{len(successful_samples) + 1:04d}_{attempts:04d}"

            sample = self.generate_single_sample(max_depth=depth, sample_id=sample_id)

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

                # 变量控制统计
                var_control = sample.get('metadata', {}).get('variable_control', {})
                actual_vars = var_control.get('actual_variables', 0)
                stats["variable_control_stats"]["avg_variables"] += actual_vars

                if var_control.get('within_limits', False):
                    stats["variable_control_stats"]["within_limits_count"] += 1
                elif actual_vars > self.max_variables:
                    stats["variable_control_stats"]["over_limit_count"] += 1
                else:
                    stats["variable_control_stats"]["under_limit_count"] += 1

                # 可视化统计
                if 'visualization_path' in sample:
                    stats["visualization_stats"]["generated_count"] += 1
                    self.logger.info(f"🎨 可视化文件: {sample['visualization_path']}")
                else:
                    stats["visualization_stats"]["failed_count"] += 1

                self.logger.info(
                    f"✅ 成功生成样本 {len(successful_samples)} (质量分数: {quality_score:.2f}, 变量: {actual_vars})")

            else:
                stats["failed"] += 1
                self.logger.warning(f"❌ 样本生成失败 (尝试 {attempts})")

        # 计算平均变量数量
        if stats["successful"] > 0:
            stats["variable_control_stats"]["avg_variables"] /= stats["successful"]

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

        self.logger.info("=" * 60)
        self.logger.info("📊 数据集生成统计（变量数量控制版 + matplotlib可视化）")
        self.logger.info(f"成功率: {success_rate:.1f}% ({stats['successful']}/{stats['total_attempts']})")
        self.logger.info(f"平均质量分数: {avg_quality:.3f}")
        self.logger.info(f"语义域分布: {stats['semantic_domains']}")
        self.logger.info(f"复杂度分布: {stats['complexity_levels']}")

        # 变量控制统计
        var_stats = stats["variable_control_stats"]
        self.logger.info("🎯 变量控制统计:")
        self.logger.info(f"  平均变量数量: {var_stats['avg_variables']:.1f}")
        self.logger.info(f"  范围内样本: {var_stats['within_limits_count']}")
        self.logger.info(f"  超出上限: {var_stats['over_limit_count']}")
        self.logger.info(f"  低于下限: {var_stats['under_limit_count']}")
        self.logger.info(f"  目标范围: {self.min_variables}-{self.max_variables}")

        # 可视化统计
        viz_stats = stats["visualization_stats"]
        self.logger.info("🎨 可视化统计:")
        self.logger.info(f"  功能状态: {'启用' if viz_stats['enabled'] else '禁用'}")
        self.logger.info(f"  可视化引擎: {viz_stats.get('visualization_engine', 'Unknown')}")
        if viz_stats['enabled']:
            self.logger.info(f"  成功生成: {viz_stats['generated_count']}")
            self.logger.info(f"  生成失败: {viz_stats['failed_count']}")
            self.logger.info(f"  输出目录: {viz_stats['output_directory']}")
            viz_success_rate = (viz_stats['generated_count'] /
                                (viz_stats['generated_count'] + viz_stats['failed_count']) * 100
                                if (viz_stats['generated_count'] + viz_stats['failed_count']) > 0 else 0)
            self.logger.info(f"  可视化成功率: {viz_success_rate:.1f}%")

        self.logger.info("=" * 60)

    def generate_sample_with_custom_visualization(
            self,
            max_depth: int = 3,
            sample_id: Optional[str] = None,
            viz_style: str = "modern",
            viz_format: str = "png"
    ) -> Optional[Dict[str, Any]]:
        """
        生成单个样本并自定义可视化选项

        :param max_depth: 最大推理深度
        :param sample_id: 样本ID
        :param viz_style: 可视化风格 ("modern", "classic", "minimal")
        :param viz_format: 可视化格式 ("png", "pdf", "svg")
        :return: 样本数据
        """
        if sample_id is None:
            sample_id = f"custom_{random.randint(1000, 9999)}"

        # 临时保存原始设置
        original_enable = self.enable_visualization

        # 启用可视化
        self.enable_visualization = True

        try:
            # 生成样本
            sample = self.generate_single_sample(max_depth=max_depth, sample_id=sample_id)

            # 如果样本生成成功且需要自定义可视化
            if sample and (viz_style != "modern" or viz_format != "png"):
                # 这里需要重新生成可视化，但需要保存root_node
                # 注意：当前实现中root_node没有保存到sample中
                self.logger.info(f"🎨 自定义可视化选项: 风格={viz_style}, 格式={viz_format}")
                self.logger.warning("⚠️ 自定义可视化需要在generate_single_sample中保存root_node")

            return sample

        finally:
            # 恢复原始设置
            self.enable_visualization = original_enable

    def create_visualization_gallery(self, samples: List[Dict], output_dir: str = "output/gallery"):
        """
        为多个样本创建可视化画廊

        :param samples: 样本列表
        :param output_dir: 输出目录
        """
        try:
            from dag.visualizer import create_comparison_visualization

            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # 收集有可视化路径的样本
            viz_samples = [s for s in samples if 'visualization_path' in s]

            if not viz_samples:
                self.logger.warning("没有找到包含可视化的样本")
                return

            # 创建摘要页面
            summary_html = self._create_html_summary(viz_samples, output_dir)

            summary_path = os.path.join(output_dir, "visualization_gallery.html")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_html)

            self.logger.info(f"🎨 可视化画廊已创建: {summary_path}")

        except Exception as e:
            self.logger.error(f"创建可视化画廊失败: {e}")

    def _create_html_summary(self, samples: List[Dict], output_dir: str) -> str:
        """创建HTML摘要页面"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head><title>DAG 可视化画廊</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".sample { border: 1px solid #ccc; margin: 20px 0; padding: 15px; }",
            ".viz-image { max-width: 400px; height: auto; }",
            ".metadata { background: #f5f5f5; padding: 10px; margin: 10px 0; }",
            "</style></head><body>",
            f"<h1>DAG 可视化画廊 ({len(samples)} 个样本)</h1>"
        ]

        for i, sample in enumerate(samples, 1):
            metadata = sample.get('metadata', {})
            viz_path = sample.get('visualization_path', '')

            # 计算相对路径
            if viz_path:
                rel_path = os.path.relpath(viz_path, output_dir)
            else:
                rel_path = "无可视化"

            html_parts.extend([
                f"<div class='sample'>",
                f"<h3>样本 {i}</h3>",
                f"<img src='{rel_path}' class='viz-image' alt='DAG可视化' />",
                f"<div class='metadata'>",
                f"<p><strong>推理深度:</strong> {metadata.get('reasoning_depth', 'N/A')}</p>",
                f"<p><strong>变量数量:</strong> {metadata.get('variables_count', 'N/A')}</p>",
                f"<p><strong>语义域:</strong> {metadata.get('semantic_domain', 'N/A')}</p>",
                f"<p><strong>复杂度:</strong> {metadata.get('logical_complexity', 'N/A')}</p>",
                f"<p><strong>质量分数:</strong> {sample.get('quality_score', 'N/A'):.3f}</p>",
                f"</div></div>"
            ])

        html_parts.extend(["</body></html>"])
        return '\n'.join(html_parts)


def main():
    """主函数示例（变量数量控制版 + matplotlib可视化）"""
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

    # 初始化带matplotlib可视化功能的数据集生成器
    generator = DatasetGenerator(
        llm_dispatcher=llm,
        prompt_template_path="prompt/lsat_prompt.txt",
        max_variables=10,  # 最大10个变量
        min_variables=3,  # 最小3个变量
        enable_visualization=True,  # 启用可视化
        viz_output_dir="output/dag_visualizations"  # 可视化输出目录
    )

    # 生成数据集（每个样本都会自动生成对应的DAG图片）
    generator.generate_dataset(
        num_samples=1,  # 先生成少量样本进行测试
        output_path="output/controlled_lsat_dataset_with_matplotlib_viz.jsonl",
        max_depth_range=(8, 10)  # 降低深度范围以便快速测试
    )

    # 示例：生成单个样本并自定义可视化
    print("\n" + "=" * 50)
    print("🎨 生成单个样本并自定义可视化")
    print("=" * 50)

    single_sample = generator.generate_sample_with_custom_visualization(
        max_depth=10,
        sample_id="demo_matplotlib",
        viz_style="modern",
        viz_format="png"
    )

    if single_sample:
        print(f"✅ 单个样本生成成功")
        if 'visualization_path' in single_sample:
            print(f"🎨 可视化路径: {single_sample['visualization_path']}")
        print(f"📊 质量分数: {single_sample.get('quality_score', 0):.3f}")
        print(f"🔢 变量数量: {single_sample.get('metadata', {}).get('variables_count', 0)}")
    else:
        print("❌ 单个样本生成失败")


if __name__ == "__main__":
    main()