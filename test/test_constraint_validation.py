# 文件：test/test_constraint_validation.py
# 说明：测试双向约束验证功能

import sys
from pathlib import Path

# 添加根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import json
from utils.consistency_validator import ConsistencyValidator


def test_constraint_validation():
    """测试约束验证器功能"""
    print("=== 🔍 测试双向约束验证器 ===")

    # 测试不同严格程度
    for strictness in ["lenient", "medium", "strict"]:
        print(f"\n📊 测试 {strictness} 模式:")
        validator = ConsistencyValidator(strictness_level=strictness)

    # 测试样本1：好的样本（优化后）
    good_sample = {
        "context": "在法庭审理中，如果原告提交了证据(Var_0)并且证人出庭作证(Var_1)，那么案件可以进入审理阶段(Var_2)。在这个案件中，原告已经提交了关键证据(Var_0)，证人也同意出庭作证(Var_1)。",
        "question": "基于法庭程序要求，如果原告提交了证据且证人出庭作证，可以得出什么结论？",
        "answers": [
            "A. 案件可以进入审理阶段，因为满足了所有前提条件",
            "B. 案件将败诉，因为证据不足",
            "C. 需要更多证据，证人证词不够",
            "D. 证人证词无效，无法进入审理"
        ],
        "label": "A",
        "z3": [
            "Var_0 = Bool('Var_0')",
            "Var_1 = Bool('Var_1')",
            "Var_2 = Bool('Var_2')",
            "Implies(And(Var_0, Var_1), Var_2)"
        ]
    }

    # 测试样本2：问题样本（保持问题以测试验证器）
    bad_sample = {
        "context": "学生需要提交作业才能通过考试。天气很好。",
        "question": "可以得出什么结论？",
        "answers": ["A. 学生通过考试", "B. 天气影响考试", "C. 作业很重要", "D. 无法确定"],
        "label": "A",
        "z3": [
            "Var_0 = Bool('Var_0')",
            "Var_1 = Bool('Var_1')",
            "Var_2 = Bool('Var_2')",
            "Implies(Var_0, Var_1)"
        ]
    }

    # 测试好样本
    print(f"  📋 高质量样本 ({strictness}):")
    is_valid, violations = validator.validate_sample(good_sample)
    print(f"  验证结果: {'✅ 通过' if is_valid else '❌ 失败'}")
    if violations:
        print(f"  违规情况: {violations[:2]}...")  # 只显示前两个

    # 测试问题样本
    print(f"  📋 问题样本 ({strictness}):")
    is_valid, violations = validator.validate_sample(bad_sample)
    print(f"  验证结果: {'✅ 通过' if is_valid else '❌ 失败'}")
    if violations:
        print(f"  违规数量: {len(violations)}")


    # 详细测试medium模式
    print(f"\n📋 详细测试 medium 模式:")
    validator = ConsistencyValidator(strictness_level="medium")
    is_valid, violations = validator.validate_sample(good_sample)
    print(f"高质量样本结果: {'✅ 通过' if is_valid else '❌ 失败'}")
    if violations:
        print("违规情况:")
        for violation in violations:
            print(f"  - {violation}")

    # 测试修复建议
    if violations:
        suggestions = validator.suggest_fixes(violations)
        print("\n💡 修复建议:")
        for fix_type, suggestion in suggestions.items():
            print(f"  - {fix_type}: {suggestion}")

    print("\n=== ✅ 约束验证测试完成 ===")


def test_enhanced_prompt_builder():
    """测试增强的Prompt构建器"""
    print("\n=== 🛠️ 测试增强Prompt构建器 ===")

    from utils.enhanced_prompt_builder import EnhancedPromptBuilder

    # 创建测试数据
    z3_exprs = [
        "Var_0 = Bool('Var_0')",
        "Var_1 = Bool('Var_1')",
        "Var_2 = Bool('Var_2')",
        "Implies(And(Var_0, Var_1), Var_2)"
    ]

    var_bindings = {
        "Var_0": "evidence_submitted",
        "Var_1": "witness_testified",
        "Var_2": "case_proceeds"
    }

    logical_steps = [
        {
            "rule": "SimpleAndRule",
            "conclusion": "And(Var_0, Var_1)",
            "premises": ["Var_0", "Var_1"]
        },
        {
            "rule": "ImplicationRule",
            "conclusion": "Var_2",
            "premises": ["And(Var_0, Var_1)"]
        }
    ]

    # 测试基础prompt构建
    try:
        # 使用测试用的简化模板
        test_template = """Given the Z3 expressions: {z3_exprs}
Variables: {var_bindings}
Steps: {logical_steps}
Generate LSAT question."""

        # 创建临时模板文件
        test_template_path = "test_template.txt"
        with open(test_template_path, 'w') as f:
            f.write(test_template)

        builder = EnhancedPromptBuilder(test_template_path)

        enhanced_prompt = builder.build_constrained_prompt(
            z3_exprs=z3_exprs,
            var_bindings=var_bindings,
            logical_steps=logical_steps
        )

        print("✅ 增强Prompt构建成功")
        print("📝 Prompt预览:")
        print(enhanced_prompt[:300] + "..." if len(enhanced_prompt) > 300 else enhanced_prompt)

        # 清理临时文件
        Path(test_template_path).unlink()

    except Exception as e:
        print(f"❌ 增强Prompt构建失败: {e}")

    print("\n=== ✅ 增强Prompt测试完成 ===")


def test_integration():
    """集成测试：完整的约束验证流程"""
    print("\n=== 🔄 集成测试：约束验证流程 ===")

    # 模拟完整流程
    try:
        from dag.dag_builder import build_reasoning_dag
        from dag.validator import validate_logical_steps

        # 1. 生成DAG
        print("🏗️ 生成推理DAG...")
        root, logical_steps = build_reasoning_dag(max_depth=2)

        # 2. 验证逻辑
        print("🔍 验证逻辑步骤...")
        valid_steps, failed_steps = validate_logical_steps(logical_steps)
        print(f"有效步骤: {len(valid_steps)}, 失败步骤: {len(failed_steps)}")

        if valid_steps:
            # 3. 模拟变量绑定
            print("🔗 生成变量绑定...")
            import re
            variables = set()
            for step in valid_steps:
                for premise in step.get('premises', []):
                    vars_found = re.findall(r'Var_\d+', premise)
                    variables.update(vars_found)

            var_bindings = {}
            legal_terms = ["evidence_submitted", "witness_testified", "contract_signed", "case_filed"]
            for i, var in enumerate(sorted(variables)):
                if i < len(legal_terms):
                    var_bindings[var] = legal_terms[i]
                else:
                    var_bindings[var] = f"condition_{i}"

            print(f"变量绑定: {var_bindings}")

            # 4. 构建改进的模拟样本用于验证
            mock_sample = {
                "context": f"在法律案件处理中，当{var_bindings.get(list(variables)[0], 'condition1')}并且{var_bindings.get(list(variables)[1] if len(variables) > 1 else list(variables)[0], 'condition2')}时，那么{var_bindings.get(list(variables)[2] if len(variables) > 2 else list(variables)[0], 'result')}。根据案件记录，{var_bindings.get(list(variables)[0], 'condition1')}已经确认，{var_bindings.get(list(variables)[1] if len(variables) > 1 else list(variables)[0], 'condition2')}也已满足。",
                "question": "基于法律程序要求，如果两个前提条件都满足，可以得出什么结论？",
                "answers": [
                    f"A. {var_bindings.get(list(variables)[2] if len(variables) > 2 else list(variables)[0], 'result')}将会发生",
                    "B. 案件将被驳回",
                    "C. 需要更多证据",
                    "D. 无法确定结果"
                ],
                "label": "A",
                "z3": [f"{var} = Bool('{var}')" for var in variables] + [
                    f"Implies(And({list(variables)[0]}, {list(variables)[1] if len(variables) > 1 else list(variables)[0]}), {list(variables)[2] if len(variables) > 2 else list(variables)[0]})"
                ]
            }

            # 5. 约束验证
            print("✅ 执行约束验证...")
            validator = ConsistencyValidator()
            is_valid, violations = validator.validate_sample(mock_sample)

            print(f"集成测试结果: {'✅ 通过' if is_valid else '❌ 需改进'}")
            if violations:
                print("发现的问题:")
                for violation in violations:
                    print(f"  - {violation}")

        else:
            print("⚠️ 没有有效的逻辑步骤可用于测试")

    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== ✅ 集成测试完成 ===")


if __name__ == "__main__":
    print("🧪 双向约束验证完整测试")
    print("=" * 50)

    # 基础功能测试
    test_constraint_validation()

    # Prompt构建器测试
    test_enhanced_prompt_builder()

    # 集成测试
    test_integration()

    print("\n" + "=" * 50)
    print("✅ 所有测试完成")
    print("\n使用说明:")
    print("1. 新的约束验证器会自动检查变量一致性、语义连贯性等")
    print("2. 增强的Prompt构建器会根据历史问题调整生成指导")
    print("3. 在dataset_generator.py中，现在会进行多轮约束验证重试")
    print("4. 可以通过日志查看约束验证的详细结果")