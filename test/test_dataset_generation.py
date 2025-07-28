# 文件：test_dataset_generation.py
# 说明：测试数据集生成流程

import logging
import json
from pathlib import Path
from dataset_generator import DatasetGenerator
from api_key.llm_dispatcher import LLMDispatcher


def test_single_sample():
    """测试单个样本生成"""
    print("=== 🧪 测试单个样本生成 ===")

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 检查API key文件
    api_key_files = [
        "../api_key/ds-api_key.txt"
    ]

    available_key = "../api_key/ds-api_key.txt"
    model_name = "deepseek-chat"

    for key_file in api_key_files:
        if Path(key_file).exists():
            with open(key_file, 'r') as f:
                key_content = f.read().strip()
                if key_content:
                    available_key = key_file
                    model_name = "gpt4" if "openai" in key_file else "deepseek-chat"
                    print(f"✅ 找到可用的API key: {key_file}")
                    break

    if not available_key:
        print("❌ 未找到可用的API key文件，创建模拟响应进行测试")
        test_mock_generation()
        return

    try:
        # 初始化LLM调度器
        llm = LLMDispatcher(
            model_name=model_name,
            api_key_path=available_key,
            retries=2
        )

        # 检查prompt模板
        prompt_path = "../prompt/lsat_prompt.txt"
        if not Path(prompt_path).exists():
            print(f"⚠️  Prompt模板文件不存在: {prompt_path}")
            print("创建默认模板...")
            create_default_prompt_template(prompt_path)

        # 初始化数据集生成器
        generator = DatasetGenerator(
            llm_dispatcher=llm,
            prompt_template_path=prompt_path
        )

        # 生成单个样本
        print("\n🔄 生成单个样本...")
        sample = generator.generate_single_sample(max_depth=3)

        if sample:
            print("✅ 样本生成成功!")
            print("\n📋 样本内容:")
            print(f"Context: {sample.get('context', 'N/A')[:100]}...")
            print(f"Question: {sample.get('question', 'N/A')[:100]}...")
            print(f"答案数量: {len(sample.get('answers', []))}")
            print(f"正确答案: {sample.get('label', 'N/A')}")

            # 保存样本
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            with open(output_dir / "test_sample.json", 'w', encoding='utf-8') as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)

            print(f"\n💾 样本已保存到: {output_dir / 'test_sample.json'}")
        else:
            print("❌ 样本生成失败")

    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


def test_mock_generation():
    """模拟生成过程（不调用实际LLM）"""
    print("\n=== 🎭 模拟生成流程测试 ===")

    from dag.dag_builder import build_reasoning_dag
    from dag.validator import validate_logical_steps
    from distractor.generator import DistractorGenerator
    import z3

    try:
        # 1. 构建推理DAG
        print("🏗️  构建推理DAG...")
        root, logical_steps = build_reasoning_dag(max_depth=3)
        print(f"生成了 {len(logical_steps)} 个推理步骤")

        # 2. 验证步骤
        print("🔍 验证推理步骤...")
        valid_steps, failed_steps = validate_logical_steps(logical_steps)
        print(f"验证通过: {len(valid_steps)} 步, 失败: {len(failed_steps)} 步")

        if valid_steps:
            # 3. 生成变量绑定
            print("🔗 生成变量绑定...")
            variables = []
            for step in valid_steps:
                for premise in step.get('premises', []):
                    import re
                    vars_found = re.findall(r'Var_\d+', premise)
                    variables.extend(vars_found)

            variables = sorted(list(set(variables)))
            print(f"找到变量: {variables[:5]}{'...' if len(variables) > 5 else ''}")

            # 4. 生成干扰项
            print("🎯 生成干扰项...")
            simple_vars = [z3.Bool(var) for var in variables[:5]]  # 限制数量避免过多
            generator = DistractorGenerator(available_vars=simple_vars)
            distractors = generator.generate_all(valid_steps[:2], num_per_strategy=1)
            print(f"生成了 {len(distractors)} 个干扰项")

            # 5. 格式化输出
            print("\n📝 格式化的输入信息:")
            print("Z3表达式示例:")
            for i, step in enumerate(valid_steps[:2], 1):
                print(f"  Step {i}: {step.get('conclusion', 'N/A')}")

            print("\n干扰项示例:")
            for i, d in enumerate(distractors[:3], 1):
                print(f"  Distractor {i}: {d.get('description', 'N/A')}")

            print("\n✅ 模拟生成流程完成 - 所有组件正常工作")
        else:
            print("⚠️  没有有效的推理步骤可用于生成")

    except Exception as e:
        print(f"❌ 模拟生成过程中出错: {e}")
        import traceback
        traceback.print_exc()


def create_default_prompt_template(path: str):
    """创建默认的prompt模板"""
    default_template = """You are a symbolic z3py logic code generator and an LSAT-style question author.

Given the Z3-style logic expressions below, generate a single-choice LSAT-style reasoning question:

Z3 Rules:
{z3_exprs}

Variable Descriptions:
{var_bindings}

Reasoning Chain:
{logical_steps}

Instructions:
1. Create a realistic scenario that illustrates the reasoning chain
2. Provide four answer choices (A, B, C, D) 
3. Return only valid JSON format

Output JSON format:
{{
  "context": "...",
  "question": "...", 
  "answers": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "label": "A",
  "z3": ["var1 = Bool('var1')", "Implies(var1, var2)"]
}}"""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(default_template)


def test_batch_generation():
    """测试批量生成（小规模）"""
    print("\n=== 🚀 测试批量生成 ===")

    # 这里只做模拟，不实际调用LLM
    print("批量生成需要实际的LLM调用，建议:")
    print("1. 确保API key文件存在且有效")
    print("2. 运行 dataset_generator.py 的 main() 函数")
    print("3. 根据需要调整 num_samples 参数")


if __name__ == "__main__":
    print("🔬 数据集生成测试")
    print("=" * 50)

    test_single_sample()
    test_batch_generation()

    print("\n" + "=" * 50)
    print("✅ 测试完成")
    print("\n使用说明:")
    print("1. 确保 api_key/ 目录下有有效的API key文件")
    print("2. 运行 python dataset_generator.py 开始批量生成")
    print("3. 生成的数据集将保存为 JSONL 格式")