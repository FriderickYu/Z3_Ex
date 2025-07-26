# 时间表与思路实现

## 思路

![general_chart](images/general_flow.jpg)

## 时间顺序

1. 完成Z3 Rule类的编写, 要求不局限于First-order Logic, 要充分体现出z3的优势
   1. Propositional Logic: axioms
   2. 基本的First-order Logic:
      1. universal instantiation
      2. conjunction introduction
      3. conjunction elimination
      4. disjunction introduction
      5. disjunction elimination
      6. proof by contradiction
   3. Integer and Real Arithmetic
   4. Array Theory
   5. Bit-Vector
   6. Constraints
   7. Function
2. 基于Z3 Rule类测试输入输出
3. 使用GPT4-o, 编写Prompt Engineering代替人工的输入，输出三个对应的文件 => `Natural Language.json`, `Z3-like Expression.json`, `Z3 Program.py`. 这里的Prompt Engineering应体现z3逻辑, 建议参考`LSAT`数据集逻辑
4. 丰富Z3 Rules类的编写，涉及一些复杂逻辑
   1. (选做) 实现Rules之间的嵌套
5. 使用`pseudo vocabulary`, 例如`lorpus, wurpus`代替专有名词、形容词等
6. 添加`distractors`，这里可以考虑在数据集中引入`MCQ`机制而不是仅仅证明`SAT`  or `UNSAT`, 或者引入一些错误的Z3 Rules
7. 未完待续...

* [Z3PY参考手册](https://arabelatso.github.io/2018/06/14/Z3%20API%20in%20Python/)

```text
ARNG_Generator/
├── main.py                         # 主程序入口（可支持 CLI）
├── config.py                       # 全局配置文件（样本数量、启用模块等）

├── registry/                       # 注册器模块（规则、干扰项等插件统一管理）
│   ├── __init__.py
│   ├── rule_registry.py            # 规则注册中心
│   └── distractor_registry.py      # 干扰项注册中心

├── rules/                          # 逻辑规则目录
│   ├── __init__.py
│   ├── base_rule.py                # 所有规则的抽象基类
│   ├── propositional/              # 子目录：命题逻辑规则
│   │   ├── __init__.py
│   │   └── modus_ponens.py         # 示例：MP 推理规则
│   ├── predicate/
│   │   └── __init__.py
│   └── arithmetic/
│       └── __init__.py

├── reasoning/                      # 推理链模块（DAG建模）
│   ├── __init__.py
│   ├── reasoning_step.py           # 推理步骤节点定义
│   └── dag_builder.py              # 推理图构建器

├── distractors/                    # 干扰项模块
│   ├── __init__.py
│   ├── base_distractor.py          # 抽象基类
│   └── negation_distractor.py      # 示例：否定类干扰项

├── nl_generation/                  # 自然语言生成模块（Z3表达式 → NL）
│   ├── __init__.py
│   └── nl_converter.py             # 基于prompt + LLM的表达式转文本

├── prompts/                        # Prompt模板库（支持多样表达结构）
│   ├── __init__.py
│   ├── implication.txt             # 示例：蕴含类模板
│   └── quantifier.txt              # 示例：量词类模板

├── validation/                     # 验证模块
│   ├── __init__.py
│   ├── validator.py                # 样本链条+选项验证
│   └── cache_manager.py            # 去重/缓存/记录管理

├── utils/                          # 通用工具函数
│   ├── __init__.py
│   └── z3_utils.py                 # Z3语法封装（变量创建、公式生成等）

└── data/                           # 样本输出/缓存区
    ├── __init__.py
    └── samples/
        └── train.jsonl             # 可扩展数据集导出

```
