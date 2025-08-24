# PPT演示讲解

我做了一个项目，现在我想就这个项目写PPT用来参加和一个博士生导师的面试（我要读博），你的任务是根据项目的代码，和我提供的项目资料来帮我写一个提纲。虽说是PPT提纲，但是你要提供的内容越详尽越好，因为我要通过你详尽的内容来浓缩写成PPT甚至是介绍报告；不要一口气生成，要一个一个主题生成

你的论述内容应该沿着这个顺序：
1. Introduction: 简单介绍一下这个项目干了什么、为什么要做这个项目？这个项目在现在大模型领域有什么意义呢？z3是什么，为什么要用z3呢？
2. Motivation: 这个项目为什么值得我去做，在学术上有什么意义？解决了问题（之前的类似项目都做了什么、缺点在哪里、共性是什么），我这个项目跟之前的项目有什么不同？
3. Methodology: 这个项目都使用了哪些方法论呢？效果如何？为什么要使用这些方法呢？
4. Implementation: 基于Methodology, 这个项目是怎么实现的？用了哪些工具？效果如何？有什么优点和缺点？
5. Q&A: 这里你需要考虑老师可能会在这个项目问来问去,因为这个idea真的很好。比如, 为什么使用z3py, 为什么使用z3py的验证器, 为什么使用DAG, 为什么使用distractor, 为什么使用tautology_check, 为什么使用双向约束验证器, 为什么使用变量提取器, 为什么使用z3py的求解器, 为什么使用z3py的模型解释器, 为什么使用z3py的模型解释, 变量是如何生成的, 如何和自然语言绑定的, 如何从rule_pool拿到具体的rule并构建成dag，等等，不仅仅局限于这些，你要结合其他论文和这个项目代码回答这些问题，回答时间在30~90s内

我这个项目的描述如下（细节请你根据代码来填充）：

该项目通过Z3PY构建了Rule库，通过DAG来构建成可控制深度和光度的图，并生成一种类z3表达式；通过这种类z3表达式，结合llm+prompt生成带有实际自然语言的多步骤推理数据，该项目创新点如下：
1. 使用了z3py(你需要指出为什么使用z3py，好在哪里), 使用了z3py的验证器 -> validator
2. 通过z3py构建了大约80个z3 rule(还没有实现，不过你先写上并考虑到)
3. 通过DAG构建数据推理逻辑图, 并且实现了可视化
4. 设计了distractor, 同时也设计了tautology_check防止无意义的恒等式产生
5. 双向约束验证器，确保Z3逻辑、变量绑定和自然语言的一致性
6. 变量提取器，专注于控制变量数量和质量
7. gibberish策略
8. 等等，这里需要你通过读取代码和其他论文做对比

QA可能还会问一些问题：[test_ar.json](..%2FAR-LSAT%2Fcomplete_lsat_data%2Ftest_ar.json)
1. 你现在rules有70多个了，你打算如何将其进行组合，有些规则之前只能形式上拼接，但逻辑上是无法拼接的，你打算怎么做
2. 你是具体如何实现的设计了distractor, 还有tautology_check兜底？
3. 为什么要有变量提取器

等等

这个项目主要是参考的论文如下：
1. Critical Thinking for Language Models
2. Enhancing Reasoning Capabilities of LLMs via Principled Synthetic Logic Corpus
3. Learning Deductive Reasoning from Synthetic Corpus based on Formal Logic
4. ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language
5. Transformers as Soft Reasoners over Language
6. ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning

在你论述的时候也请认真参考这6篇文献

