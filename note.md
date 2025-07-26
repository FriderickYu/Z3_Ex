构造的问题必须强制依赖多条逻辑规则嵌套组合，答案必须要沿着结构 DAG 路径完成演绎链才能得到。

给你一张图，这张图来自于FLDx2的那篇论文，你可以参照他这里的multi-step deductive reasoning的思路设计

我的想法是，
Generating deductive reasoning samples
* A/Several given hypothesis -> prove/disprove
  * Combining given facts
  * Following rigid reasoning rules

为了避免一种情况，即拿到的rules很多，但是answer没有通过rules组合嵌套生成，仅仅是随机挑了某个简单的rule推断出。
除了使用FLDX2相似的结构，我的想法还有一些
* 如果你使用逻辑树或者DAG结构的话，我希望树/图的深度可以通过参数进行指定

你觉得这种设计如何，如果按照这种设计的话, answer的生成需要怎么设计呢？