# 量词操作

"""
* ExistentialGeneralization（存在泛化）：`P(a) ⊢ ∃x P(x)`
* UniversalInstantiationRule（全称实例化）：`∀x P(x) ⊢ P(a)` **已存在**
* UniversalGeneralization（全称泛化，侧条件）：`P(x) ⊢ ∀x P(x) （侧条件：x 不自由出现在未解除前提中）`
* ExistentialElimination（存在消去，证明规则）： `∃x P(x) ⇒ 取 fresh 常量 c，在子证明中以假设 P(c) 推出目标 φ，且 c 不出现在 φ/任何开放前提 中，则得 ⊢ φ`
"""