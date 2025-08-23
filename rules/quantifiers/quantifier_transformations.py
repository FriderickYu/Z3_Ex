# 量词变换与分配

"""
* QuantifierNegation-1：`¬∀x P(x) ⟷ ∃x ¬P(x)`
* QuantifierNegation-2：`¬∃x P(x) ⟷ ∀x ¬P(x)`
* UniversalDistribution（对∧的分配）：`∀x (P(x)∧Q(x)) ⟷ (∀x P(x)) ∧ (∀x Q(x))`
* ExistentialDistribution（对∨的分配）：`∃x (P(x)∨Q(x)) ⟷ (∃x P(x)) ∨ (∃x Q(x))`
* ExistentialConjunction（一向）：`∃x (P(x)∧Q(x)) ⊢ (∃x P(x)) ∧ (∃x Q(x))`
"""