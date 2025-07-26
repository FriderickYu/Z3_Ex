import random
from typing import List

from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from utils.logger_utils import setup_logger

logger = setup_logger("random_semantic_variable_names")


def generate_variable_names(n: int, exclude: set = None) -> List[str]:
    exclude = exclude or set()
    names = []
    while len(names) < n:
        word = random.choice(list(wn.all_lemma_names(pos='n')))
        camel = ''.join(x.capitalize() for x in word.split('_'))
        var = f"{camel}Var"
        if var not in exclude and var not in names:
            names.append(var)
    return names


def build_symbol_table(vars: List[str]) -> dict:
    from z3 import Bool
    return {v: Bool(v) for v in vars}