from typing import List


class PromptBuilder:
    """
    负责构造prompt文本。
    """
    def __init__(self, template: str):
        self.template = template

    def build(self, z3_exprs: List[str]) -> str:
        """
        根据Z3表达式构造prompt。

        :param z3_exprs: Z3表达式列表
        :return: prompt字符串
        """
        return self.template.replace('{z3_exprs}', '\n'.join(z3_exprs))