from typing import List, Dict


def build_var_binding_string(var_descriptions: List[Dict[str, str]]) -> str:
    """
    将变量描述信息列表转换为格式化的字符串，用于嵌入 prompt 中。

    :param var_descriptions: 每个 dict 代表某个 rule 的变量-描述映射
    :return: 以 - var: description 形式拼接的字符串
    """
    lines = []
    for desc_dict in var_descriptions:
        for var, description in desc_dict.items():
            lines.append(f"- {var}: {description}")
    return '\n'.join(lines)
