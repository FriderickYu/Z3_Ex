def load_prompt_template(path: str) -> str:
    """
    从文件中加载Prompt模板。
    :param path: 模板文件路径
    :return: 模板字符串
    """
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()