import json
from typing import Optional, Dict
from api_key.llm_dispatcher import LLMDispatcher
from utils.logger_utils import setup_logger

logger = setup_logger("llm_caller")

class LLMParser:
    """
    调用大模型API并解析结果。
    """
    def __init__(self, dispatcher: LLMDispatcher):
        self.dispatcher = dispatcher

    def call_and_parse(self, prompt: str) -> Optional[Dict]:
        """
        调用LLM并解析JSON结果。

        :param prompt: 提示词
        :return: 解析成功返回样本数据, 否则返回None
        """
        resp = self.dispatcher.call(prompt)
        if not resp:
            logger.warning("Empty response from LLM.")
            return None

        try:
            json_str = resp[resp.find('{'):resp.rfind('}') + 1]
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return None