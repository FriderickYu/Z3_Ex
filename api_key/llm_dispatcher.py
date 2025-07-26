import os
import time
import openai
import logging
from typing import Iterator, AsyncIterator, Union

logger = logging.getLogger("sample_generator")


class LLMDispatcher:
    """
    封装大模型（GPT-4 / DeepSeek）调用接口，支持重试、异步、流式输出等。
    """
    def __init__(
        self,
        model_name: str,
        api_key_path: str,
        retries: int = 3,
        backoff: float = 1.0
    ):
        """
        初始化 LLM 调度器。

        :param model_name: 模型标识符，支持 "gpt4" 或 "deepseek-chat"
        :param api_key_path: 存储 API key 的文件路径
        :param retries: 最大重试次数
        :param backoff: 重试退避时间（秒）
        """
        self.model_name = model_name.lower()
        self.api_key_path = api_key_path
        self.retries = retries
        self.backoff = backoff

        if self.model_name not in ('gpt4', 'deepseek-chat'):
            raise ValueError("model_name must be 'gpt4' or 'deepseek-chat'.")

    def _load_key(self) -> str:
        """从文件加载 API key"""
        if not os.path.exists(self.api_key_path):
            logger.error(f"API key file not found: {self.api_key_path}")
            raise FileNotFoundError(f"{self.api_key_path} not found.")

        key = open(self.api_key_path, 'r', encoding='utf-8').read().strip()
        if not key:
            logger.error("API key file is empty.")
            raise ValueError("API key file is empty.")

        logger.debug("API key loaded successfully.")
        return key

    def _prepare(self) -> str:
        """配置 openai 库基础参数，并返回模型 ID"""
        key = self._load_key()
        if self.model_name == 'deepseek-chat':
            openai.api_base = 'https://api.deepseek.com'
            model_id = 'deepseek-chat'
        else:
            openai.api_base = 'https://api.openai.com/v1'
            model_id = 'gpt-4'
        openai.api_key = key
        logger.debug(f"Prepared model endpoint: {model_id}")
        return model_id

    def call(
        self,
        prompt: str,
        stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        同步调用大模型。

        :param prompt: 传入的用户 Prompt 内容
        :param stream: 是否以流式模式返回
        :return: 模型响应字符串或字符串流
        """
        model_id = self._prepare()
        attempt, wait = 0, self.backoff

        while attempt < self.retries:
            try:
                logger.info(f"Calling LLM ({model_id}), attempt {attempt + 1}")
                resp = openai.ChatCompletion.create(
                    model=model_id,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=1.0,
                    stream=stream,
                    timeout=60
                )
                if stream:
                    def gen():
                        for chunk in resp:
                            delta = chunk.choices[0].delta
                            if 'content' in delta:
                                yield delta['content']
                    return gen()
                return resp.choices[0].message.content.strip()
            except Exception as e:
                attempt += 1
                logger.warning(f"LLM call failed (attempt {attempt}): {e}")
                time.sleep(wait)
                wait *= 2

        logger.error("LLM call failed after maximum retries.")
        return '' if not stream else iter(())

    async def call_async(
        self,
        prompt: str,
        stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """
        异步调用大模型。

        :param prompt: 传入的用户 Prompt 内容
        :param stream: 是否以流式模式返回
        :return: 模型响应字符串或字符串流（异步）
        """
        model_id = self._prepare()
        attempt, wait = 0, self.backoff

        while attempt < self.retries:
            try:
                logger.info(f"Async calling LLM ({model_id}), attempt {attempt + 1}")
                resp = await openai.ChatCompletion.acreate(
                    model=model_id,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=1.0,
                    stream=stream,
                    timeout=60
                )
                if stream:
                    async def agen():
                        async for chunk in resp:
                            delta = chunk.choices[0].delta
                            if 'content' in delta:
                                yield delta['content']
                    return agen()
                return resp.choices[0].message.content.strip()
            except Exception as e:
                attempt += 1
                logger.warning(f"Async LLM call failed (attempt {attempt}): {e}")
                time.sleep(wait)
                wait *= 2

        logger.error("Async LLM call failed after maximum retries.")
        return '' if not stream else (x for x in [])