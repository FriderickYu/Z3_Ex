import os
import time
import logging
from typing import Iterator, AsyncIterator, Union

logger = logging.getLogger("sample_generator")


class LLMDispatcher:
    """
    封装大模型（GPT-4 / DeepSeek）调用接口，支持重试、异步、流式输出等。
    兼容 OpenAI >= 1.0.0 版本
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

    def _get_client(self):
        """获取 OpenAI 客户端实例"""
        try:
            # 尝试使用新版 OpenAI API (>= 1.0.0)
            from openai import OpenAI

            key = self._load_key()

            if self.model_name == 'deepseek-chat':
                client = OpenAI(
                    api_key=key,
                    base_url='https://api.deepseek.com'
                )
                model_id = 'deepseek-chat'
            else:
                client = OpenAI(api_key=key)
                model_id = 'gpt-4'

            logger.debug(f"Using OpenAI >= 1.0.0 API with model: {model_id}")
            return client, model_id, "new_api"

        except ImportError:
            # 回退到旧版 API
            logger.debug("OpenAI >= 1.0.0 not available, falling back to legacy API")
            return self._prepare_legacy(), "legacy_api"

    def _prepare_legacy(self):
        """配置旧版 openai 库基础参数，并返回模型 ID"""
        try:
            import openai

            key = self._load_key()
            if self.model_name == 'deepseek-chat':
                openai.api_base = 'https://api.deepseek.com'
                model_id = 'deepseek-chat'
            else:
                openai.api_base = 'https://api.openai.com/v1'
                model_id = 'gpt-4'
            openai.api_key = key
            logger.debug(f"Prepared legacy API with model: {model_id}")
            return model_id

        except ImportError:
            raise ImportError("OpenAI library not found. Please install: pip install openai")

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
        attempt, wait = 0, self.backoff

        while attempt < self.retries:
            try:
                logger.info(f"Calling LLM ({self.model_name}), attempt {attempt + 1}")

                # 获取客户端
                client_info = self._get_client()

                if len(client_info) == 3:  # 新版 API
                    client, model_id, api_type = client_info
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=1.0,
                        stream=stream,
                        timeout=60
                    )

                    if stream:
                        def gen():
                            for chunk in response:
                                if chunk.choices[0].delta.content:
                                    yield chunk.choices[0].delta.content

                        return gen()
                    else:
                        return response.choices[0].message.content.strip()

                else:  # 旧版 API
                    import openai
                    model_id = client_info

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
                if attempt < self.retries:
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
        attempt, wait = 0, self.backoff

        while attempt < self.retries:
            try:
                logger.info(f"Async calling LLM ({self.model_name}), attempt {attempt + 1}")

                # 获取客户端
                client_info = self._get_client()

                if len(client_info) == 3:  # 新版 API
                    client, model_id, api_type = client_info
                    response = await client.chat.completions.acreate(
                        model=model_id,
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=1.0,
                        stream=stream,
                        timeout=60
                    )

                    if stream:
                        async def agen():
                            async for chunk in response:
                                if chunk.choices[0].delta.content:
                                    yield chunk.choices[0].delta.content

                        return agen()
                    else:
                        return response.choices[0].message.content.strip()

                else:  # 旧版 API 异步
                    import openai
                    model_id = client_info

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
                if attempt < self.retries:
                    time.sleep(wait)
                    wait *= 2

        logger.error("Async LLM call failed after maximum retries.")
        return '' if not stream else (x for x in [])