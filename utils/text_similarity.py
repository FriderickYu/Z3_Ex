"""
utils/text_similarity.py
-------------------------

提供文本相似度计算和过滤的工具类，基于词频向量的余弦相似度。

使用示例：
    from utils.text_similarity import TextSimilarity
    sim_filter = TextSimilarity(threshold=0.8)
    if sim_filter.is_unique("text1"):
        sim_filter.add("text1")
    if sim_filter.is_unique("text2"):
        sim_filter.add("text2")

该类适用于在生成数据时控制文本之间的相似度，避免重复。
"""

import re
from collections import Counter
from math import sqrt
from typing import List


class TextSimilarity:
    """基于余弦相似度的文本相似度过滤器。"""

    def __init__(self, threshold: float) -> None:
        """
        :param threshold: 相似度阈值，取值区间 [0.0, 1.0]。当两段文本相似度大于等于该阈值时视为重复。
        """
        self.threshold = max(0.0, min(threshold, 1.0))
        self.contexts: List[str] = []

    @staticmethod
    def _preprocess(text: str) -> List[str]:
        """
        对文本进行预处理：小写化、提取字母数字单词列表。
        """
        return re.findall(r"[A-Za-z0-9]+", text.lower())

    @staticmethod
    def _vectorize(words: List[str]) -> Counter:
        """构建词频向量。"""
        return Counter(words)

    @staticmethod
    def _cosine_similarity(vec_a: Counter, vec_b: Counter) -> float:
        """
        计算两个词频向量的余弦相似度。
        """
        if not vec_a or not vec_b:
            return 0.0
        # 交集部分的点积
        dot = sum(vec_a[w] * vec_b[w] for w in (set(vec_a.keys()) & set(vec_b.keys())))
        norm_a = sqrt(sum(count * count for count in vec_a.values()))
        norm_b = sqrt(sum(count * count for count in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """公开方法：计算两段文本的相似度。"""
        words_a = self._preprocess(text_a)
        words_b = self._preprocess(text_b)
        vec_a = self._vectorize(words_a)
        vec_b = self._vectorize(words_b)
        return self._cosine_similarity(vec_a, vec_b)

    def is_unique(self, text: str) -> bool:
        """
        判断给定文本与已记录文本相比是否低于相似度阈值。
        返回 True 表示可以接受，False 表示重复过高。
        """
        for ctx in self.contexts:
            if self.compute_similarity(text, ctx) >= self.threshold:
                return False
        return True

    def add(self, text: str) -> None:
        """保存已接受的文本，供后续相似度比较。"""
        self.contexts.append(text)