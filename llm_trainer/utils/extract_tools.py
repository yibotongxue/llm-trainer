import json
import re
from typing import Any

from .logger import app_logger


def extract_after_last_think_tag(s: str) -> str:
    """
    检查字符串中是否存在</think>标签：
    - 若存在，返回最后一个</think>标签之后的内容
    - 若不存在，返回原字符串
    """
    tag = "</think>"
    last_index = s.rfind(tag)  # 从右向左查找最后一次出现的索引

    if last_index != -1:  # 找到标签
        return s[last_index + len(tag) :]  # 截取标签后的内容
    else:
        return s  # 未找到标签，返回原字符串


def extract_last_tag_content(s: str, tag_name: str) -> str | None:
    """
    提取字符串中最后一个指定标签的内容
    :param s: 输入字符串
    :param tag_name: 标签名（如 "final_answer"）
    :return: 最后一个标签内容（未找到返回 None）
    """
    # 构建正则表达式（考虑标签属性、换行符等）
    pattern = rf"<{tag_name}[^>]*>(.*?)</{tag_name}>"
    matches = re.findall(pattern, s, re.DOTALL)
    return matches[-1] if matches else None


def extract_json_dict(s: str) -> dict[str, Any] | list[dict[str, Any]] | None:
    pattern = r"```(?:[^\n]*)?\n([\s\S]*?)```"
    matches = re.findall(pattern, s)
    if not matches or len(matches) < 1:
        app_logger.error(f"No code block found in the string: {s}")
        return None
    json_block = matches[-1].strip()
    try:
        return json.loads(json_block)  # type: ignore [no-any-return]
    except json.JSONDecodeError:
        app_logger.error(f"Failed to parse JSON from the code block: {json_block}")
        return None
