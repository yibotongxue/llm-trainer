import re


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
