# -*- coding: utf-8 -*-
"""
中文随机句子生成器

用于生成 TTS 测试文本，不依赖外部 AI API。
"""

import random
from typing import List, Tuple

# 主语词库
SUBJECTS = [
    "我", "你", "他", "她", "我们", "大家", "小明", "小红", "老师", "学生",
    "医生", "工程师", "科学家", "艺术家", "音乐家", "作家", "记者", "警察",
    "这位先生", "那位女士", "我的朋友", "他的同事", "她的家人", "公司",
    "团队", "项目组", "研发部门", "市场部", "客户", "用户",
]

# 时间词库
TIME_PHRASES = [
    "今天", "明天", "昨天", "上周", "下周", "这个月", "上个月", "今年",
    "最近", "刚才", "马上", "立刻", "很快", "不久前", "过去",
    "早上", "中午", "下午", "晚上", "凌晨", "周末", "假期期间",
]

# 地点词库
LOCATIONS = [
    "在公司", "在家里", "在学校", "在图书馆", "在咖啡厅", "在会议室",
    "在公园", "在商场", "在医院", "在机场", "在火车站", "在地铁站",
    "在办公室", "在实验室", "在教室", "在操场", "在餐厅", "在酒店",
]

# 动词短语词库
VERB_PHRASES = [
    "正在开发一个新的功能", "完成了一项重要的任务", "参加了一个技术会议",
    "学习了新的编程语言", "解决了一个复杂的问题", "提交了项目报告",
    "设计了一套新的方案", "测试了最新的版本", "优化了系统性能",
    "讨论了未来的发展计划", "制定了下一步的工作安排", "回顾了过去的工作成果",
    "分析了市场数据", "研究了用户需求", "改进了产品体验",
    "组织了团队活动", "培训了新员工", "更新了技术文档",
    "修复了几个重要的问题", "部署了新的服务", "监控了系统运行状态",
    "收集了用户反馈", "整理了项目资料", "准备了演示材料",
]

# 形容词词库
ADJECTIVES = [
    "高效的", "专业的", "创新的", "稳定的", "可靠的", "智能的",
    "先进的", "实用的", "便捷的", "优秀的", "杰出的", "卓越的",
]

# 名词词库
NOUNS = [
    "系统", "平台", "应用", "服务", "方案", "产品", "技术", "工具",
    "项目", "团队", "计划", "目标", "成果", "进展", "效率", "质量",
]

# 连接词
CONNECTORS = [
    "并且", "同时", "而且", "另外", "此外", "因此", "所以", "然后",
]

# 结尾语
ENDINGS = [
    "这是一个很好的开始。",
    "我们对此感到非常满意。",
    "期待能有更好的结果。",
    "这将带来积极的影响。",
    "相信未来会更加美好。",
    "让我们继续努力。",
    "这是值得庆祝的成就。",
    "我们会继续保持这种势头。",
    "这体现了团队的实力。",
    "我们为此感到自豪。",
]


def generate_simple_sentence() -> str:
    """生成简单句"""
    subject = random.choice(SUBJECTS)
    time_phrase = random.choice(TIME_PHRASES) if random.random() > 0.3 else ""
    location = random.choice(LOCATIONS) if random.random() > 0.5 else ""
    verb_phrase = random.choice(VERB_PHRASES)

    parts = [time_phrase, subject, location, verb_phrase]
    parts = [p for p in parts if p]  # 过滤空字符串
    return "".join(parts) + "。"


def generate_compound_sentence() -> str:
    """生成复合句"""
    sentence1 = generate_simple_sentence().rstrip("。")
    connector = random.choice(CONNECTORS)
    sentence2 = generate_simple_sentence().rstrip("。")

    return f"{sentence1}，{connector}{sentence2}。"


def generate_descriptive_sentence() -> str:
    """生成描述性句子"""
    subject = random.choice(SUBJECTS)
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    verb_phrase = random.choice(VERB_PHRASES)

    return f"{subject}开发了一个{adj}{noun}，{verb_phrase}。"


def generate_single_text(length_range: Tuple[int, int] = (50, 100)) -> str:
    """
    生成单个测试文本

    Args:
        length_range: 文本长度范围 (min, max)

    Returns:
        生成的文本
    """
    min_len, max_len = length_range
    target_len = random.randint(min_len, max_len)

    text = ""
    sentence_generators = [
        generate_simple_sentence,
        generate_compound_sentence,
        generate_descriptive_sentence,
    ]

    while len(text) < target_len:
        generator = random.choice(sentence_generators)
        sentence = generator()
        text += sentence

    # 如果超出太多，截断到最近的句号
    if len(text) > max_len + 20:
        # 找到目标长度附近的句号
        end_pos = text.rfind("。", 0, max_len + 10)
        if end_pos > min_len:
            text = text[: end_pos + 1]

    return text


def generate_test_texts(
    count: int = 50,
    length_range: Tuple[int, int] = (50, 100),
) -> List[str]:
    """
    生成测试文本列表

    Args:
        count: 生成数量
        length_range: 文本长度范围

    Returns:
        文本列表
    """
    texts = []
    for _ in range(count):
        text = generate_single_text(length_range)
        texts.append(text)

    return texts


if __name__ == "__main__":
    # 测试文本生成
    texts = generate_test_texts(5, (50, 100))
    for i, text in enumerate(texts, 1):
        print(f"[{i}] ({len(text)}字): {text}")
        print()
