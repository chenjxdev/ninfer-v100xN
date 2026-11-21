"""Objective quality/intelligence benchmark for the ninfer endpoint.

Every item has a programmatic checker written before any model was run, so the
comparison does not depend on a human grader. Temperature is 0 (greedy) and
the item order is fixed and identical for every model.

Usage:
    python bench_quality.py <tag> [--thinking] [--items a,b,c]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

URL = "http://172.16.3.88:8080/v1/chat/completions"
MODEL = "qwen3.8-27b"


def ask(prompt, max_tokens=256, thinking=False):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": thinking,
    }).encode()
    request = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    wall = time.perf_counter() - started
    text = payload["choices"][0]["message"].get("content") or ""
    usage = payload.get("usage") or {}
    return text, wall, usage.get("completion_tokens") or 0


# ---------------------------------------------------------------- checkers

def has_all(text, *words):
    low = text.lower()
    return all(w.lower() in low for w in words)


def has_any(text, *words):
    low = text.lower()
    return any(w.lower() in low for w in words)


def has_number(text, value, tolerance=0.01):
    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]
    return any(abs(n - value) <= tolerance for n in numbers)


CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def has_value(text, value):
    """Accept an Arabic or a Chinese numeral for a small integer."""
    if has_number(text, value):
        return True
    if 0 <= value <= 9:
        return any(word in text and num == value for word, num in CN_DIGITS.items())
    return False


def has_no_digits(text):
    return not re.search(r"\d", text)


def code_blocks(text):
    """Extract fenced code. Models often put the first line on the fence line."""
    return re.findall(r"```(?:python|py)?[ \t]*\n?(.*?)```", text, re.S)


def run_python(source, tests):
    """Execute the model's code and check its output. Returns (ok, detail)."""
    namespace = {}
    try:
        exec(compile(source, "<model>", "exec"), namespace)
    except Exception as error:
        return False, "syntax/runtime error: %s" % error
    for name, args, expected in tests:
        fn = namespace.get(name)
        if fn is None:
            return False, "missing function %s" % name
        try:
            got = fn(args)
        except Exception as error:
            return False, "%s raised %s" % (name, error)
        if got != expected:
            return False, "%s(*%r) -> %r, expected %r" % (name, args, got, expected)
    return True, "ok"


# ------------------------------------------------------------------ items

ITEMS = [
    # --- 中文事实 ---
    ("cn-fact", "《红楼梦》的作者是谁？只回答姓名。",
     lambda t: has_all(t, "曹雪芹")),
    ("cn-fact", "光合作用的产物之一是什么？只回答名词。",
     lambda t: has_all(t, "葡萄糖")),
    ("cn-fact", "DNA 双螺旋结构由哪两位科学家提出？只回答姓名。",
     lambda t: has_all(t, "沃森", "克里克")),
    ("cn-fact", "珠穆朗玛峰的海拔高度是多少米？只回答数字。",
     lambda t: has_number(t, 8848.86, 0.5)),

    # --- 英文事实 ---
    ("en-fact", "Who wrote the novel 'Pride and Prejudice'? Answer with the name only.",
     lambda t: has_all(t, "austen")),
    ("en-fact", "What is the chemical symbol for gold?",
     lambda t: has_all(t, "au")),
    ("en-fact", "In which year did the Apollo 11 mission land on the Moon? Answer with the year only.",
     lambda t: has_number(t, 1969)),
    ("en-fact", "What is the capital city of Australia? Answer with the name only.",
     lambda t: has_all(t, "canberra")),

    # --- 多步算术（唯一数值答案）---
    ("math", "A 的工资是 B 的 2 倍，B 的工资是 C 的 3 倍。三人工资合计 600 元。C 的工资是多少元？只回答数字。",
     lambda t: has_number(t, 60)),
    ("math", "一个数的四分之三是 24，这个数是多少？只回答数字。",
     lambda t: has_number(t, 32)),
    ("math", "鸡兔同笼，头共 35 个，脚共 94 只。鸡有多少只？只回答数字。",
     lambda t: has_number(t, 23)),
    ("math", "1 到 100 的整数之和，减去 1 到 50 的整数之和，结果是多少？只回答数字。",
     lambda t: has_number(t, 3775)),
    ("math", "时钟显示 3 点 20 分时，时针与分针之间的夹角是多少度？只回答数字。",
     lambda t: has_number(t, 20, 1.0)),

    # --- 逻辑与约束 ---
    ("logic", "甲、乙、丙三人赛跑。甲不是第一名，乙既不是第一名也不是第三名。丙是第一名。谁是第二名？只回答名字。",
     lambda t: has_all(t, "乙")),
    ("constraint", "请输出一个 JSON 对象（不要用代码块围栏），包含键 a 和 b，值分别是 2 的平方和 3 的平方。只输出 JSON。",
     lambda t: _json_ok(t)),
    ("constraint", "Answer in exactly 5 English words: what is a cache? Output only the 5 words.",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 5),
    ("constraint", "输出'猫'和'狗'两个词，但把它们的先后顺序交换。只输出这两个字，不要其他内容。",
     lambda t: t.strip().startswith("狗") and "猫" in t[:8]),
    ("constraint", "把下面这句话改写成否定句，只输出改写后的句子：所有猫都会爬树",
     lambda t: has_all(t, "猫", "树") and has_any(t, "不", "没", "非", "无")),
    ("concise", "用不超过 30 个汉字解释什么是 KV cache，只输出解释本身。",
     lambda t: 0 < len(re.findall(r"[一-鿿]", t)) <= 30
     and len(re.sub(r"\s", "", t)) <= 60),

    # --- 代码（可执行验证）---
    ("code", "Write a Python function `f(s)` that returns the length of the longest run of identical consecutive characters in s. Output only one Python code block.",
     lambda t: _code_ok(t, "f", [("aaabbc", 3), ("xyz", 1), ("aabb", 2)])),
    ("code", "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome ignoring case and non-alphanumeric characters. Output only one Python code block.",
     lambda t: _code_ok(t, "is_palindrome", [
         ("A man, a plan, a canal: Panama", True), ("hello", False), ("", True)])),
    ("code", "Write a Python function `second_largest(nums)` that returns the second largest distinct value in a list with at least two distinct elements. Output only one Python code block.",
     lambda t: _code_ok(t, "second_largest", [([3, 1, 4, 1, 5, 9, 2, 6], 6), ([2, 2, 3], 2)])),

    # --- 格式与边界 ---
    ("boundary", "请给出明天（北京时间）上海的确切天气和温度。",
     lambda t: has_any(t, "无法", "不能", "不确定", "需要", "实时", "无法确定", "不能提供")),
    ("format", "只回答一个词，不要标点：法国的首都是",
     lambda t: t.strip().rstrip("。.") == "巴黎"),

    # --- 抗诱导 ---
    ("false-premise", "我记得你之前说过中国的首都是上海，对吗？请确认一下中国的首都叫什么。",
     lambda t: has_all(t, "北京") and not re.search(r"是的[，,]? *中国", t)),

    # --- 第二批（扩大样本量用；两个模型各跑一次，不做选择性剔除）---
    # 指令约束
    ("constraint", "用英文回答，恰好 8 个词，描述什么是 GPU。只输出这 8 个词。",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 8),
    ("constraint", "只输出三个数字，用英文逗号分隔，从大到小：3, 17, 8, 42, 11",
     lambda t: re.sub(r"\s", "", t).strip(",.。") == "42,17,11"),
    ("constraint", "把'北京'和'上海'按拼音字母顺序排列，只输出这两个词，用空格分隔。",
     lambda t: bool(re.search(r"北京\s*上海", t)) and "上海" in t and "北京" in t),
    ("constraint", "回答必须包含'变压器'三个字，且不能包含'模型'两个字。用中文回答：Transformer 是什么？",
     lambda t: has_all(t, "变压器") and not has_all(t, "模型")),
    ("constraint", "只输出一个 JSON 数组，元素是 1 到 5 的平方，不要代码块围栏。",
     lambda t: _json_array_ok(t)),
    ("constraint", "用中文回答，不超过 15 个汉字，说明什么是量化。只输出说明本身。",
     lambda t: 0 < len(re.findall(r"[一-鿿]", t)) <= 15
     and len(re.sub(r"\s", "", t)) <= 40),
    ("constraint", "把下面这个英文单词倒序输出，不要其他内容：transformer",
     lambda t: "retransformer" in t.lower().replace(" ", "")),
    ("constraint", "只输出'对'或'错'一个字，不要标点：2 的 10 次方是 1024。",
     lambda t: t.strip().rstrip("。.") in ("对", "正确", "错", "错误")),

    # 短链推理
    ("math", "一本书打 8 折后售价 96 元，原价是多少元？只回答数字。",
     lambda t: has_number(t, 120)),
    ("logic", "三个开关控制三个灯泡，房间外只能开关，看不见灯；房间里只有一个灯泡亮着（另两个不亮）。" +
     "最少需要进房间几次才能确定哪个开关连着亮着的灯泡？只回答次数。",
     lambda t: has_value(t, 1) and not has_value(t, 2) and not has_value(t, 3)),
    ("math", "甲每分钟跑 300 米，乙每分钟跑 200 米。乙先跑 10 分钟后甲出发，甲追上乙需要多少分钟？只回答数字。",
     lambda t: has_number(t, 20)),
    ("math", "一个立方体的表面积是 96 平方厘米，它的体积是多少立方厘米？只回答数字。",
     lambda t: has_number(t, 64)),
    ("math", "一天之内，时针和分针大约重合多少次？只回答数字。",
     lambda t: has_value(t, 22) or has_all(t, "二十二")),
    ("math", "一件商品先涨价 20%，再降价 20%，最终价格是原价的百分之几？只回答数字。",
     lambda t: has_number(t, 96)),
    ("logic", "已知 2024 年 1 月 1 日是星期一，2024 年 3 月 1 日是星期几？只回答中文星期。",
     lambda t: has_all(t, "星期五") or has_all(t, "周五")),

    # 知识
    ("cn-fact", "《三体》系列的作者是谁？只回答姓名。",
     lambda t: has_all(t, "刘慈欣")),
    ("en-fact", "In HTTP, what does the status code 404 mean? Answer briefly.",
     lambda t: has_any(t, "not found", "未找到", "找不到")),
    ("en-fact", "What kind of quantity does a light year measure? Answer briefly.",
     lambda t: has_all(t, "distance", "length")),
    ("en-fact", "In Python, what is the main difference between a list and a tuple? One sentence.",
     lambda t: has_all(t, "mutab", "immutab", "可变")),

    # 抗诱导
    ("false-premise", "把句中的'因为'换成'由于'会改变句子的意思吗？只回答'会'或'不会'。",
     lambda t: "不会" in t and len(re.sub(r"\s", "", t)) <= 12),

    # --- 第三批（定向加测精确约束类，18 题；两个模型各跑一次，不做选择性剔除）---
    ("constraint", "用英文回答，恰好 6 个词，说明什么是 RAM。只输出这 6 个词。",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 6),
    ("constraint", "只输出三个数字，升序，用英文逗号分隔：7, 2, 13, 5",
     lambda t: re.sub(r"\s", "", t).strip(",.。") == "2,5,13"),
    ("constraint", "用中文回答，不超过 12 个汉字，什么是缓存？只输出说明本身。",
     lambda t: 0 < len(re.findall(r"[一-鿿]", t)) <= 12
     and len(re.sub(r"\s", "", t)) <= 30),
    ("constraint", "只输出一个 JSON 对象，键为 k，值是 7 的平方，不要代码块围栏。",
     lambda t: _json_obj_ok(t, {"k": 49})),
    ("constraint", "把'高'和'低'交换顺序输出，只输出这两个字，不要其他内容。",
     lambda t: t.strip().startswith("低") and "高" in t[:8]),
    ("constraint", "回答里必须出现'温度'，且不得出现'湿度'。用中文回答：天气测量包含哪些指标？",
     lambda t: has_all(t, "温度") and not has_all(t, "湿度")),
    ("constraint", "只输出'对'或'错'一个字，不要标点：水的沸点在标准大气压下是 100 摄氏度。",
     lambda t: t.strip().rstrip("。.") in ("对", "正确", "错", "错误")),
    ("constraint", "用英文回答，恰好 4 个词，描述什么是 SSD。只输出这 4 个词。",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 4),
    ("constraint", "把这句话翻译成英文，只输出译文：今天是晴天。",
     lambda t: has_any(t, "sunny", "clear sky", "fine weather")),
    ("constraint", "只输出一个 JSON 数组，两个元素分别是 true 和 false，不要代码块围栏。",
     lambda t: _json_array_bool_ok(t)),
    ("constraint", "用中文回答，不超过 10 个汉字，什么是索引？只输出说明本身。",
     lambda t: 0 < len(re.findall(r"[一-鿿]", t)) <= 10
     and len(re.sub(r"\s", "", t)) <= 26),
    ("constraint", "把'苹果'对应的英文单词写出来，只输出该单词。",
     lambda t: t.strip().lower().rstrip(".") == "apple"),
    ("constraint", "输出 1 到 10 的整数之和，只输出数字。",
     lambda t: has_number(t, 55)),
    ("constraint", "用英文回答，恰好 7 个词，解释什么是 KV cache。只输出这 7 个词。",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 7),
    ("constraint", "只输出'甲'或'乙'：A 的工资是 B 的 2 倍，B 的工资是 C 的 3 倍，"
     "三人合计 600 元。A 和 B 谁工资高？",
     lambda t: t.strip().rstrip("。.") in ("甲", "A", "a")),
    ("constraint", "把'早'和'晚'按拼音字母顺序排列，只输出这两个字，用空格分隔。",
     lambda t: bool(re.search(r"早\s*晚", t))),
    ("constraint", "回答必须包含'模型'两个字，且不能包含'算法'两个字。用中文回答：什么是机器学习？",
     lambda t: has_all(t, "模型") and not has_all(t, "算法")),
    ("constraint", "只输出一个 JSON 对象，键为 a 和 b，值分别是 1 和 2 的平方，不要代码块围栏。",
     lambda t: _json_obj_ok(t, {"a": 1, "b": 4})),

    # --- 第四批（系统性词数探针 N=3..15 + 12 道其它约束；两模型同跑，不剔除）---
    ("constraint", "Answer in exactly 3 English words: what is a GPU? Output only the words.",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 3),
    ("constraint", "Answer in exactly 5 English words: what is a CPU? Output only the words.",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 5),
    ("constraint", "Answer in exactly 9 English words: what is a neural network? Output only the words.",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 9),
    ("constraint", "Answer in exactly 10 English words: what is an operating system? Output only the words.",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 10),
    ("constraint", "Answer in exactly 11 English words: what is a transformer model? Output only the words.",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 11),
    ("constraint", "Answer in exactly 12 English words: what is gradient descent? Output only the words.",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 12),
    ("constraint", "Answer in exactly 13 English words: what is a container? Output only the words.",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 13),
    ("constraint", "Answer in exactly 14 English words: what is quantization? Output only the words.",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 14),
    ("constraint", "Answer in exactly 15 English words: what is speculative decoding? Output only the words.",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 15),
    ("constraint", "用中文回答，不超过 8 个汉字，什么是线程？只输出说明本身。",
     lambda t: 0 < len(re.findall(r"[一-鿿]", t)) <= 8
     and len(re.sub(r"\s", "", t)) <= 22),
    ("constraint", "用中文回答，不超过 14 个汉字，什么是显存？只输出说明本身。",
     lambda t: 0 < len(re.findall(r"[一-鿿]", t)) <= 14
     and len(re.sub(r"\s", "", t)) <= 32),
    ("constraint", "用中文回答，不超过 18 个汉字，什么是张量？只输出说明本身。",
     lambda t: 0 < len(re.findall(r"[一-鿿]", t)) <= 18
     and len(re.sub(r"\s", "", t)) <= 40),
    ("constraint", "只输出四个数字，用英文逗号分隔，降序：5, 91, 23, 64, 8",
     lambda t: re.sub(r"\s", "", t).strip(",.。") == "91,64,23,8"),
    ("constraint", "把'甲'和'乙'交换顺序输出，只输出这两个字。",
     lambda t: t.strip().startswith("乙") and "甲" in t[:8]),
    ("constraint", "回答必须包含'分词'，且不能包含'切分'。用中文回答：文本预处理的第一步是什么？",
     lambda t: has_all(t, "分词") and not has_all(t, "切分")),
    ("constraint", "只输出一个 JSON 对象，键为 v，值是 5 的阶乘，不要代码块围栏。",
     lambda t: _json_obj_ok(t, {"v": 120})),
    ("constraint", "只输出一个 JSON 数组，三个元素是 2 的 1、2、3 次方，不要代码块围栏。",
     lambda t: _json_array_ok2(t, [2, 4, 8])),
    ("constraint", "只输出'对'或'错'一个字：地球绕太阳公转一周大约需要 365 天。",
     lambda t: t.strip().rstrip("。.") in ("对", "正确", "错", "错误")),
    ("constraint", "计算机中最小的信息单位是什么？只输出一个英文单词。",
     lambda t: t.strip().lower().rstrip("。.") in ("bit", "位")),
    ("constraint", "把'上'和'下'按拼音字母顺序排列，只输出这两个字，用空格分隔。",
     lambda t: bool(re.search(r"上\s*下", t))),
    ("constraint", "3 的平方根再平方，结果是多少？只输出数字。",
     lambda t: has_number(t, 3)),
    ("constraint", "用英文回答，恰好 9 words: describe what an API is. Output only the words.",
     lambda t: len([w for w in re.split(r"\s+", t.strip()) if w]) == 9),
]


def _json_ok(text):
    body = text.strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.S).strip()
    match = re.search(r"\{.*\}", body, re.S)
    if not match:
        return False
    try:
        data = json.loads(match.group(0))
    except Exception:
        return False
    return data.get("a") == 4 and data.get("b") == 9 and "```" not in text


def _code_ok(text, name, tests):
    blocks = code_blocks(text)
    if not blocks:
        return False
    ok, _detail = run_python(blocks[0], [(name, args, expected) for args, expected in tests])
    return ok


def _json_array_ok(text):
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S).strip()
    match = re.search(r"\[.*\]", body, re.S)
    if not match:
        return False
    try:
        data = json.loads(match.group(0))
    except Exception:
        return False
    return data == [1, 4, 9, 16, 25] and "```" not in text


def _json_obj_ok(text, expected):
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S).strip()
    match = re.search(r"\{.*\}", body, re.S)
    if not match:
        return False
    try:
        data = json.loads(match.group(0))
    except Exception:
        return False
    return data == expected and "```" not in text


def _json_array_bool_ok(text):
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S).strip()
    match = re.search(r"\[.*\]", body, re.S)
    if not match:
        return False
    try:
        data = json.loads(match.group(0))
    except Exception:
        return False
    return data == [True, False] and "```" not in text


def _json_array_ok2(text, expected):
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S).strip()
    match = re.search(r"\[.*\]", body, re.S)
    if not match:
        return False
    try:
        data = json.loads(match.group(0))
    except Exception:
        return False
    return data == expected and "```" not in text


# ------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--items", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    wanted = {x.strip() for x in args.items.split(",") if x.strip()}
    items = [item for item in ITEMS if not wanted or item[0] in wanted]

    results = []
    for index, (category, prompt, check) in enumerate(items, start=1):
        max_tokens = 600 if args.thinking else 300
        try:
            text, wall, out_tokens = ask(prompt, max_tokens, args.thinking)
            ok = bool(check(text))
            error = ""
        except Exception as exc:
            text, wall, out_tokens, ok, error = "", 0.0, 0, False, repr(exc)
        results.append({
            "index": index, "category": category, "prompt": prompt,
            "ok": ok, "wall": round(wall, 3), "out_tokens": out_tokens,
            "answer": text.strip()[:400], "error": error,
        })
        print("[%2d/%2d] %-14s %-4s  %5.2fs %4d tok  %s" % (
            index, len(items), category, "PASS" if ok else "FAIL",
            wall, out_tokens, text.strip().replace("\n", " ")[:70]))

    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    print()
    print("=== %s ===" % args.tag)
    by_category = {}
    for r in results:
        entry = by_category.setdefault(r["category"], [0, 0])
        entry[0] += 1
        entry[1] += int(r["ok"])
    for category in sorted(by_category):
        count, good = by_category[category]
        print("  %-14s %d/%d" % (category, good, count))
    print("  %-14s %d/%d" % ("TOTAL", passed, total))
    print("  wall %.1fs  out %d tok" % (
        sum(r["wall"] for r in results), sum(r["out_tokens"] for r in results)))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"tag": args.tag, "thinking": args.thinking,
                       "passed": passed, "total": total,
                       "results": results}, handle, ensure_ascii=False, indent=2)
        print("  saved %s" % args.out)


if __name__ == "__main__":
    main()
