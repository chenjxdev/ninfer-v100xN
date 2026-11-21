import json
from math import comb


def sign_test(wins, losses):
    """Two-sided exact sign test on the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


q = json.load(open("/tmp/bench/qad-constraint.json"))
d = json.load(open("/tmp/bench/daisy-constraint.json"))
print("=== 精确约束类专项（%d 题）===" % q["total"])
print("  QAD       %d/%d (%.0f%%)" % (q["passed"], q["total"], 100.0 * q["passed"] / q["total"]))
print("  DaisyRose %d/%d (%.0f%%)" % (d["passed"], d["total"], 100.0 * d["passed"] / d["total"]))
wins = sum(1 for a, b in zip(q["results"], d["results"]) if a["ok"] and not b["ok"])
loss = sum(1 for a, b in zip(q["results"], d["results"]) if b["ok"] and not a["ok"])
both_fail = sum(1 for a, b in zip(q["results"], d["results"]) if not a["ok"] and not b["ok"])
print("  不一致：QAD 独赢 %d，DaisyRose 独赢 %d；两模型都错 %d" % (wins, loss, both_fail))
print("  双侧符号检验 p = %.4f" % sign_test(wins, loss))

print()
print("=== 词数约束专项（按要求的词数 N 分档）===")
import re
buckets = {}
for a, b in zip(q["results"], d["results"]):
    m = re.search(r"(?:恰好|exactly)\s*(\d+)\s*(?:English\s*)?words?", a["prompt"], re.I)
    if not m:
        continue
    n = int(m.group(1))
    e = buckets.setdefault(n, [0, 0, 0])
    e[0] += 1
    e[1] += int(a["ok"])
    e[2] += int(b["ok"])
for n in sorted(buckets):
    total, qok, dok = buckets[n]
    print("  N=%-3d QAD %d/%d   DaisyRose %d/%d" % (n, qok, total, dok, total))

print()
print("=== 全部题（45 题主集 + 30 题约束专项）===")
try:
    qn = json.load(open("/tmp/bench/qad-nothink.json"))
    dn = json.load(open("/tmp/bench/daisy-nothink.json"))
    print("  主集全题   QAD %d/%d   DaisyRose %d/%d" % (
        qn["passed"], qn["total"], dn["passed"], dn["total"]))
    w2 = sum(1 for a, b in zip(qn["results"], dn["results"]) if a["ok"] and not b["ok"])
    l2 = sum(1 for a, b in zip(qn["results"], dn["results"]) if b["ok"] and not a["ok"])
    print("  主集不一致：QAD 独赢 %d，DaisyRose 独赢 %d  p = %.4f" % (w2, l2, sign_test(w2, l2)))
except FileNotFoundError:
    pass
try:
    qt = json.load(open("/tmp/bench/qad-think.json"))
    dt = json.load(open("/tmp/bench/daisy-think.json"))
    print("  难题(开思考) QAD %d/%d   DaisyRose %d/%d" % (
        qt["passed"], qt["total"], dt["passed"], dt["total"]))
    w3 = sum(1 for a, b in zip(qt["results"], dt["results"]) if a["ok"] and not b["ok"])
    l3 = sum(1 for a, b in zip(qt["results"], dt["results"]) if b["ok"] and not a["ok"])
    print("  难题不一致：QAD 独赢 %d，DaisyRose 独赢 %d  p = %.4f" % (w3, l3, sign_test(w3, l3)))
except FileNotFoundError:
    pass
