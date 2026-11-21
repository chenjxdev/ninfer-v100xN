import json
from math import comb


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0, 0
    k = min(wins, losses)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p), n


def load(name):
    try:
        return json.load(open("/tmp/bench/%s.json" % name))
    except FileNotFoundError:
        return None


def report(label, q, d, show_diff=True):
    if not q or not d:
        print("(%s 缺数据)" % label)
        return
    print("=== %s ===" % label)
    cq, cd = {}, {}
    for r in q["results"]:
        e = cq.setdefault(r["category"], [0, 0]); e[0] += 1; e[1] += int(r["ok"])
    for r in d["results"]:
        e = cd.setdefault(r["category"], [0, 0]); e[0] += 1; e[1] += int(r["ok"])
    for cat in sorted(set(cq) | set(cd)):
        a, b = cq.get(cat, [0, 0]), cd.get(cat, [0, 0])
        mark = "" if a[1] == b[1] else ("   DaisyRose +%d" % (b[1] - a[1]) if b[1] > a[1]
                                        else "   QAD +%d" % (a[1] - b[1]))
        print("  %-14s QAD %2d/%-3d  DaisyRose %2d/%-3d%s" % (cat, a[1], a[0], b[1], b[0], mark))
    w = sum(1 for a, b in zip(q["results"], d["results"]) if a["ok"] and not b["ok"])
    l = sum(1 for a, b in zip(q["results"], d["results"]) if b["ok"] and not a["ok"])
    both = sum(1 for a, b in zip(q["results"], d["results"]) if not a["ok"] and not b["ok"])
    p, n = sign_test(w, l)
    print("  %-14s QAD %2d/%-3d  DaisyRose %2d/%-3d" % ("TOTAL", q["passed"], q["total"], d["passed"], d["total"]))
    print("  不一致：QAD 独赢 %d，DaisyRose 独赢 %d，两模型都错 %d；双侧符号检验 p = %.4f" % (w, l, both, p))
    print()


report("全题（关思考）", load("qad-nothink"), load("daisy-nothink"))
report("难题（开思考）", load("qad-think"), load("daisy-think"))
report("精确约束专项", load("qad-constraint"), load("daisy-constraint"))

q, d = load("qad-nothink"), load("daisy-nothink")
if q and d:
    print("=== 逐题差异（关思考）===")
    for a, b in zip(q["results"], d["results"]):
        if a["ok"] != b["ok"]:
            print("  [%s] %s" % (a["category"], a["prompt"][:58].replace("\n", " ")))
            print("     QAD   %-4s %s" % ("PASS" if a["ok"] else "FAIL", a["answer"][:88].replace("\n", " ")))
            print("     Daisy %-4s %s" % ("PASS" if b["ok"] else "FAIL", b["answer"][:88].replace("\n", " ")))
    print()
    print("=== 两模型都错的题（区分度为 0 的部分）===")
    cats = {}
    for a, b in zip(q["results"], d["results"]):
        if not a["ok"] and not b["ok"]:
            cats[a["category"]] = cats.get(a["category"], 0) + 1
    print("  ", cats)
