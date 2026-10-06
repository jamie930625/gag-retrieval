"""Golden-example smoke test.

    python3 -m gag_retrieval.selftest

Part 1 — anchor extraction (word-boundary + content rules) on the four target
correspondence shapes and the known noise shapes.
Part 2 — identity ranking: "love you" (title word) must outrank "this moment"
(nobody's identity word) — the v1 inversion bug.
"""

from __future__ import annotations

from .hooks import identity_weight
from .lexical import find_anchors, near_dup_ratio
from .lrc import LineGroup, Song
from .zhnorm import normalize

ANCHOR_CASES = [
    # (a, b, expected_anchor_or_None, label)
    ("我想飛", "想飛上天 和太陽肩並肩", "想飞", "接續/包含（我要飛→我相信）"),
    ("這是我們的快樂", "分手快樂 祝你快樂", "快乐", "頂真/同詞（偷偷→分手快樂）"),
    ("我愛你", "愛你", "爱你", "高頻詞組也成立（我愛你→愛你）"),
    ("大藝術家", "髒藝術家", "藝術家", "近重複（差一字）"),
    ("心偷偷的放晴", "当你也偷偷的看着我", "偷偷", "詞界對齊：偷偷 而非 偷偷的/的看"),
    ("当我在偷偷的看着你", "表现多一点点 让我能 真的看见", None, "「的看」跨詞界不可成立"),
    ("oh除此之外没有什么没讲开", "不论发生什么事我永远爱你", None, "「什么」語法架構詞不可成立"),
    ("我的心裡只有你", "你是我的唯一", None, "純虛詞/代詞重疊"),
    ("把你一生交给我", "你真的愿意就请给我惊喜", None, "「给我」語法詞不可成立"),
    ("今天嫁给我好吗", "就从今天起 I wish", None, "「今天」時間詞不可單獨成梗"),
    ("爱的巴士总是走了又停", "他们总是笑着", None, "「总是」弱語意詞不可成立"),
    ("不愿忘了这一切", "这一刻我终于勇敢说爱你", None, "「这一」跨詞碎片不可成立"),
    ("此刻我多么想要拥抱你", "这一刻我终于勇敢说爱你", None, "「刻我」跨詞碎片不可成立"),
    ("这一刻 我觉得 我们懂得", "这一刻我终于勇敢说爱你", None, "「这一刻(我)」時間詞不可成立"),
    ("Yeah 讓我們搖擺", "Yeah 今晚不回家", None, "感嘆詞 Yeah 不構成梗"),
    ("You're all that I want", "想起 Really", None, "英文跨詞不可誤匹配"),
]


def _mini_song(title: str, lines: list[tuple[str, int]]) -> Song:
    """Build a Song in memory: lines = [(text, repeats)]."""
    s = Song(song_id=title, title=title, title_norm=normalize(title), path="")
    for i, (text, reps) in enumerate(lines):
        s.lines.append(LineGroup(text=text, norm=normalize(text),
                                 times=[float(10 * i)] * reps, first_idx=i))
    return s


def main():
    failures = 0

    print("── Part 1: 錨點抽取 ──")
    for a, b, expect, label in ANCHOR_CASES:
        anchors = find_anchors(normalize(a), normalize(b))
        if expect is None:
            ok = not anchors
        else:
            ok = expect in anchors and not any(
                x in ("的看", "什么", "这一刻我") for x in anchors)
        failures += (not ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {label}\n"
              f"        「{a}」↔「{b}」 anchors={anchors}")

    print("\n── Part 2: 同句反轉 ──")
    nd = near_dup_ratio(normalize("今天你要嫁給我"), normalize("明天我要嫁給你"))
    ok = nd >= 0.5
    failures += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  今天你要嫁給我↔明天我要嫁給你 "
          f"near_dup={nd:.2f} (需 >=0.5)")

    print("\n── Part 3: 錨點身分排序（愛你 必須壓過 这一刻）──")
    shuo = _mini_song("說愛你", [
        ("我的世界变得奇妙", 2),
        ("这一刻我终于勇敢说爱你", 3),
        ("多一点 让我 心甘情愿 爱你", 1),
    ])
    aini = _mini_song("愛你", [
        ("不论发生什么事我永远爱你", 3),
        ("想我就多看一眼", 2),
    ])
    w_aini, l_aini = identity_weight("爱你", shuo, aini)
    w_zyk, l_zyk = identity_weight("这一刻", shuo, aini)
    ok = w_aini > w_zyk and w_aini == 1.0
    failures += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  identity(爱你)={w_aini}({l_aini}) > "
          f"identity(这一刻)={w_zyk}({l_zyk})")

    print("\n── Part 3.5: 主題=散佈句數，非單句重複 ──")
    from .hooks import _song_spread, theme_profile
    feiA = _mini_song("我要飛", [
        ("我要飞 飞越伤悲", 2), ("我敢飞 有梦就追", 2),
        ("飞 飞 飞 飞 我想飞", 4), ("我要用力飞 不管有多远", 4),
        ("这是我们的快乐", 3),
    ])
    sp_fly = _song_spread(feiA, "飞")
    sp_happy = _song_spread(feiA, "快乐")
    prof = [w for w, _, _ in theme_profile(feiA)]
    ok = sp_fly >= 4 and sp_happy == 1 and "飞" in prof and "快乐" not in prof
    failures += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  飞 spread={sp_fly}(主題) vs "
          f"快乐 spread={sp_happy}(單句loop, 非主題)  profile={prof}")

    print("\n── Part 4: 頂真接龍（單字 tail→head）──")
    from .retrieve import _score_anchors
    scored = _score_anchors(normalize("这一刻我终于勇敢说爱你"),
                            normalize("爱就是有我常烦着你"), shuo, aini)
    ok = scored is not None and scored[1] == "爱" and scored[0] >= 0.6
    failures += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  …说爱你 → 爱就是有我常烦着你 "
          f"-> {scored[:3] if scored else None}")
    scored2 = _score_anchors(normalize("安慰她保护着她"),
                             normalize("她们都不懂"), shuo, aini)
    ok2 = scored2 is None or scored2[0] < 0.3       # "she" is a stop char — no chained echo
    failures += (not ok2)
    print(f"  {'PASS' if ok2 else 'FAIL'}  尾字是代詞（她）不觸發接龍 "
          f"-> {scored2[:3] if scored2 else None}")

    total = len(ANCHOR_CASES) + 4
    print(f"\n{total - failures}/{total} passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
