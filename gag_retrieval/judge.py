"""Stage 2 — the LLM judge.

Input: song A's full lyrics + the stage-1 top-K candidate songs (full lyrics
AND the candidate line-pair evidence). The LLM does NOT free-associate humor:
it verifies/ranks already-found lexical correspondences and picks the exact
entry line — a discriminator with a rubric, not a comedian.

Backends (all stdlib urllib, no SDK):
    prompt  — write the prompt to a file for manual pasting (default; free)
    gemini  — GEMINI_API_KEY [+ GEMINI_MODEL, default gemini-2.5-flash]
    openai  — any OpenAI-compatible /chat/completions:
              OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
    vllm    — the project's local server: OMNI_VLLM_URL [+ OMNI_VLLM_MODEL]
"""

from __future__ import annotations

import json
import os
import urllib.request

from .hooks import theme_profile
from .lrc import Song
from .retrieve import SongResult


def _fmt_time(t: float) -> str:
    return f"{int(t) // 60:02d}:{t % 60:05.2f}"


def _lyrics_block(song: Song) -> str:
    return "\n".join(f"[{_fmt_time(lg.times[0])}] {lg.text}"
                     + (f"  (唱{lg.repeats}次)" if lg.repeats > 1 else "")
                     for lg in song.lines)


def _theme_block(song: Song, label: str) -> str:
    prof = theme_profile(song)
    if not prof:
        return f"{label}主題語意：（無明顯重複主題）"
    parts = [f"「{w}」×{s}句/{t}次" for w, s, t in prof]
    return f"{label}主題語意（被反覆轟炸、留在觀眾耳中的詞）：" + "、".join(parts)


def build_prompt(query: Song, results: list[SongResult],
                 songs: dict[str, Song]) -> str:
    parts = [
        "你是一位華語 DJ 的接歌顧問。現在正在播放歌曲 A，你要從候選歌單中選出"
        "「歌詞字面上對應」最強的下一首歌，以及要從哪一句接進去。",
        "",
        "「歌詞字面上對應」的定義（必須直觀、現場一聽就懂）：",
        "- 接續：A 的詞組被 B 的句子直接接住/包含（例：「我想飛」→「想飛上天 和太陽肩並肩」）",
        "- 頂真/同詞呼應：A 句的關鍵詞組出現在 B 句（例：「這是我們的快樂」→「分手快樂 祝你快樂」）",
        "- 同句反轉：幾乎同一句話，人稱或時間對調（例：「今天你要嫁給我」→「明天我要嫁給你」）",
        "- 常見詞組也算數：A 在唱「我愛你」，B 接「愛你」完全成立。",
        "",
        "最重要的一條：共同詞組必須是「重點詞」——歌名詞、副歌 hook 詞、或那句的"
        "情感核心（如「愛你」「快樂」「想飛」「嫁給我」）。時間/疑問/語法架構詞"
        "（如「这一刻」「什么」「一起」「的看」）即使字面對上也【不算梗】——觀眾"
        "聽到不會有任何反應。寧可選短而有力的重點詞，不要選長而空洞的架構詞。",
        "",
        "評選規則：",
        "1. 對應必須是字面上的（共用重點詞組/包含/近乎同句）。單純主題相似（都是情歌）不算。",
        "2. B 的接入句越有辨識度越好：歌名句 > 副歌句 > 首句 > 一般句 —— 觀眾要一聽就認出 B 是哪首歌。",
        "3. 優先讓 A 的來源句也是大家熟的句子（副歌尤佳），梗的兩端都聽得懂才會笑。",
        "4. 候選順序和線索【只代表召回，不代表好壞】。請你自己讀完整歌詞重新判斷："
        "可以採用線索、可以改選同首歌更好的句對（包括線索完全沒列到的句子）、"
        "也可以判定整個候選不成立。",
        "5. 「直觀」的判準：觀眾聽到 B 句的當下，必須不假思索立刻連回 A 剛唱的那個詞。"
        "這要求錨點詞是【兩句各自的語意重心】，不是順帶出現的詞：",
        "   ✓ A 婚禮宣誓「你願意這樣做嗎 Yes I do」→ B「你真的願意就請給我驚喜」"
        "（「願意」是兩句的核心，像一問一答）",
        "   ✗ A「微風吹來浪漫的氣息」→ B「氣氛好浪漫」（「浪漫」在兩句都只是"
        "形容詞路過，兩句各說各的 —— 這是牽強，不是梗）",
        "6. 梗成立的前提——呼應的必須是 A 被轟炸過的【語意】，不是碰巧相同的"
        "【字】：現場觀眾不比對歌詞，他們耳中只留著 A 的主題語意（見下方"
        "「A 主題語意」清單——散佈在越多不同句子裡的詞，越是整首歌的意思本身）。"
        "「我要飛→飛太遠」神接，因為 A 把「飛」織進九個不同句子，觀眾整首歌都"
        "泡在『飛』的語意裡，B 用自己的主題「飛太遠」接住——這是【主題對主題】。"
        "反例：「快樂」只出現在 A 的一句歌詞（就算那句重複三次），它不是 A 的"
        "語意，拿它接歌沒有人會反應——【同字不同意＝假梗】。",
        "   語意場的呼應【不必逐字相同】：A 轟炸「飛」，B 用「翅膀」「衝上雲端」"
        "「天空沒有極限」回應也完全成立——判斷同義/意象延伸正是你的工作，"
        "字面檢索做不到這件事。",
        "7. 梗的時效物理（非常重要）：呼應必須發生在 B 進歌後的【第一秒】。"
        "優先順序：",
        "   (a) 頂真/接龍：A 句尾的詞被 B 句的【開頭】立刻接住"
        "（「我想飛」→「想飛上天…」；「…说爱你」→「爱就是有我常烦着你」）— 最理想；",
        "   (b) 歌名梗：A 句尾的詞＝B 的歌名 → enter_line 選 B【最招牌的句子】"
        "（副歌第一句或全曲第一句），梗靠「觀眾瞬間認出這首歌」成立，"
        "進歌句本身不必含錨點詞（「我愛你」→ 直接 drop《愛你》的招牌副歌）；",
        "   (c) 錨點埋在 B 句【中後段】的呼應會遲到好幾個字，效果大打折扣 — "
        "同樣的錨點，永遠優先選「B 句首含錨點」的句子，其次選招牌句進歌。",
        "8. confidence 嚴格校準：「高」保留給歌名/招牌 hook 級的完美咬合"
        "（我愛你→愛你、我們的快樂→分手快樂 這種等級）；「中」＝聽得懂、會心一笑，"
        "但錨點非招牌句或咬合不完全；「低」＝只是共用詞。小曲庫裡沒有神級接點是"
        "正常的，多數情況應該誠實地標中或低，寧可保守不要吹捧。why 欄位不要用"
        "「絕對瘋掉」「效果極佳」這類推銷語言，冷靜描述梗的機制即可。",
        "",
        "=== 歌曲 A（正在播放）===",
        f"《{query.title}》",
        _theme_block(query, "A "),
        _lyrics_block(query),
        "",
        "=== 候選歌單（順序無意義，僅為召回結果）===",
    ]
    for rank, r in enumerate(results, 1):
        song_b = songs[r.song_id]
        parts.append(f"\n--- 候選：《{song_b.title}》 ---")
        parts.append(_theme_block(song_b, "B "))
        parts.append("字面接點線索（未按好壞排序，僅供參考）：")
        seen = set()
        for ev in r.evidence:
            span = "、".join(f"「{s}」" for s in ev.anchors) if ev.anchors \
                else "(語意對應)"
            sat = (f"【「{ev.anchor}」散佈在 A 的 {ev.a_spread} 個句子/"
                   f"唱 {ev.a_sat} 次】"
                   if getattr(ev, "a_spread", 1) >= 2 else "")
            row = (f"  A[{_fmt_time(ev.a_time)}]「{ev.a_text}」 ↔ "
                   f"B[{_fmt_time(ev.b_time)}]「{ev.b_text}」"
                   f"({ev.b_label}) 共同詞組 {span}{sat}")
            if row not in seen:
                seen.add(row)
                parts.append(row)
        parts.append("完整歌詞：")
        parts.append(_lyrics_block(song_b))
    parts += [
        "",
        "請只輸出一個 JSON 物件（不要其他文字）：",
        '{',
        '  "next_song": "<歌名>",',
        '  "anchor": "<這個梗的重點詞組，如「愛你」>",',
        '  "from_line": {"text": "<A 的來源句，逐字照抄>", "time": "<mm:ss.ss>"},',
        '  "enter_line": {"text": "<B 的接入句，逐字照抄>", "time": "<mm:ss.ss>"},',
        '  "type": "<接續|頂真|同句反轉|同詞呼應>",',
        '  "why": "<一句話說明這個梗，觀眾為什麼一聽就懂>",',
        '  "confidence": "<高|中|低>",',
        '  "runner_ups": [{"song": "...", "anchor": "...", "enter_line": "...", "why": "..."}]',
        '}',
    ]
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────────

def _http_json(url: str, payload: dict, headers: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_json(text: str) -> dict | None:
    """Pull the first balanced {...} out of a model reply (tolerates ``` fences)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def call_gemini(prompt: str) -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    data = _http_json(url, {"contents": [{"parts": [{"text": prompt}]}]}, {})
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_openai_compatible(prompt: str, *, base_url: str, api_key: str = "",
                           model: str = "") -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = _http_json(base_url.rstrip("/") + "/chat/completions",
                      {"model": model, "temperature": 0.2,
                       "messages": [{"role": "user", "content": prompt}]},
                      headers)
    return data["choices"][0]["message"]["content"] or ""


def run_judge(prompt: str, backend: str) -> tuple[str | None, dict | None]:
    """(raw_reply, parsed_json). backend='prompt' returns (None, None)."""
    if backend == "prompt":
        return None, None
    if backend == "gemini":
        raw = call_gemini(prompt)
    elif backend == "openai":
        raw = call_openai_compatible(
            prompt,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    elif backend == "vllm":
        url = os.environ.get("OMNI_VLLM_URL", "")
        if not url:
            raise RuntimeError("OMNI_VLLM_URL not set")
        raw = call_openai_compatible(
            prompt, base_url=url,
            model=os.environ.get("OMNI_VLLM_MODEL", "qwen3-omni-thinking"))
    else:
        raise ValueError(f"unknown backend '{backend}'")
    return raw, _extract_json(raw)
