# Gag-Retrieval · Witty Song-Transition Retrieval

Given a Mandarin song that is currently playing, find in a lyrics corpus the next
song **and the exact line to cut into** that makes a **lyrically witty, instantly
obvious** connection — the way a live DJ does it, so the crowd gets the transition
the moment they hear it.

This is the **"song selection / cue-point selection" layer of an AI DJ medley
system**: it does not mix the two songs itself. It answers "which song comes next,
which line do we enter on, and why is it a gag", and hands options backed by full
evidence to a downstream transition agent.

---

## What kind of "gag" we want

Ideal examples (all well-known transitions in the Mandopop world):

| Previous song (A) | The line song B enters on | Gag type |
|---|---|---|
| *I Want to Fly* (the chorus keeps repeating "fly" and "I want to fly hard") | MC HotDog's *Fly Too Far*: "**Fly too far**, rushing up into the clouds" | Theme meets theme |
| Jolin Tsai's *Say Love You*: "This moment I finally have the courage to **say I love you**" | Cyndi Wang's *Love You*, signature chorus | Anadiplosis (chained echo) + title gag |
| David Tao's *Marry Me Today* | "Tomorrow I'll marry you" | Same line, reversed |

They share three properties, which are the design core of this project:

1. **The echo is of a meaning the song has hammered home, not a character that just
   happens to match.** *I Want to Fly* weaves "fly" into nine different lines, so
   the crowd spends the whole song immersed in the idea of flying, and *Fly Too Far*
   lands instantly. By contrast, if "happy" appears in only one line of a song (even
   looped three times), it is not what that song is about, and transitioning on it
   gets no reaction (**same word, different meaning = fake gag**).
2. **B must enter on a signature line people recognize instantly** (title line >
   chorus > opening line). Line 37 of an obscure song gets no laugh, however neat
   the wordplay.
3. **The echo has to happen in the first second of B.** An echo buried at the end of
   B's line arrives eight syllables late, and the gag falls flat.

---

## Two-stage architecture

```
            Given the playing song A + a lyrics corpus (N can be arbitrarily large)
                                     │
     ┌───────────────────────────────▼─────────────────────────────────┐
     │  Stage 1 · Retrieval (zero LLM; index lookups independent of N)  │
     │   character-bigram inverted index + line-head chaining index     │
     │   rank = anchor saturation (spread) × B-line recognizability     │
     │          × entry timing × popularity                             │
     │   → top-K candidate songs, each with "evidence line pairs"       │
     └───────────────────────────────┬─────────────────────────────────┘
                                     │  only 5–10 songs remain
     ┌───────────────────────────────▼─────────────────────────────────┐
     │  Stage 2 · LLM judge (Gemini / OpenAI-compatible / local vLLM)   │
     │   gets full lyrics of A and candidates + each song's theme list  │
     │   + transition-point hints                                       │
     │   decides: drop same-word-different-meaning, extend semantic     │
     │   fields (fly ↔ wings / clouds), pick the most obvious line      │
     │   → next song + A exit line + B entry line + gag type + reason   │
     │     + confidence                                                 │
     └───────────────────────────────┬─────────────────────────────────┘
                                     ▼
          Handed to the downstream transition agent
          (cue = end of exit line / start of entry line)
```

**Why two stages:** the full lyrics of N songs don't fit into one LLM call, but the
lexical / chaining indexes can narrow the field to 5–10 songs instantly (query cost
is independent of N; to scale up, swap the dict for Elasticsearch). Stage 1 only
does recall + rough ranking (it pulls in everything that might be a gag and makes no
final taste call). Stage 2 makes the judgments only an LLM can make: which one is
most obvious, which look lexically similar but are not actually funny and should be
dropped, and which hidden gems are semantic extensions.

**Why the LLM is a judge, not a comedian:** the candidates have already been
filtered by lexical rules down to ones with a clear transition point. The LLM's job
is to **verify / re-pick / reject** correspondences that were already found
(discriminating is far easier than generating), not to freely invent jokes. The
prompt deliberately withholds retrieval scores, requires the model to read the full
lyrics and re-judge on its own, and includes positive and negative examples to
calibrate confidence.

---

## Stage 1 signals (all computed automatically from LRC structure, zero annotation)

| Signal | Meaning | How it's computed |
|---|---|---|
| **Saturation spread** | How many **distinct** lines of A the anchor word is woven into = whether it is what the song "means" | Number of distinct lines containing the word (≥2 counts as a theme; a single looped line does not) |
| **Anchor identity** | Whether the word identifies one of the songs | Title word 1.0 > theme word 0.6 > ordinary word 0.3 |
| **B-line recognizability** | Whether the crowd recognizes the line B enters on | Title line > chorus (repeated line) > opening line > ordinary line |
| **Entry timing** | Whether the echo lands in B's first second | Anchor at the start of B's line ≫ buried at the end |
| **Chained echo (anadiplosis)** | B's first character picks up A's last word ("…say I love you" → "Love is…") | Dedicated line-head index (such pairs share no bigram) |

**Rank, don't exclude:** high-frequency phrases (e.g. "love you") are legitimate
gags (it is the title of *Love You*). The only hard exclusions are pure function
words and grammatical scaffolding (time / grammar / pronoun combinations such as
"of-look", "this moment", "today", "give me"). The criterion is that the anchor must
be the semantic focus of the line, not a framing word.

Stage 2 gets more than the candidates: it also gets each song's **theme list**
(`theme_profile`: the words hammered home repeatedly that stay in the crowd's ears).
That lets it match **semantic fields**, which lexical retrieval cannot: if A hammers
"fly", B answering with "wings" or "rushing up into the clouds" also counts.

---

## Installation and usage

**No third-party dependencies** (pure Python standard library); just run it with
the system `python3`.

```bash
git clone <this-repo>
cd gag-retrieval

# 1) CLI: Stage 1 retrieval + generate the Stage 2 prompt (no API key needed)
#    <song-title> = a lyrics filename in lrc_test/ without the extension
python3 -m gag_retrieval.run --corpus lrc_test --query <song-title>

# 2) Call the Stage 2 LLM judge directly
export GEMINI_API_KEY=your_key          # or put it in gag_retrieval/gemini.key (gitignored)
python3 -m gag_retrieval.run --corpus lrc_test --query <song-title> --llm gemini

# 3) Demo web page (browse the library, preview every line, dual-deck transition preview, live judge)
python3 -m gag_retrieval.server --corpus lrc_test --port 8010
#   → open http://127.0.0.1:8010 in a browser

# 4) Golden-example self-test (21 checks: four gag types + saturation + chaining + various noise exclusions)
python3 -m gag_retrieval.selftest
```

> **Port already in use?** Run `pkill -f gag_retrieval.server` first, or use `--port 8011`.

### Corpus format

One `.txt` (or `.lrc`) per song, **filename = song title**, each line `[mm:ss.xx]lyric`:

```
[00:34.97]<lyric line>
[00:40.22]<lyric line>
```

Credit lines (lyricist, composer, etc.) are skipped automatically. Mixing Traditional
and Simplified Chinese is fine (built-in Traditional→Simplified normalization; if
`opencc` is installed it is used automatically). To preview audio in the demo, put
an audio file with the same name (`.mp3/.m4a/.wav`) in the same folder (messy
filenames are fine; they are fuzzy-matched by song title).

### Demo page features

- **Library**: pick a song as "now playing".
- **RETRIEVE**: runs Stage 1 (option cards: anchor, gag type, saturation, A exit
  line / B entry line with timestamps, B-line recognizability, remaining seconds,
  section position) + Stage 2 (Gemini judge card).
- **Per-line preview**: the ▶ next to a line plays from that second.
- **Transition preview**: dual-deck Web Audio crossfade. A plays up to the exit
  line and fades out with equal power while B fades in from the entry line, so you
  hear the actual transition.
- **Medley**: any candidate can be "set as now playing" to keep chaining
  (already-played songs are excluded automatically).
- **Audio calibration**: YouTube rips often have extra intros; use the sidebar
  "Calibrate" button to align each song's time offset (saved to `offsets.json` and
  applied to all playback and previews).

---

## As a tool in the larger AI DJ system

`gag_retrieval/tool.py` wraps the same engine as two functions for agents to call,
matching two decision layers:

```python
from gag_retrieval.tool import GagCatalog
cat = GagCatalog("lrc_test")               # load once

# Tool 1 — song-selection layer (in the orchestrator / medley main loop, outside the transition agent)
cat.suggest_next_song("<song A>", k=5, exclude={"<already played song>"})

# Tool 2 — cue-selection layer (song pair already fixed; injected into the transition agent's seed / used as an inspect tool)
cat.lyric_hooks("<song A>", "<song B>", k=5)
```

**How to make it a "non-skippable first stage" rather than an optional tool:**

- The **song-selection layer** is inherently non-skippable: the transition agent
  receives two songs that have already been chosen, and that choice happens in the
  orchestrator outside it. `suggest_next_song` is a mandatory step of that
  orchestrator.
- To make the **cue-selection layer** non-skippable, **write the output of
  `lyric_hooks` into the transition agent's seed prompt** (data in the prompt cannot
  be skipped), or, following the existing agent gating of "must inspect before
  build", register it as an inspect tool that has to be called first.

**Key design principle: the tool only attaches honest metadata; it does not filter
feasibility on the agent's behalf.** Every option carries both "gag evidence"
(anchor, gag type, saturation) and "objective facts" (cue timestamps, B's remaining
runway, section position). A lexically hilarious transition with a 40 BPM gap and
clashing keys is still returned, because that may be exactly the one a generative
bridge (`generate_bridge`) can rescue, and rescuing it is the most impressive
result. **Weighing gag score against acoustic feasibility is left to the agent's
ears; the tool does not overstep.** The returned `line_a.end` / `line_b.start` are
the downstream `a_out` / `b_in` cues.

---

## Modules

```
gag_retrieval/
  zhnorm.py     text normalization (NFKC / Traditional→Simplified / punctuation / English word boundaries)
  lrc.py        LRC parsing → line groups (repeat count = chorus signal)
  hooks.py      anchor identity, saturation spread, theme_profile
  lexical.py    bigram inverted index + line-head chaining index + anchor extraction
  meta.py       line positional metadata (line_end / section / runway / pos_pct)
  retrieve.py   Stage 1 orchestration (score line pairs → aggregate per song → top-K)
  judge.py      Stage 2 prompt + LLM backends (Gemini / OpenAI-compatible / vLLM, pure urllib)
  tool.py       two functions for agents (suggest_next_song / lyric_hooks)
  server.py     demo web backend (pure http.server, with audio Range streaming)
  demo.html     single-file frontend (light theme, dual-deck preview, calibration)
  align.py      timestamp refinement skeleton (official lyrics × Whisper clock, character alignment; not wired in yet)
  run.py        command line
  selftest.py   golden-example tests
```

### Optional: timestamp refinement (align.py)

When the line-level LRC timestamps are inaccurate, use **Whisper's clock with the
official lyrics' characters**: separate vocals with Demucs → run Whisper to get its
own word-level timings (the characters may be wrong) → align Whisper's character
stream with the official lyrics so the official lyrics inherit the precise timings.
It doesn't matter if ASR mishears a word as a homophone: as long as the positions
line up, the timing transfers and the lyric text stays the official version. (The
skeleton is done but not yet wired into the demo.)

---

## License and data

The code is for research / demonstration purposes. The lyrics and audio in
`lrc_test/` belong to their respective copyright holders and are included only as
test samples for a technical demo; please do not redistribute them. `gemini.key`,
`out/`, `offsets.json`, audio files, etc. are in `.gitignore`.
