"""Writes the cases the C++ tokenizer and the cutting of a text into chunks are checked against.

usage: uv run python text_cases.py <english|japanese> <out dir>

  text-cases.jsonl   one JSON object per line: the text; the ids the model's tokenizer gives it as the official code
                     calls it, and the text the tokenizer decodes from them; and the chunks generate_audio() speaks,
                     each as split_into_best_sentences() cuts it, as prepare_text_prompt() then prepares it, with the
                     frames it speaks after the end of speech and its tokens; or, for a text the official code refuses,
                     the error it raises
  meta.json          the model, as dump.py names it

Every model takes every text, so that each tokenizer meets the other's script too. The texts reach each rule of
prepare_text_prompt() (the replaced characters, the punctuation added or replaced at the end, the capital letter, the
padding of short texts), each branch of the sentence splitter (a decimal period, an ellipsis, a sentence longer than a
chunk cut at its commas and one that cannot be cut), the byte fallback, the added tokens and the spaces of both
tokenizers; and 200 random texts of the characters they treat apart.
"""

import json
import os
import random
import sys

from pocket_tts import TTSModel
from pocket_tts import default_parameters
from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences

from pins import model

name, out_dir = sys.argv[1:3]
os.makedirs(out_dir, exist_ok=True)
pinned = model(name)
config = pinned["config"]
tts = TTSModel.load_model(config=pinned["config_path"])
tokenizer = tts.flow_lm.conditioner.tokenizer

LONG = ("this sentence has no punctuation at all and it goes on and on with many words so that it holds more than fifty "
        "tokens in total which the splitter cannot cut into smaller pieces because it finds nothing to cut it at")
COMMAS = ("First comes a part of the sentence, then a second part that keeps going, then a third part with more words, "
          "then a fourth part that is still longer than the others, and a last part that ends it.")
TEXTS = [
    "Hello world.", "hello world", "Hello world", "Hello, world:", "Hello world -", "Hello world —", "\"Quoted.\"",
    "\"Hi?\", she said.", "(Parenthesized)", "[Bracketed] text; with: marks.", "Pi is 3.14 and e is 2.71. Next one!",
    "It costs 3. 5 apples remain.", "Version 1.5.2 is out.", "Wait... what?! Really?", "Mr. Smith went to Washington.",
    "U.S.A. is big. So is Canada.", "$3.50 each.", "a;b;c", "end with closers.\"", "end with weak,", "'single quotes'",
    "«guillemets» and „low quotes“.", "Ellipsis char… and more…", "Hmm… ok", "One. Two. Three. Four. Five. Six.",
    LONG, COMMAS, LONG + ". " + COMMAS, "Short. " + COMMAS + " " + LONG + ".",
    "Line one\nLine two\r\nLine three", "a   b    c", "  leading and trailing  ", "tab\tinside", "é is first", "élan vital",
    "über cool", "ß is a letter.", "ǆ digraph", "ﬁne ligature", "İstanbul", "ⅰ roman", "Ǳ big", "ǲ title", "😀 emoji first",
    "<s>special</s> tokens <pad> and <unk>.", "<0x41> literal byte piece", "...", "?!", ".", "a", "x y", "one two three four",
    "one two three four five",
    "こんにちは。今日はいい天気ですね。", "「引用」です。『二重』も。", "（括弧）と【隅付き】と〈山括弧〉と《二重山括弧》。",
    "全角数字１２３と全角！記号＃＄％＆＊＋－／：；＜", "3.14は円周率です。次の文。", "Englishと日本語の混在、そして続き。",
    "改行\nを含む\r\n文", "　全角空白で始まる文。", "～と〜と~と―と‐", "♪音楽♪と*強調*と#タグ", "疑問？感嘆！そして…三点リーダー…",
    "、" * 3 + "読点で始まる文", "長い文章" * 30 + "。", "読点で区切られた長い文、" * 8 + "そして終わり。",
    "", "   ", "\n", "“”",
]
# Random texts of the characters the rules treat apart, of a fixed seed, so that the cases are the same on every run.
PIECES = ["a", "word", "Big", "é", "ß", "3", "14", ".", "!", "?", "…", "...", ",", ";", ":", "-", "—", "\"", "'", "(", ")",
          "[", "]", "«", "»", "“", "”", "’", " ", " ", " ", "  ", "\n", "\t", "<s>", "</s>", "<unk>", "日本", "語", "。", "、",
          "「", "」", "！", "？", "ー", "～", "１", "😀", "\u00a0", "\u3000"]
rng = random.Random(0)
TEXTS += ["".join(rng.choice(PIECES) for _ in range(rng.randint(1, 60))) for _ in range(200)]


def case(text):
    out = {"text": text, "ids": tokenizer.encode(text)}
    out["decoded"] = tokenizer.decode(out["ids"])
    rules = dict(
        pad_with_spaces_for_short_inputs=config.pad_with_spaces_for_short_inputs,
        remove_semicolons=config.remove_semicolons,
        append_terminal_punctuation=config.append_terminal_punctuation,
        capitalize_first_letter=config.capitalize_first_letter,
        replace_characters=config.replace_characters,
    )
    try:
        chunks = split_into_best_sentences(tokenizer, text, default_parameters.MAX_TOKEN_PER_CHUNK, **rules)
        out["chunks"] = []
        for chunk in chunks:
            # As generate_audio_stream() prepares each chunk again and counts two frames more after the end of speech.
            prepared, guess = prepare_text_prompt(chunk, *rules.values())
            out["chunks"].append({"text": chunk, "prepared": prepared, "frames_after_eos": guess + 2,
                                  "tokens": tts.flow_lm.conditioner.prepare(prepared)[0].tolist()})
    except ValueError as e:
        out["error"] = str(e)
    return out


with open(os.path.join(out_dir, "text-cases.jsonl"), "w", encoding="utf-8") as f:
    for text in TEXTS:
        f.write(json.dumps(case(text), ensure_ascii=False) + "\n")
weights = config.weights_path.removeprefix("hf://")
with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump({"model": {"name": name, "repository": "/".join(weights.split("/")[:2]), "revision": weights.rsplit("@", 1)[1]}}, f, indent=1)
print("wrote", len(TEXTS), "cases to", out_dir)
