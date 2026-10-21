# Pocket TTS

This page describes speech.cpp's port of Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts), which speaks
a text of any length in the voice of a recording, a frame of 80 ms at a time.

| Model | Upstream | Language | File |
|---|---|---|---|
| English | [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), `languages/english` | en | `Pocket-TTS-110M-English-Q8_0.gguf`, 145 MB |
| Japanese | [takuma104/pocket-tts-ja](https://huggingface.co/takuma104/pocket-tts-ja), `japanese` | ja | `Pocket-TTS-110M-Japanese-Q8_0.gguf`, 145 MB |

The Japanese model was trained from the English one and reads Japanese as it is written, kanji included. Both share
the codec's encoder, so a voice made with one speaks with the other.

```sh
speech voice Pocket-TTS-110M-English-Q8_0.gguf me.wav me.voice.gguf
speech tts Pocket-TTS-110M-English-Q8_0.gguf --add-voice me=me.voice.gguf --voice me -o out.wav "Hello world. This is a test."
speech tts Pocket-TTS-110M-Japanese-Q8_0.gguf --add-voice me=me.voice.gguf --voice me -o out.wav "今日はいい天気ですね。"
```

## What it does

A transformer of 6 layers makes the speech a frame of 80 ms at a time, and Mimi, the codec, decodes each frame into
24 kHz audio as soon as it is made, so the first audio comes after one frame.

- A text of any length is one request. It is cut into chunks of at most 50 tokens at its sentences, and a sentence
  longer than that at its commas, as the official code cuts it; each chunk starts again from the voice, and the chunks'
  audio is joined without a pause.
- The text is prepared as the official code prepares it. The English model capitalizes the first letter and ends the
  text with a period. The Japanese model turns 。 and 、 into . and ,, full-width letters and digits into half-width
  ones, and deletes brackets and quotation marks.
- The speech ends where the model says it ends, a few frames later.

Not supported: the official predefined voices (`alba`, `marius` and the rest), the other languages and the 24-layer,
drifting and small models of kyutai/pocket-tts, the official int8 quantization, and the official settings
`noise_clamp`, `eos_threshold`, `frames_after_eos` and `max_tokens`, which keep their defaults. A seed gives other audio
than the same seed in the official code.

## Voices

The model has no voice of its own: a request speaks in a voice you add.

- **A WAVE file** of at most 30 s, at any rate. Adding it encodes it, which took 0.08 s on Metal and 0.33 s on the CPU
  for 13.7 s of audio (Apple M4 Max). Its trailing silence is cut and a short pause added.
- **A voice file**, which `speech voice` writes once from one WAVE file. It loads at once and works with both models in
  any type.

The voice copies the recording's sound, its noise and room included, so a clean recording gives clean speech.

## Options

| Option | Default | Range and notes |
|---|---|---|
| `voice` | required | a voice added since loading |
| `language` | `auto` | the model's one language; only checked |
| `seed` | drawn | 0 to 2^53 - 1 |
| `temperature` | 0.3 | 0 or more: the variance of the noise each frame is drawn from; at 0 every seed gives the same audio |
| `steps` | 1 | 1 or more: the steps from the noise to each frame |
| `max_seconds` | none | above 0: the longest the whole speech may be |

A chunk that reaches the longest the model speaks for it, its tokens / 3 + 2 s, before the speech ends stops there; the
request goes on with the next chunk and ends with the stop reason `model_limit`.

## Files

| Type | English | Japanese |
|---|---|---|
| F16 | 219 MB | 219 MB |
| Q8_0 | 145 MB | 145 MB |
| Q6_K | 126 MB | 126 MB |

The codec and the part that turns noise into each frame stay in F16 in every type, so there is no Q5_K or Q4_K file.

## Speed

With a voice of a recording, a Japanese text of 15 s and an English one of 10 s, on an Apple M4 Max on 2026-10-20 and on
Windows with an RTX 5090 and a Core Ultra 9 285 on 2026-10-21:

| Device | Type | RTFx | Time to first audio |
|---|---|---|---|
| Metal, M4 Max | F16 | 24 to 26 | 0.010 to 0.013 s |
| Metal, M4 Max | Q8_0 | 26 to 27 | 0.011 to 0.013 s |
| CPU, M4 Max | F16 | 18 to 19 | 0.024 to 0.048 s |
| CPU, M4 Max | Q8_0 | 20 to 21 | 0.020 to 0.039 s |
| Vulkan, RTX 5090 | F16 | 32 | 0.014 to 0.026 s |
| Vulkan, RTX 5090 | Q8_0 | 32 | 0.015 to 0.026 s |
| CPU, Core Ultra 9 285 | F16 | 8.3 to 8.7 | 0.058 to 0.108 s |
| CPU, Core Ultra 9 285 | Q8_0 | 8.3 to 9.5 | 0.049 to 0.097 s |

## License

The English model is Kyutai's, under CC-BY-4.0. Kyutai's terms forbid voice cloning without the speaker's explicit
consent, deception, and presenting generated speech as a real recording. The Japanese model is under CC-BY-NC-4.0, for
non-commercial use alone, since it was trained on the Emilia dataset, and under the same terms.
