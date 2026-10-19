# Pocket TTS requests take a temperature, steps and max_seconds, and the model's own settings for the rest

## Context

Pocket TTS makes its speech a latent of 80 ms at a time: the flow language model reads the voice prompt, the text and
the latents so far, and its flow head turns Gaussian noise into the next latent, while an end-of-speech head says when
to stop. The official `TTSModel` takes its settings in two places. `load_model()` takes `temp`, the variance of the
noise (0.3 by the configuration), `sampler_decode_steps`, the steps of the flow head (1), `noise_clamp`, a bound of the
noise (none), and `eos_threshold`, the logit above which speech ends (-4). `generate_audio()` takes the text, the voice
state, `frames_after_eos`, the frames spoken after the end (3 for a text of 4 words or fewer and 1 otherwise, each plus
2), `max_tokens`, the tokens of a chunk of the text (50), and `copy_state`; `get_state_for_audio_prompt()` takes the
voice's recording and `truncate`, which keeps its first 30 s and which the official server and `export-voice` set. Each
chunk of the text speaks at most its tokens / 3 + 2 s. The noise comes from PyTorch's generator, which nothing outside
PyTorch reproduces.

## Decision

- **`temperature` is the official `temp`**, the variance of the noise each latent is drawn from, from 0 (no noise) up,
  its default the file's, which the converter takes from the configuration's `default_temperature`.
- **`steps` is the official `sampler_decode_steps`**, the steps of LSD from noise to the latent, from 1 up, its default
  the file's 1.
- **`max_seconds` bounds the speech of a whole request**, as Qwen3-TTS's and Drama 3's do: the official code bounds
  each chunk alone, and a long text speaks for as long as its chunks add up to.
- **A chunk that reaches its longest before the end of speech ends there**, as the official code ends it, and the request
  goes on to the next chunk and reports `model_limit` at its end, where the official code only logs a warning.
- **`seed` seeds speech.cpp's own draw of the noise.** The same seed gives the same audio on the same device, and not
  the official audio, whose noise PyTorch draws.
- **The rest stays the model's own**: `noise_clamp`, `eos_threshold`, `frames_after_eos` and `max_tokens` keep the
  official defaults, which the file holds. A request may not set them.
- **A voice longer than 30 s is refused**, naming it, where the official server and `export-voice` keep its first 30 s.
- **`language` is only checked** against the model's one language, which a request does not choose.
- **Not offered:** `speed`, `seconds` and `duration_scale`, since the model decides the length as it speaks;
  `copy_state`, since every request starts from its voice; `quantize`, the official int8 dynamic quantization, since
  `speech quantize` makes files of other weight types.

The alternatives were turned down:

- `noise_clamp`, `eos_threshold`, `frames_after_eos` and `max_tokens` as options. Each would be a new name in the C
  API's vocabulary for one family, and their defaults are the settings the models were evaluated with.
- Cutting a voice at 30 s, as the official server does. The voice would lose what the caller gave without a word.
- Drawing the noise as PyTorch does, so that a seed gives the official audio. It would tie the library to PyTorch's
  generator for no gain in what anyone hears, and the checks start each step from the official noise instead.

## Consequences

The C API's `temperature` covers a variance of noise as well as a divisor of logits, and its documentation says both.
