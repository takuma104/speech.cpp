# Pocket TTS has no built-in voices and speaks in voices made from recordings

## Context

The official Pocket TTS names 27 predefined voices, such as `alba` and `estelle`. Each is the flow language model's
keys and values after the voice's recording, a state precomputed for one language's weights and published beside them
(`embeddings/<voice>.safetensors`): about 6 MB a voice for a model of 6 layers and 25 MB for one of 24, 165 MB and
660 MB for the 27, against 219 MB for the 6-layer model's weights in bfloat16. The official code refuses a predefined
voice for any other weights. The recordings behind them come from kyutai/tts-voices, each under its own license, some
non-commercial (`cosette` is CC-BY-NC-4.0). takuma104/pocket-tts-ja publishes no state, only recordings to clone. Every
official request starts from a voice's state.

## Decision

- **A Pocket TTS model file holds no voice**, and its model information lists none.
- **A request names a voice the program added**: a voice file made from recordings, or a WAVE file, which the model
  encodes with Mimi and prompts with as the official `get_state_for_audio_prompt()` does.

The alternatives were turned down:

- The official states in the model file. They would make the file of a 6-layer model nearly twice as large, and bind
  every voice to one language's weights.
- Voices made from kyutai/tts-voices' recordings in the model file. Each recording's license would travel with every
  file, some of them non-commercial, for voices a caller can make from any recording.
- A voice of the model's own choosing, as Irodori-TTS's `none` is. The official code has none to check it against.

## Consequences

A program that speaks with Pocket TTS adds a voice before its first request; the documentation of the model says how.
