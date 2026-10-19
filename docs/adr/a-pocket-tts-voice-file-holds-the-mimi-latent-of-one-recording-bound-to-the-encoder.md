# A Pocket TTS voice file holds the Mimi latent of one recording, bound to the encoder

## Context

The official Pocket TTS speaks in the voice of one recording: `get_state_for_audio_prompt()` reads it, resamples it to
24 kHz with scipy's `resample_poly()`, ends it on a pause (`end_on_pause()`), encodes it with Mimi's encoder to a latent
of 32 values every 80 ms, projects the latent into the flow language model (`speaker_proj`) and runs the model over it,
whose keys and values are the voice's state. `export-voice` saves that state, about 6 MB for 10 s with a 6-layer model,
which only the weights it was made with can read. The Japanese model of takuma104/pocket-tts-ja keeps the official
encoder bit for bit, with its own decoder and flow language model, so both models encode a recording to the same
latent (hash cab0835d of the encoder's tensors). Encoding 13.7 s took 0.33 s on the CPU and 0.08 s on Metal of an Apple
M4 Max (F32, 2026-10-20), and the model runs over a latent of 10 s in milliseconds.

## Decision

- **A voice file holds the latent of one recording**, made by `speech voice MODEL REF.wav VOICE.gguf` or
  `speech_voice_make()`, with the recording's length and rate and the kind of device that encoded it. A WAVE file is
  added as a voice too, encoded when it is added.
- **It binds to Mimi's encoder by a hash**, `pocket-tts.mimi.encoder_sha256`, the SHA-256 of the official encoder's
  tensors, which the converter writes; a voice made with one Pocket TTS model speaks with every model of the same
  encoder, in any weight type, and a model of another encoder refuses it.
- **The recording is read as the official code reads it**: its channels averaged, ended on a pause, encoded whole. It is
  resampled with the library's kaiser_best filter, as Irodori-TTS's references are; a recording at 24 kHz gives the
  official latent, and one at another rate a latent of another filter (a 16 kHz FLEURS recording read 40.4 dB from the
  official reading, 2026-10-20).
- **A voice is one recording, as it is.** Several references, a loudness and an embedding are refused naming them, and so
  is a recording longer than 30 s, where the official server keeps its first 30 s.

The alternatives were turned down:

- The official state in the voice file. It would bind a voice to one model's weights, at about 400 times the size of
  the latent for a 6-layer model.
- `speaker_proj` of the latent. It binds the voice to one model's projection and saves nothing worth having.
- scipy's `resample_poly()` for Pocket TTS's recordings. It would give the official latent at every rate, for a second
  filter in the library whose likeness to the speaker nothing has measured against kaiser_best's.
- Several recordings joined, as Irodori-TTS joins its references. The official code takes one, and nothing could check
  a prompt of several against it.

## Consequences

A voice made with the English model speaks with the Japanese one and back. A voice file names 0.8.11 in
`speech.requires`, the first release that reads its layout 1.
