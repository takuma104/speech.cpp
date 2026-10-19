# Pocket TTS passes each frame of its speech as it is made, decoded a frame at a time

## Context

Pocket TTS makes its speech a latent of 80 ms at a time, and Mimi's decoder turns latents into samples, carrying the
state of every causal stage from one call to the next. The official generate_audio() decodes in a thread of its own
whatever latents have queued since its last call, one at a time on a machine that decodes faster than the model makes
them. The decoder's samples depend on the grouping only by rounding, and on Metal by more: one frame is 16 columns at
the transformer's rate, which ggml multiplies with its few-row kernels, whose sums run in another order than its kernel
for more columns. Measured on an Apple M4 Max with the F32 files on 2026-10-20, decoding a frame at a time departs from
calls of 2 to 8 frames by 61.5 to 63.9 dB SNR on Metal and by 129.9 dB or more on the CPU, and a request spoke 20 to 22
times faster than real time on Metal and 12 to 14 times on the CPU, its first audio 0.012 to 0.015 s after the request
on Metal.

## Decision

- **Each step's latent is decoded alone and passed on at once**, 1920 samples, the first 5 ms of a chunk faded in as
  the official code fades them. The schedule does not depend on time, so the same request with the same seed gives the
  same samples on the same device.
- **A cancel waits at most one step and one frame's decoding.**

The alternatives were turned down:

- Decoding the frames queued since the last call, as the official thread does, or groups sized by the measured speed.
  On Metal the samples would change with the machine's load.
- A fixed group of several frames a call. The first audio would wait for the group's steps, for a decoding that is
  already many times faster than real time a frame at a time.
