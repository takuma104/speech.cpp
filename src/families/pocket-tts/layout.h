#pragma once

#include "model-file.h"

namespace pocket {

/**
 * Layout 1 of a Pocket TTS model file, which reference/pocket-tts/convert.py writes: the tokenizer and the rules the
 * official code applies to a text, the flow language model with its LSD flow head, and Mimi, the codec, whose encoder
 * makes a voice from a recording and whose decoder turns latents into samples; every key of pocket-tts.* and speech.*
 * that the family reads, and the tensors the keys call for.
 */
extern const Layout model_layout;

}  // namespace pocket
