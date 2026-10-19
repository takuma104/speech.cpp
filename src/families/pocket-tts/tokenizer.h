#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model-file.h"
#include "unigram.h"

namespace pocket {

/**
 * The tokenizer of a Pocket TTS model, a SentencePiece Unigram model with byte fallback, which encodes and decodes a
 * text as the library the official code runs it through does. Hugging Face tokenizers, which reads a tokenizer.json:
 * the added tokens are split out of the text first, each other part gets "▁" in front and every space turned into "▁",
 * and is cut into words before each "▁", each word a path of its own; decoding skips the added tokens, drops every "▁"
 * of the first piece and turns the others into spaces, and spells a run of byte pieces that is no UTF-8 as one U+FFFD a
 * byte. sentencepiece, which reads a .model: the text gets "▁" in front and every space turned into "▁", one path over
 * all of it, no control, unknown or byte piece matched in it; decoding skips the control pieces, spells the unknown
 * piece " ⁇ ", drops the "▁" in front of the first other piece, and spells each byte that starts no UTF-8 character as
 * U+FFFD.
 */
class Tokenizer {
public:
    explicit Tokenizer(const ModelFile & m);

    std::vector<int32_t> encode(const std::string & text) const;
    /** The text of `n` ids from `ids`, as the official code's tokenizer.decode() gives it. */
    std::string decode(const int32_t * ids, size_t n) const;
    std::string decode(const std::vector<int32_t> & ids) const { return decode(ids.data(), ids.size()); }

private:
    bool sentencepiece_ = false;
    std::vector<std::string> tokens_;
    Unigram unigram_;
    /** The added tokens tokenizers matches in a text and skips in decoding, empty for sentencepiece. */
    std::vector<int32_t> special_;
    int32_t unknown_ = 0;
    /** The byte each byte piece stands for, -1 for every other piece. */
    std::vector<int> byte_of_;
    /** sentencepiece's control pieces, which decode to nothing. */
    std::vector<bool> control_;
};

}  // namespace pocket
