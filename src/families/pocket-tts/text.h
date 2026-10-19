#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "model-file.h"
#include "tokenizer.h"

namespace pocket {

/** A chunk of a text, which the model speaks starting again from the voice. */
struct Chunk {
    /** The chunk as the sentence splitter cuts it, and as it is prepared before the model reads it. */
    std::string text, prepared;
    std::vector<int32_t> tokens;
    /** The frames the model speaks after the end of speech. */
    int frames_after_eos = 0;
};

/**
 * What the official code does to a text before the model reads it, from the rules of the model file: it prepares the
 * text (prepare_text_prompt(): characters replaced, line breaks turned into spaces, a capital first letter, the
 * punctuation that ends a sentence, spaces in front of a short text), cuts it into sentences at the tokens of ".!...?"
 * but for a decimal period, cuts a sentence longer than a chunk at the tokens of ",;:", and joins the sentences into
 * chunks of at most text.max_tokens tokens each (split_into_best_sentences()), each prepared again. Strings are UTF-8 and
 * the rules Python's: whitespace is what str.isspace() takes, and the capital and the digits are those of the Python
 * that runs the official code.
 */
class Text {
public:
    explicit Text(const ModelFile & m);

    const Tokenizer & tokenizer() const { return tokenizer_; }

    /**
     * The text as prepare_text_prompt() prepares it, and the number of its words, which says how many frames the model
     * speaks after the end of speech. A text of nothing but whitespace, and of characters the rules delete, throws an
     * Error of the input, as does text that is not UTF-8.
     */
    std::pair<std::string, int> prepare(const std::string & text) const;

    /** The chunks of `text`, each as split_into_best_sentences() cuts it and generate_audio() prepares and tokenizes it. */
    std::vector<Chunk> chunks(const std::string & text) const;

private:
    /** The sentences of `text` grouped into chunks, as split_into_best_sentences() gives them. */
    std::vector<std::string> split(const std::string & text) const;
    /** The positions where segments of `tokens` begin after a run of `boundary` tokens, 0 first and the end last. */
    std::vector<size_t> boundaries(const std::vector<int32_t> & tokens, const std::vector<int32_t> & boundary, bool skip_decimals) const;

    Tokenizer tokenizer_;
    int max_tokens_ = 0, short_words_ = 0, frames_short_ = 0, frames_long_ = 0;
    bool pad_ = false, remove_semicolons_ = false, terminal_ = false, capitalize_ = false;
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> replace_;
};

}  // namespace pocket
