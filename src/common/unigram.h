#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * A SentencePiece Unigram model, which cuts a text into the pieces of the path of the highest total score, as Hugging
 * Face tokenizers and sentencepiece both run it. A character that no piece of its own length covers may stand alone as
 * an unknown piece, scored 10 below the lowest piece; consecutive unknown characters are fused into one, then spelled
 * in the byte pieces <0x00> to <0xFF>. Of two paths of the same score, the one found first stays: the pieces that start
 * earlier, and of those the shorter.
 */
class Unigram {
public:
    /**
     * The model of `tokens` and their `scores`, whose unknown piece is `unknown`. `matched`, where given, says of each
     * piece whether a text may contain it: sentencepiece never matches its control, unknown and byte pieces in a text,
     * where tokenizers matches every piece. A model without a byte piece of every byte, or a score of every piece,
     * throws an Error of the file, its message starting with `source`.
     */
    Unigram(const std::vector<std::string> & tokens, const std::vector<double> & scores, int32_t unknown, const std::string & source,
            const std::vector<bool> * matched = nullptr);

    /** Appends the ids of the pieces of `text`, one path over all of it, to `ids`. */
    void encode(const std::string & text, std::vector<int32_t> & ids) const;

    /** The id of the byte piece of `b`. */
    int32_t byte_piece(unsigned char b) const { return bytes_[b]; }

private:
    std::unordered_map<std::string, int32_t> ids_;
    std::vector<double> scores_;
    size_t longest_ = 0;
    double unknown_score_ = 0;
    int32_t unknown_ = 0;
    int32_t bytes_[256];
};
