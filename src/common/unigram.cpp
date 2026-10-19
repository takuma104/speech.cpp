#include "unigram.h"

#include <algorithm>
#include <cstdio>

#include "error.h"

namespace {

/** The Unigram path ends at a byte offset; `start` is where its last piece starts, -1 while unreached. */
struct PathEnd {
    int32_t id = -1;
    double score = 0;
    int64_t start = -1;
};

/** The penalty under the lowest piece score for a character no piece covers, sentencepiece's kUnkPenalty. */
constexpr double kUnknownPenalty = 10.0;

size_t utf8_length(unsigned char c) {
    return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
}

}  // namespace

Unigram::Unigram(const std::vector<std::string> & tokens, const std::vector<double> & scores, int32_t unknown, const std::string & source,
                 const std::vector<bool> * matched)
    : scores_(scores), unknown_(unknown) {
    if (scores.size() != tokens.size()) throw Error(Fault::File, source + "the tokenizer does not have a score for each token");
    if (unknown < 0 || (size_t) unknown >= tokens.size()) throw Error(Fault::File, source + "the tokenizer's unknown piece is not one of its pieces");
    std::unordered_map<std::string, int32_t> all;
    double lowest = 0;
    for (size_t i = 0; i < tokens.size(); i++) {
        all.emplace(tokens[i], (int32_t) i);
        lowest = std::min(lowest, scores[i]);
        if (matched && !(*matched)[i]) continue;
        ids_.emplace(tokens[i], (int32_t) i);
        longest_ = std::max(longest_, tokens[i].size());
    }
    unknown_score_ = lowest - kUnknownPenalty;
    for (int b = 0; b < 256; b++) {
        char name[8];
        std::snprintf(name, sizeof name, "<0x%02X>", b);
        const auto it = all.find(name);
        if (it == all.end()) throw Error(Fault::File, source + "the tokenizer has no byte token " + name);
        bytes_[b] = it->second;
    }
}

void Unigram::encode(const std::string & s, std::vector<int32_t> & ids) const {
    const size_t n = s.size();
    std::vector<PathEnd> best(n + 1);
    best[0].start = 0;
    for (size_t start = 0; start < n;) {
        const size_t char_length = std::min(utf8_length((unsigned char) s[start]), n - start);
        const double till_here = best[start].score;
        bool covered = false;
        for (size_t length = 1; length <= longest_ && start + length <= n; length++) {
            const auto it = ids_.find(s.substr(start, length));
            if (it == ids_.end()) continue;
            PathEnd & end = best[start + length];
            const double score = till_here + scores_[it->second];
            if (end.start < 0 || score > end.score) end = {it->second, score, (int64_t) start};
            if (length == char_length) covered = true;
        }
        if (!covered) {
            PathEnd & end = best[start + char_length];
            const double score = till_here + unknown_score_;
            if (end.start < 0 || score > end.score) end = {unknown_, score, (int64_t) start};
        }
        start += char_length;
    }
    // Back from the end; consecutive unknown characters are fused into one piece, then spelled in bytes.
    std::vector<std::pair<std::string, int32_t>> pieces;
    for (size_t end = n; end > 0;) {
        const PathEnd & p = best[end];
        const std::string text = s.substr((size_t) p.start, end - (size_t) p.start);
        if (p.id == unknown_ && !pieces.empty() && pieces.back().second == unknown_) {
            pieces.back().first = text + pieces.back().first;
        } else {
            pieces.push_back({text, p.id});
        }
        end = (size_t) p.start;
    }
    for (auto it = pieces.rbegin(); it != pieces.rend(); ++it) {
        if (it->second != unknown_) {
            ids.push_back(it->second);
            continue;
        }
        const auto known = ids_.find(it->first);
        if (known != ids_.end()) {
            ids.push_back(known->second);
            continue;
        }
        for (unsigned char b : it->first) ids.push_back(bytes_[b]);
    }
}
