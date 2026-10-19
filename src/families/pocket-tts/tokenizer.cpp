#include "tokenizer.h"

#include <algorithm>
#include <cstdio>

#include "error.h"
#include "utf8.h"

namespace pocket {

namespace {

const std::string kSpace = "\xE2\x96\x81";
const std::string kReplacement = "\xEF\xBF\xBD";

/** `s` with every space turned into "▁", after a "▁" of its own. */
std::string escaped(const std::string & s) {
    std::string out = kSpace;
    for (char c : s) {
        if (c == ' ') {
            out += kSpace;
        } else {
            out += c;
        }
    }
    return out;
}

/** `s` with every "▁" turned into a space. */
std::string unescaped(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size();) {
        if (s.compare(i, kSpace.size(), kSpace) == 0) {
            out += ' ';
            i += kSpace.size();
        } else {
            out += s[i++];
        }
    }
    return out;
}

/** The bytes of `run` as sentencepiece decodes them: each UTF-8 character as it is, each byte that starts none as U+FFFD. */
std::string lenient(const std::string & run) {
    std::string out;
    for (size_t i = 0; i < run.size();) {
        const unsigned char c = (unsigned char) run[i];
        const size_t length = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (length > 0 && i + length <= run.size() && decode_utf8(run.substr(i, length))) {
            out += run.substr(i, length);
            i += length;
        } else {
            out += kReplacement;
            i++;
        }
    }
    return out;
}

Unigram make_unigram(const ModelFile & m) {
    const std::string p = "pocket-tts.tokenizer.";
    const std::vector<std::string> tokens = m.str_array(p + "tokens");
    const int32_t unknown = (int32_t) m.u32(p + "unknown_id");
    if (m.str(p + "runtime") != "sentencepiece") return Unigram(tokens, m.f64_array(p + "scores"), unknown, m.path() + ": ");
    std::vector<bool> matched(tokens.size(), true);
    matched.at((size_t) unknown) = false;
    if (m.has(p + "control_ids")) {
        for (int32_t id : m.i32_array(p + "control_ids")) matched.at((size_t) id) = false;
    }
    for (size_t i = 0; i < tokens.size(); i++) {
        unsigned value;
        char end;
        if (tokens[i].size() == 6 && std::sscanf(tokens[i].c_str(), "<0x%2X%c", &value, &end) == 2 && end == '>') matched[i] = false;
    }
    return Unigram(tokens, m.f64_array(p + "scores"), unknown, m.path() + ": ", &matched);
}

}  // namespace

Tokenizer::Tokenizer(const ModelFile & m) : unigram_(make_unigram(m)) {
    const std::string p = "pocket-tts.tokenizer.";
    sentencepiece_ = m.str(p + "runtime") == "sentencepiece";
    tokens_ = m.str_array(p + "tokens");
    unknown_ = (int32_t) m.u32(p + "unknown_id");
    if (m.has(p + "special_ids")) special_ = m.i32_array(p + "special_ids");
    control_.assign(tokens_.size(), false);
    if (m.has(p + "control_ids")) {
        for (int32_t id : m.i32_array(p + "control_ids")) control_.at((size_t) id) = true;
    }
    byte_of_.assign(tokens_.size(), -1);
    for (int b = 0; b < 256; b++) byte_of_.at((size_t) unigram_.byte_piece((unsigned char) b)) = b;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    std::vector<int32_t> ids;
    if (sentencepiece_) {
        if (!text.empty()) unigram_.encode(escaped(text), ids);
        return ids;
    }
    // A part of the text between added tokens, cut into words before each "▁", each a path of its own.
    const auto part = [&](const std::string & s) {
        if (s.empty()) return;
        const std::string e = escaped(s);
        for (size_t start = 0; start < e.size();) {
            size_t next = e.find(kSpace, start + kSpace.size());
            if (next == std::string::npos) next = e.size();
            unigram_.encode(e.substr(start, next - start), ids);
            start = next;
        }
    };
    std::string pending;
    for (size_t i = 0; i < text.size();) {
        // The leftmost and then longest added token, matched on the text before anything else.
        int32_t added = -1;
        for (int32_t id : special_) {
            const std::string & t = tokens_[(size_t) id];
            if (text.compare(i, t.size(), t) == 0 && (added < 0 || t.size() > tokens_[(size_t) added].size())) added = id;
        }
        if (added >= 0) {
            part(pending);
            pending.clear();
            ids.push_back(added);
            i += tokens_[(size_t) added].size();
        } else {
            pending += text[i++];
        }
    }
    part(pending);
    return ids;
}

std::string Tokenizer::decode(const int32_t * ids, size_t n) const {
    std::string out, run;
    const auto flush = [&]() {
        if (run.empty()) return;
        if (sentencepiece_) {
            out += lenient(run);
        } else if (decode_utf8(run)) {
            out += run;
        } else {
            for (size_t i = 0; i < run.size(); i++) out += kReplacement;
        }
        run.clear();
    };
    bool first = true;
    for (size_t i = 0; i < n; i++) {
        const int32_t id = ids[i];
        if (id < 0 || (size_t) id >= tokens_.size()) throw Error(Fault::OutOfRange, "the token " + std::to_string(id) + " is not one of the tokenizer's");
        if (sentencepiece_ ? control_[(size_t) id] : std::find(special_.begin(), special_.end(), id) != special_.end()) continue;
        if (byte_of_[(size_t) id] >= 0) {
            run += (char) byte_of_[(size_t) id];
            first = false;
            continue;
        }
        flush();
        const std::string & piece = tokens_[(size_t) id];
        if (sentencepiece_ && id == unknown_) {
            out += " \xE2\x81\x87 ";
        } else if (!first) {
            out += unescaped(piece);
        } else if (sentencepiece_) {
            out += unescaped(piece.compare(0, kSpace.size(), kSpace) == 0 ? piece.substr(kSpace.size()) : piece);
        } else {
            for (size_t j = 0; j < piece.size();) {
                if (piece.compare(j, kSpace.size(), kSpace) == 0) {
                    j += kSpace.size();
                } else {
                    out += piece[j++];
                }
            }
        }
        first = false;
    }
    flush();
    return out;
}

}  // namespace pocket
