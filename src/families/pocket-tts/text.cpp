#include "text.h"

#include <algorithm>

#include "error.h"
#include "unicode.h"
#include "utf8.h"

namespace pocket {

namespace {

struct CodepointRange {
    uint32_t first, last;
};

/** str.upper() of a character it changes, in UTF-8. */
struct UpperMapping {
    uint32_t cp;
    const char * upper;
};

#include "unicode.inc"

template <size_t N>
bool in_table(const CodepointRange (&table)[N], uint32_t cp) {
    const auto it = std::upper_bound(std::begin(table), std::end(table), cp, [](uint32_t c, const CodepointRange & r) { return c < r.first; });
    return it != std::begin(table) && cp <= (it - 1)->last;
}

using Text32 = std::vector<uint32_t>;

/** U+2026, the ellipsis, which ends a sentence as a period does. */
constexpr uint32_t kEllipsis = 0x2026;

Text32 decoded(const std::string & s) {
    std::optional<Text32> cps = decode_utf8(s);
    if (!cps) throw Error(Fault::InvalidArgument, "the text is not UTF-8; give it as UTF-8", "text");
    return *cps;
}

bool contains(const Text32 & set, uint32_t c) {
    return std::find(set.begin(), set.end(), c) != set.end();
}

/** str.strip(), or str.rstrip(chars) where `chars` is given. */
Text32 strip(const Text32 & t, bool left = true, const Text32 * chars = nullptr) {
    const auto stripped = [&](uint32_t c) { return chars ? contains(*chars, c) : python_space(c); };
    size_t a = 0, b = t.size();
    while (left && a < b && stripped(t[a])) a++;
    while (b > a && stripped(t[b - 1])) b--;
    return Text32(t.begin() + (ptrdiff_t) a, t.begin() + (ptrdiff_t) b);
}

/** str.split(): the runs of characters between whitespace. */
std::vector<Text32> words(const Text32 & t) {
    std::vector<Text32> out;
    for (size_t i = 0; i < t.size();) {
        while (i < t.size() && python_space(t[i])) i++;
        if (i == t.size()) break;
        const size_t start = i;
        while (i < t.size() && !python_space(t[i])) i++;
        out.emplace_back(t.begin() + (ptrdiff_t) start, t.begin() + (ptrdiff_t) i);
    }
    return out;
}

/** str.replace(from, to), each occurrence from the left once. */
Text32 replaced(const Text32 & t, const Text32 & from, const Text32 & to) {
    Text32 out;
    for (size_t i = 0; i < t.size();) {
        if (i + from.size() <= t.size() && std::equal(from.begin(), from.end(), t.begin() + (ptrdiff_t) i)) {
            out.insert(out.end(), to.begin(), to.end());
            i += from.size();
        } else {
            out.push_back(t[i++]);
        }
    }
    return out;
}

Text32 upper(uint32_t c) {
    const auto it = std::lower_bound(std::begin(kUpperMappings), std::end(kUpperMappings), c, [](const UpperMapping & m, uint32_t v) { return m.cp < v; });
    if (it == std::end(kUpperMappings) || it->cp != c) return {c};
    return *decode_utf8(it->upper);
}

/**
 * _ensure_terminal_punctuation(): a text that does not end, closing quotes and brackets aside, with a sentence's
 * punctuation gets a period, in place of a trailing comma, colon or dash, and after the closers otherwise.
 */
Text32 terminated(const Text32 & text) {
    static const Text32 terminal = {'.', '!', '?', kEllipsis}, weak = {',', ';', ':', '-', 0x2013, 0x2014};
    static const Text32 closers = {'"', '\'', 0x201D, 0x2019, ')', ']', 0xBB};
    Text32 closers_space = closers;
    closers_space.push_back(' ');
    const Text32 core = strip(text, false, &closers_space);
    const Text32 tail = strip(Text32(text.begin() + (ptrdiff_t) core.size(), text.end()));
    if (core.empty() || contains(terminal, core.back())) return text;
    if (contains(weak, core.back())) {
        Text32 weak_space = weak;
        weak_space.push_back(' ');
        Text32 out = strip(core, false, &weak_space);
        out.push_back('.');
        out.insert(out.end(), tail.begin(), tail.end());
        return out;
    }
    Text32 out = text;
    out.push_back('.');
    return out;
}

}  // namespace

Text::Text(const ModelFile & m) : tokenizer_(m) {
    const std::string p = "pocket-tts.";
    max_tokens_ = m.size(p + "text.max_tokens");
    pad_ = m.boolean(p + "text.pad_with_spaces");
    remove_semicolons_ = m.boolean(p + "text.remove_semicolons");
    terminal_ = m.boolean(p + "text.append_terminal_punctuation");
    capitalize_ = m.boolean(p + "text.capitalize_first_letter");
    short_words_ = (int) m.u32(p + "generation.short_text_words");
    frames_short_ = (int) m.u32(p + "generation.frames_after_eos_short");
    frames_long_ = (int) m.u32(p + "generation.frames_after_eos_long");
    if (m.has(p + "text.replace_from")) {
        const std::vector<std::string> from = m.str_array(p + "text.replace_from"), to = m.str_array(p + "text.replace_to");
        for (size_t i = 0; i < from.size(); i++) replace_.push_back({decoded(from[i]).at(0), decoded(to[i])});
    }
}

std::pair<std::string, int> Text::prepare(const std::string & input) const {
    Text32 text = strip(decoded(input));
    if (!replace_.empty()) {
        // " ".join(text.translate(str.maketrans(replace_characters)).split())
        Text32 translated;
        for (uint32_t c : text) {
            const auto it = std::find_if(replace_.begin(), replace_.end(), [&](const auto & r) { return r.first == c; });
            if (it == replace_.end()) {
                translated.push_back(c);
            } else {
                translated.insert(translated.end(), it->second.begin(), it->second.end());
            }
        }
        Text32 joined;
        for (const Text32 & w : words(translated)) {
            if (!joined.empty()) joined.push_back(' ');
            joined.insert(joined.end(), w.begin(), w.end());
        }
        // re.sub(r"([.!?…])\s*[,;:]", r"\1", text): a sentence's mark keeps nothing of a weak mark after it.
        static const Text32 marks = {'.', '!', '?', kEllipsis}, weak = {',', ';', ':'};
        text.clear();
        for (size_t i = 0; i < joined.size();) {
            text.push_back(joined[i]);
            size_t j = i + 1;
            if (contains(marks, joined[i])) {
                while (j < joined.size() && python_space(joined[j])) j++;
                if (j < joined.size() && contains(weak, joined[j])) {
                    i = j + 1;
                    continue;
                }
            }
            i++;
        }
    }
    if (text.empty()) throw Error(Fault::InvalidArgument, "the text is empty once its spaces and the characters the model deletes are gone; give a text to speak", "text");
    text = replaced(replaced(replaced(text, {'\n'}, {' '}), {'\r'}, {' '}), {' ', ' '}, {' '});
    if (remove_semicolons_) text = replaced(text, {';'}, {','});
    const int word_count = (int) words(text).size();
    if (capitalize_ && !in_table(kUppercase, text[0])) {
        Text32 capital = upper(text[0]);
        capital.insert(capital.end(), text.begin() + 1, text.end());
        text = capital;
    }
    if (terminal_) text = terminated(text);
    if (pad_ && words(text).size() < 5) text.insert(text.begin(), 8, ' ');
    return {encode_utf8(text), word_count};
}

std::vector<size_t> Text::boundaries(const std::vector<int32_t> & tokens, const std::vector<int32_t> & boundary, bool skip_decimals) const {
    std::vector<size_t> indices = {0};
    bool previous = false;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (std::find(boundary.begin(), boundary.end(), tokens[i]) != boundary.end()) {
            previous = true;
            continue;
        }
        if (previous && skip_decimals) {
            // _is_decimal_period_boundary(): a digit, a period, then a digit, as the tokens before and after decode.
            const Text32 before = decoded(tokenizer_.decode(tokens.data(), i)), after = decoded(tokenizer_.decode(tokens.data() + i, tokens.size() - i));
            if (before.size() >= 2 && before.back() == '.' && in_table(kDigits, before[before.size() - 2]) && !after.empty() && in_table(kDigits, after[0])) {
                previous = false;
                continue;
            }
        }
        if (previous) indices.push_back(i);
        previous = false;
    }
    indices.push_back(tokens.size());
    return indices;
}

std::vector<std::string> Text::split(const std::string & input) const {
    const std::string text = encode_utf8(strip(decoded(prepare(input).first)));
    const std::vector<int32_t> tokens = tokenizer_.encode(text);
    // The official code takes the tokens of ".!...?" and ",;:" from its tokenizer, the first, the space in front, left out.
    std::vector<int32_t> ends = tokenizer_.encode(".!...?"), commas = tokenizer_.encode(",;:");
    ends.erase(ends.begin());
    commas.erase(commas.begin());
    const auto segments = [&](const std::vector<int32_t> & t, const std::vector<size_t> & at) {
        std::vector<std::pair<size_t, std::string>> out;
        for (size_t i = 0; i + 1 < at.size(); i++) out.push_back({at[i + 1] - at[i], tokenizer_.decode(t.data() + at[i], at[i + 1] - at[i])});
        return out;
    };
    std::vector<std::pair<size_t, std::string>> refined;
    for (const auto & [n, sentence] : segments(tokens, boundaries(tokens, ends, true))) {
        if ((int) n <= max_tokens_) {
            refined.push_back({n, sentence});
            continue;
        }
        const std::vector<int32_t> sub = tokenizer_.encode(encode_utf8(strip(decoded(sentence))));
        const auto parts = segments(sub, boundaries(sub, commas, false));
        if (parts.size() > 1) {
            refined.insert(refined.end(), parts.begin(), parts.end());
        } else {
            refined.push_back({n, sentence});
        }
    }
    std::vector<std::string> chunks;
    std::string current;
    size_t count = 0;
    for (const auto & [n, sentence] : refined) {
        if (current.empty()) {
            current = sentence;
            count = n;
        } else if (count + n > (size_t) max_tokens_) {
            chunks.push_back(encode_utf8(strip(decoded(current))));
            current = sentence;
            count = n;
        } else {
            current += " " + sentence;
            count += n;
        }
    }
    if (!current.empty()) chunks.push_back(encode_utf8(strip(decoded(current))));
    return chunks;
}

std::vector<Chunk> Text::chunks(const std::string & text) const {
    std::vector<Chunk> out;
    for (const std::string & piece : split(text)) {
        const auto [prepared, word_count] = prepare(piece);
        out.push_back({piece, prepared, tokenizer_.encode(prepared), word_count <= short_words_ ? frames_short_ : frames_long_});
    }
    return out;
}

}  // namespace pocket
