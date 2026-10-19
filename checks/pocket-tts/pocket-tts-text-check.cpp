// Checks Pocket TTS's tokenizer and its cutting of a text into chunks against the official code, on the cases of
// reference/pocket-tts/text_cases.py: for each text, the ids the tokenizer gives it and the text it decodes from them,
// and the chunks the official generate_audio() speaks, each as the sentence splitter cuts it, as it is prepared, with
// the frames spoken after the end of speech and its tokens; or, for a text the official code refuses, a refusal. Every
// value must be the official one exactly.
//
// usage: pocket-tts-text-check <model.gguf> <text-cases.jsonl>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "args.h"
#include "error.h"
#include "json-reader.h"
#include "pocket-tts/layout.h"
#include "pocket-tts/text.h"

using namespace pocket;

namespace {

std::vector<int32_t> ids_of(const JsonValue & v) {
    std::vector<int32_t> out;
    for (const JsonValue & item : v.items) out.push_back((int32_t) std::stol(item.text));
    return out;
}

std::string shown(const std::vector<int32_t> & ids) {
    std::string out;
    for (size_t i = 0; i < ids.size() && i < 12; i++) out += (i ? " " : "") + std::to_string(ids[i]);
    return out + (ids.size() > 12 ? " ..." : "");
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <text-cases.jsonl>\n", args[0].c_str());
        return 2;
    }
    const ModelFile model(args[1], model_layout);
    const Text text(model);
    std::ifstream f(std::filesystem::u8path(args[2]), std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", args[2].c_str());
        return 2;
    }
    int cases = 0, failed = 0, chunks = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const JsonValue c = parse_json(line);
        const std::string input = c.member("text")->text;
        cases++;
        std::vector<std::string> wrong;
        const std::vector<int32_t> ids = text.tokenizer().encode(input), want_ids = ids_of(*c.member("ids"));
        if (ids != want_ids) wrong.push_back("ids " + shown(ids) + " where the official ones are " + shown(want_ids));
        const std::string decoded = text.tokenizer().decode(want_ids);
        if (decoded != c.member("decoded")->text) wrong.push_back("decodes to \"" + decoded + "\" where the official decodes \"" + c.member("decoded")->text + "\"");
        if (const JsonValue * error = c.member("error")) {
            try {
                text.chunks(input);
                wrong.push_back("cut into chunks where the official code refuses it: " + error->text);
            } catch (const Error & e) {
                if (e.fault() != Fault::InvalidArgument) wrong.push_back(std::string("refused with another kind of error: ") + e.what());
            }
        } else {
            const std::vector<Chunk> got = text.chunks(input);
            const JsonValue & want = *c.member("chunks");
            chunks += (int) want.items.size();
            if (got.size() != want.items.size()) wrong.push_back(std::to_string(got.size()) + " chunks where the official code cuts " + std::to_string(want.items.size()));
            for (size_t i = 0; i < got.size() && i < want.items.size(); i++) {
                const JsonValue & w = want.items[i];
                const std::string at = "chunk " + std::to_string(i) + " ";
                if (got[i].text != w.member("text")->text) wrong.push_back(at + "\"" + got[i].text + "\" where the official is \"" + w.member("text")->text + "\"");
                if (got[i].prepared != w.member("prepared")->text) wrong.push_back(at + "prepared as \"" + got[i].prepared + "\" where the official is \"" + w.member("prepared")->text + "\"");
                if (got[i].frames_after_eos != std::stoi(w.member("frames_after_eos")->text)) wrong.push_back(at + "speaks another number of frames after the end");
                if (got[i].tokens != ids_of(*w.member("tokens"))) wrong.push_back(at + "tokens " + shown(got[i].tokens) + " where the official ones are " + shown(ids_of(*w.member("tokens"))));
            }
        }
        if (!wrong.empty()) {
            failed++;
            std::printf("\"%s\":\n", input.c_str());
            for (const std::string & w : wrong) std::printf("  %s\n", w.c_str());
        }
    }
    std::printf("%d of %d texts as the official code gives them, %d chunks\n", cases - failed, cases, chunks);
    return failed == 0 && cases > 0 ? 0 : 1;
}
