// Checks Pocket TTS's synthesis of a whole request against the official generate_audio() on a dump of
// reference/pocket-tts/dump.py: the voice prompt ended on a pause from the dump's resampled reference; with the dump's
// reference recording, also its reading, mixed to mono, resampled with the library's filter and ended on a pause; and
// the request's text spoken in the dump's voice latent with the official noise of every step, which must stop at the
// same frames as the official speech, its samples compared with the official ones.
//
// usage: pocket-tts-synthesis-check <model.gguf> <dump dir> [reference.wav] [gpu|cpu|device name]

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "json-reader.h"
#include "npy.h"
#include "pocket-tts/layout.h"
#include "pocket-tts/synthesizer.h"
#include "weight-types.h"

using namespace pocket;

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <dump dir> [reference.wav] [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    const std::string dir = args[2];
    const auto is_wav = [](const std::string & a) { return a.size() > 4 && a.compare(a.size() - 4, 4, ".wav") == 0; };
    const std::string reference = args.size() > 3 && is_wav(args[3]) ? args[3] : "";
    const size_t device_arg = reference.empty() ? 3 : 4;
    if (!takes_weights(args[1], model_layout, {"F32"}, "its thresholds are those measured with them")) return 2;
    ggml_backend_t backend = init_backend(args.size() > device_arg ? args[device_arg] : "");
    std::printf("backend: %s\n", ggml_backend_name(backend));
    std::ifstream f(std::filesystem::u8path(dir + "/meta.json"), std::ios::binary);
    const JsonValue meta = parse_json(std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()));
    const bool cpu = ggml_backend_is_cpu(backend);
    bool ok = true;

    Synthesizer synth(args[1], backend);
    const ModelFile rules_file(args[1], model_layout);
    const ReferenceRules rules(rules_file);
    const Npy resampled = read_npy(dir + "/ref_resampled.npy"), prompt = read_npy(dir + "/ref_prompt.npy");
    const std::vector<float> ended = end_on_pause(resampled.f32, synth.sample_rate(), rules);
    const Diff de = compare(ended, prompt.f32);
    print_diff("ended on a pause", de);
    // The cut, the fade and the pause are the official ones but for the rounding of a frame's loudness in float32.
    ok = ok && ended.size() == prompt.f32.size() && de.snr_db > 120;
    if (!reference.empty()) {
        const Reading reading = read_reference(reference, synth.sample_rate(), rules, "reference");
        const Diff dr = compare(reading.audio, prompt.f32);
        print_diff("reading the WAVE file", dr);
        // At the model's rate the reading is the official one; at another the library's kaiser_best filter is not
        // scipy's resample_poly(), and the reading only has to be the official one's length.
        const bool resampled_here = std::stoi(meta.member("reference")->member("sample_rate")->text) != synth.sample_rate();
        ok = ok && reading.audio.size() == prompt.f32.size() && (resampled_here || dr.snr_db > 120);
    }

    Request r;
    r.text = meta.member("text")->text;
    r.steps = std::stoi(meta.member("steps")->text);
    int64_t frames = 0;
    for (size_t c = 0; c < meta.member("chunks")->items.size(); c++) {
        const Npy noise = read_npy(dir + "/chunk" + std::to_string(c) + "_noise.npy");
        r.noise.insert(r.noise.end(), noise.f32.begin(), noise.f32.end());
        frames += std::stoll(meta.member("chunks")->items[c].member("frames")->text);
    }
    const Npy latent = read_npy(dir + "/latent.npy");
    const Voice voice{latent.f32, latent.shape[0]};
    std::vector<float> audio;
    const Ending ending = synth.synthesize(r, voice, [&](const float * s, size_t n) {
        audio.insert(audio.end(), s, s + n);
        return true;
    });
    const std::vector<float> official = read_npy(dir + "/audio.npy").f32;
    std::printf("%-34s %lld frames, the official %lld\n", "speech", (long long) (audio.size() / synth.hop()), (long long) frames);
    const Diff da = compare(audio, official);
    print_diff("samples", da);
    // Measured on an Apple M4 Max on 2026-10-20 with the ja-short and en-long dumps, every speech of the official
    // frames: on the CPU 62.0 and 78.5 dB, on Metal 27.1 and 25.6 dB; on Vulkan on an RTX 5090 on 2026-10-21, 26.2 and
    // 21.8 dB. Each step reads the latent of the step before, so the error of every step, 112.9 dB or more on the CPU,
    // 62.4 dB or more on Metal and 58.2 dB or more on Vulkan from the official inputs (pocket-tts-lm-check), feeds into
    // the next.
    ok = ok && audio.size() == official.size() && da.snr_db > (cpu ? 50 : 20);
    ok = ok && (ending == Ending::Complete || ending == Ending::ModelLimit);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
