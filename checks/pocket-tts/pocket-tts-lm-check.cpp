// Checks Pocket TTS's flow language model and its flow head against the official implementation on a dump of
// reference/pocket-tts/dump.py, in the order data flows: the voice prompt's rows from the dump's latent and the keys and
// values the model holds after it; then for each chunk of the dump, from the voice again, the rows of the chunk's tokens
// and the output after them, and each step on the latent the official model read before it (teacher forcing) with the
// official noise: its output, its logit of the end of speech, which must fall on the same side of the threshold, and
// its latent. The flow head alone on the official outputs and noise: each step's velocity and the latent.
//
// usage: pocket-tts-lm-check <model.gguf> <dump dir> [gpu|cpu|device name]

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "npy.h"
#include "pocket-tts/flow-lm.h"
#include "pocket-tts/layout.h"
#include "weight-types.h"

using namespace pocket;

namespace {

bool check(const std::string & what, const std::vector<float> & got, const std::vector<float> & want, double min_snr) {
    const Diff d = compare(got, want);
    print_diff(what, d);
    return got.size() == want.size() && d.snr_db > min_snr;
}

std::vector<float> row(const Npy & a, int64_t r) {
    const int64_t n = a.size() / a.shape[0];
    return std::vector<float>(a.f32.begin() + r * n, a.f32.begin() + (r + 1) * n);
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <dump dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    const std::string dir = args[2];
    if (!takes_weights(args[1], model_layout, {"F32"}, "its thresholds are those measured with them")) return 2;
    ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
    std::printf("backend: %s\n", ggml_backend_name(backend));
    const ModelFile model(args[1], backend, model_layout, [](const std::string & name) { return name.rfind("mimi.", 0) != 0; });
    FlowLM lm(model, backend);
    const float threshold = model.f32("pocket-tts.sampler.eos_threshold");
    // Measured on an Apple M4 Max on 2026-10-20 with the ja-short and en-long dumps, the lowest SNR: on the CPU 137.9 dB
    // for the voice prompt's rows, 112.9 dB for the model, 114.5 dB for the latents of its steps and 134.4 dB for the
    // flow head alone; on Metal, whose matrix kernel rounds its inputs to half precision, 138.1, 62.4, 71.8 and 135.1 dB,
    // the flow head's products of one column keeping float32. The tokens' rows are the table's, bit for bit. On Vulkan
    // on an RTX 5090 on 2026-10-21, 73.6, 58.2, 75.2 and 135.3 dB: Vulkan multiplies the voice prompt's latent rounded
    // to half precision, where Metal keeps float32, and rounding the latent alone on the CPU gives the same 73.6 and
    // 74.0 dB for the two dumps.
    const bool cpu = ggml_backend_is_cpu(backend);
    const double rows_db = 120, voice_db = cpu ? 120 : 60, lm_db = cpu ? 100 : 55, flow_db = cpu ? 100 : 60, head_db = 120;
    bool ok = true;

    const Npy latent = read_npy(dir + "/latent.npy"), prompt = read_npy(dir + "/voice_prompt.npy"), kv = read_npy(dir + "/voice_kv.npy");
    const int64_t frames = latent.shape[0], voice = prompt.shape[0], dim = lm.dim();
    ok = check("voice prompt rows", lm.voice_rows(latent.f32, frames), prompt.f32, voice_db) && ok;
    lm.voice(latent.f32, frames);
    std::vector<float> keys, values, want_keys, want_values;
    for (int l = 0; l < (int) kv.shape[0]; l++) {
        std::vector<float> k, v;
        lm.read_cache(l, k, v);
        keys.insert(keys.end(), k.begin(), k.end());
        values.insert(values.end(), v.begin(), v.end());
        const int64_t n = voice * dim;
        want_keys.insert(want_keys.end(), kv.f32.begin() + (2 * l) * n, kv.f32.begin() + (2 * l + 1) * n);
        want_values.insert(want_values.end(), kv.f32.begin() + (2 * l + 1) * n, kv.f32.begin() + (2 * l + 2) * n);
    }
    ok = check("voice keys", keys, want_keys, lm_db) && ok;
    ok = check("voice values", values, want_values, lm_db) && ok;

    for (int c = 0; std::filesystem::exists(std::filesystem::u8path(dir + "/chunk" + std::to_string(c) + "_tokens.npy")); c++) {
        const std::string p = dir + "/chunk" + std::to_string(c) + "_", what = "chunk " + std::to_string(c) + " ";
        std::vector<int32_t> tokens;
        for (float t : read_npy(p + "tokens.npy").f32) tokens.push_back((int32_t) t);
        lm.keep(voice);
        ok = check(what + "text rows", lm.text_rows(tokens), read_npy(p + "text_embd.npy").f32, rows_db) && ok;
        ok = check(what + "output after the text", lm.text(tokens).hidden, read_npy(p + "prefill_out.npy").f32, lm_db) && ok;

        const Npy in = read_npy(p + "lm_in.npy"), out = read_npy(p + "lm_out.npy"), eos = read_npy(p + "eos.npy"), noise = read_npy(p + "noise.npy");
        const Npy drawn = read_npy(p + "latent.npy"), velocity = read_npy(p + "flow_out.npy");
        const int64_t steps = in.shape[0];
        const int lsd = (int) velocity.shape[1];
        std::vector<float> outputs, logits, latents, alone, velocities;
        int disagree = 0;
        for (int64_t i = 0; i < steps; i++) {
            std::vector<float> previous = row(in, i);
            if (std::isnan(previous[0])) previous.clear();
            const StepOutput s = lm.step(previous, row(noise, i), lsd);
            outputs.insert(outputs.end(), s.lm.hidden.begin(), s.lm.hidden.end());
            logits.push_back(s.lm.eos);
            latents.insert(latents.end(), s.latent.begin(), s.latent.end());
            // A logit within the step's error of the threshold may fall on either side by rounding.
            if ((s.lm.eos > threshold) != (eos.f32[(size_t) i] > threshold) && std::fabs(eos.f32[(size_t) i] - threshold) > 1e-2) disagree++;
            std::vector<float> v;
            const std::vector<float> a = lm.flow(row(out, i), row(noise, i), lsd, &v);
            alone.insert(alone.end(), a.begin(), a.end());
            velocities.insert(velocities.end(), v.begin(), v.end());
        }
        ok = check(what + "outputs of " + std::to_string(steps) + " steps", outputs, out.f32, lm_db) && ok;
        ok = check(what + "end-of-speech logits", logits, eos.f32, lm_db) && ok;
        std::printf("%-34s %d of %lld\n", (what + "ends on the other side").c_str(), disagree, (long long) steps);
        ok = ok && disagree == 0;
        ok = check(what + "latents", latents, drawn.f32, flow_db) && ok;
        ok = check(what + "flow head velocities", velocities, velocity.f32, head_db) && ok;
        ok = check(what + "flow head latents", alone, drawn.f32, head_db) && ok;
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
