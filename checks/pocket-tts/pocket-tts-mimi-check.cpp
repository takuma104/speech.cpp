// Checks Mimi, the codec of Pocket TTS, against the official implementation on a dump of reference/pocket-tts/dump.py,
// each stage in the order data flows. Encoder: the first convolution, each block, SEANet, the transformer and the latent
// of the dump's voice prompt. Decoder: for each chunk of the dump, the quantizer's projection, the upsampling, the
// transformer, SEANet's first convolution, its blocks and the samples of the latents the official decoder read, decoded
// a frame at a time as the dump decoded them. Decoding a few frames at a time: each chunk's latents in calls of 1, 2, 3,
// 5 and 8 frames and at once, against a frame at a time.
//
// usage: pocket-tts-mimi-check <model.gguf> <dump dir> [gpu|cpu|device name]

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "npy.h"
#include "pocket-tts/layout.h"
#include "pocket-tts/mimi.h"
#include "weight-types.h"

using namespace pocket;

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/** A dump of [channels, time] in ggml's order: [time][channels], the channel fastest. */
std::vector<float> channel_first(const Npy & a) {
    const size_t rows = (size_t) a.shape[0], cols = (size_t) a.shape[1];
    std::vector<float> out(a.f32.size());
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++) out[c * rows + r] = a.f32[r * cols + c];
    return out;
}

/** Compares `got` with the dump `name`, prints it, and says whether it has the dump's size and passes `min_snr`. */
bool check(const std::string & what, const std::vector<float> & got, const std::vector<float> & want, double min_snr) {
    const Diff d = compare(got, want);
    print_diff(what, d);
    return got.size() == want.size() && d.snr_db > min_snr;
}

/** The latents of `latent`, row-major [frames, dim], from frame `first` on, `n` of them. */
std::vector<float> frames_of(const std::vector<float> & latent, int dim, int64_t first, int64_t n) {
    return std::vector<float>(latent.begin() + first * dim, latent.begin() + (first + n) * dim);
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
    const ModelFile model(args[1], backend, model_layout, [](const std::string & name) { return name.rfind("mimi.", 0) == 0 || name == "lm.bos"; });
    Mimi mimi(model, backend);
    // Measured on an Apple M4 Max on 2026-10-20 with the ja-short and en-long dumps, the lowest SNR: on the CPU 102.9 dB
    // for the encoder (its transformer), 104.9 dB for the decoder and 129.9 dB for calls of other sizes against a frame
    // at a time; on Metal, whose matrix kernel rounds its inputs to half precision, 51.1, 66.8 and 61.5 dB. Metal
    // multiplies the 16 columns of one frame at the transformer's rate with its few-row kernels, whose sums run in
    // another order than its kernel for more columns, so a frame at a time departs from calls of 2 to 8 frames by the
    // same 61.5 to 63.9 dB.
    const bool cpu = ggml_backend_is_cpu(backend);
    const double encoder_db = cpu ? 95 : 45, decoder_db = cpu ? 95 : 60, windows_db = cpu ? 120 : 55;
    bool ok = true;

    const Npy prompt = read_npy(dir + "/ref_prompt.npy");
    MimiStages stages;
    auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> latent = mimi.encode(prompt.f32, &stages);
    std::printf("encoding %.2f s of audio: %.3f s\n", (double) prompt.f32.size() / mimi.sample_rate(), seconds_since(t0));
    ok = check("encoder conv_in", stages.at("conv_in"), channel_first(read_npy(dir + "/enc_conv_in.npy")), encoder_db) && ok;
    for (int b = 0; std::filesystem::exists(std::filesystem::u8path(dir + "/enc_block" + std::to_string(b) + ".npy")); b++) {
        const std::string name = "block" + std::to_string(b);
        ok = check("encoder " + name, stages.at(name), channel_first(read_npy(dir + "/enc_" + name + ".npy")), encoder_db) && ok;
    }
    ok = check("encoder seanet", stages.at("seanet"), channel_first(read_npy(dir + "/enc_seanet.npy")), encoder_db) && ok;
    ok = check("encoder transformer", stages.at("transformer"), channel_first(read_npy(dir + "/enc_transformer.npy")), encoder_db) && ok;
    ok = check("latent", latent, read_npy(dir + "/latent.npy").f32, encoder_db) && ok;

    const int dim = mimi.latent_dim();
    for (int c = 0; std::filesystem::exists(std::filesystem::u8path(dir + "/chunk" + std::to_string(c) + "_dec_latent.npy")); c++) {
        const std::string p = dir + "/chunk" + std::to_string(c) + "_";
        const Npy dec_latent = read_npy(p + "dec_latent.npy");
        const int64_t frames = dec_latent.shape[0];
        std::map<std::string, std::vector<float>> joined;
        std::vector<float> audio;
        mimi.start_decoding();
        t0 = std::chrono::steady_clock::now();
        for (int64_t f = 0; f < frames; f++) {
            MimiStages s;
            const std::vector<float> a = mimi.decode(frames_of(dec_latent.f32, dim, f, 1), 1, &s);
            audio.insert(audio.end(), a.begin(), a.end());
            for (const auto & [name, values] : s) joined[name].insert(joined[name].end(), values.begin(), values.end());
        }
        std::printf("chunk %d: decoding %lld frames a frame at a time: %.3f s\n", c, (long long) frames, seconds_since(t0));
        // A stage of a frame at a time is the frame's columns one after another, which is the order of the whole.
        const std::string what = "chunk " + std::to_string(c) + " ";
        ok = check(what + "quant", joined.at("quant"), channel_first(read_npy(p + "dec_quant.npy")), decoder_db) && ok;
        ok = check(what + "up", joined.at("up"), channel_first(read_npy(p + "dec_up.npy")), decoder_db) && ok;
        ok = check(what + "transformer", joined.at("transformer"), channel_first(read_npy(p + "dec_transformer.npy")), decoder_db) && ok;
        ok = check(what + "conv_in", joined.at("conv_in"), channel_first(read_npy(p + "dec_conv_in.npy")), decoder_db) && ok;
        for (int b = 0; joined.count("block" + std::to_string(b)); b++) {
            const std::string name = "block" + std::to_string(b);
            ok = check(what + name, joined.at(name), channel_first(read_npy(p + "dec_" + name + ".npy")), decoder_db) && ok;
        }
        ok = check(what + "samples", audio, read_npy(p + "dec_out.npy").f32, decoder_db) && ok;

        for (int64_t size : {2, 3, 5, 8, 0}) {
            const int64_t step = size ? size : frames;
            std::vector<float> windowed;
            mimi.start_decoding();
            for (int64_t f = 0; f < frames; f += step) {
                const int64_t n = std::min(step, frames - f);
                const std::vector<float> a = mimi.decode(frames_of(dec_latent.f32, dim, f, n), n);
                windowed.insert(windowed.end(), a.begin(), a.end());
            }
            const std::string calls = size ? "in calls of " + std::to_string(size) : "at once";
            ok = check(what + calls + " against 1", windowed, audio, windows_db) && ok;
        }
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
