#include "layout.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "error.h"
#include "utf8.h"

namespace pocket {

namespace {

/**
 * How the layout stores a tensor in a file of each weight type. A matrix or the lookup table of the flow language
 * model's transformer, which ggml_mul_mat() and ggml_get_rows() alone read, takes the file's type. The flow head, which
 * draws each latent, and Mimi's convolutions and matrices take F16, or F32 in an F32 file, as the codecs of the other
 * families do. The norms, the biases, the layer scales, the latent's projections and statistics, the end-of-speech
 * head and the convolution of the waveform take F32.
 */
const Storage kMatrix = quantized_storage({{GGML_TYPE_F32, "0.8.12"},
                                           {GGML_TYPE_F16, "0.8.12"},
                                           {GGML_TYPE_Q8_0, "0.8.12"},
                                           {GGML_TYPE_Q6_K, "0.8.12"},
                                           {GGML_TYPE_Q5_K, "0.8.12"},
                                           {GGML_TYPE_Q4_K, "0.8.12"}});
const Storage kHalf = half_storage();
const Storage kF32 = float32_storage();

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

/** A float key that must be finite and at least `least`. */
float bounded(const ModelFile & m, const std::string & key, float least) {
    const float v = m.f32(key);
    require(std::isfinite(v) && v >= least, m, key + " is not a finite number of at least " + std::to_string(least));
    return v;
}

/** The same of a float64 key, a Python float of the official code. */
double bounded64(const ModelFile & m, const std::string & key, double least) {
    const double v = m.f64(key);
    require(std::isfinite(v) && v >= least, m, key + " is not a finite number of at least " + std::to_string(least));
    return v;
}

/** A transformer's layers of width `dim` and feed-forward width `ffn`, with layer scales where `scaled`. */
void add_transformer(std::vector<TensorSpec> & t, const std::string & prefix, int layers, int64_t dim, int64_t ffn, const Storage & matrix,
                     bool scaled) {
    add_numbered(t, prefix + ".blk.", layers,
                 {{"attn_norm.weight", {dim}, kF32}, {"attn_norm.bias", {dim}, kF32},   {"attn_qkv", {dim, 3 * dim}, matrix},
                  {"attn_out", {dim, dim}, matrix},  {"ffn_norm.weight", {dim}, kF32},  {"ffn_norm.bias", {dim}, kF32},
                  {"ffn_up", {dim, ffn}, matrix},    {"ffn_down", {ffn, dim}, matrix}});
    if (scaled) add_numbered(t, prefix + ".blk.", layers, {{"attn_scale", {dim}, kF32}, {"ffn_scale", {dim}, kF32}});
}

/**
 * The tokenizer's keys: its runtime, pieces, scores and unknown piece, the added tokens tokenizers matches in a text and
 * the control pieces of sentencepiece.
 */
void read_tokenizer(const ModelFile & m) {
    const std::string p = "pocket-tts.tokenizer.";
    const std::string runtime = m.one_of(p + "runtime", {"tokenizers", "sentencepiece"});
    const std::vector<std::string> tokens = m.str_array(p + "tokens");
    require(!tokens.empty() && m.f64_array(p + "scores").size() == tokens.size(), m, p + "scores does not have one score per piece");
    require(m.u32(p + "unknown_id") < tokens.size(), m, p + "unknown_id is not one of the pieces");
    for (int b = 0; b < 256; b++) {
        char name[8];
        std::snprintf(name, sizeof name, "<0x%02X>", b);
        require(std::count(tokens.begin(), tokens.end(), name) == 1, m, "the tokenizer has no single piece " + std::string(name) + " for its byte fallback");
    }
    require(!m.has(p + "special_ids") || runtime == "tokenizers", m, "only a tokenizer that tokenizers runs has " + p + "special_ids");
    require(!m.has(p + "control_ids") || runtime == "sentencepiece", m, "only a tokenizer that sentencepiece runs has " + p + "control_ids");
    for (const char * key : {"special_ids", "control_ids"}) {
        if (!m.has(p + key)) continue;
        for (int32_t id : m.i32_array(p + key)) require(id >= 0 && (size_t) id < tokens.size(), m, p + key + " holds an id past the pieces");
    }
}

/** The rules of a text: the chunk's tokens, the flags, and the characters replaced, each one character. */
void read_text(const ModelFile & m) {
    const std::string p = "pocket-tts.text.";
    m.size(p + "max_tokens");
    for (const char * flag : {"pad_with_spaces", "remove_semicolons", "append_terminal_punctuation", "capitalize_first_letter"}) m.boolean(p + flag);
    require(m.has(p + "replace_from") == m.has(p + "replace_to"), m, p + "replace_from and " + p + "replace_to come together");
    if (m.has(p + "replace_from")) {
        const std::vector<std::string> from = m.str_array(p + "replace_from");
        require(m.str_array(p + "replace_to").size() == from.size(), m, p + "replace_to does not have one string per character replaced");
        for (const std::string & c : from) {
            const std::optional<std::vector<uint32_t>> points = decode_utf8(c);
            require(points && points->size() == 1, m, p + "replace_from holds \"" + c + "\", which is not one character");
        }
    }
}

/** Reads every key of the layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<TensorSpec> tensors(const ModelFile & m) {
    check_model_keys(m, "synthesis", "checked");
    const std::string p = "pocket-tts.";
    m.one_of(p + "flow", {"lsd"});

    const int lm_layers = m.count(p + "lm.num_layers"), lm_heads = m.size(p + "lm.num_heads");
    bounded(m, p + "lm.rope_base", 1);
    bounded(m, p + "lm.norm_eps", 0);
    const bool bos_before_voice = m.boolean(p + "lm.bos_before_voice");
    const int flow_blocks = m.count(p + "flow.num_blocks");
    bounded(m, p + "flow.norm_eps", 0);
    bounded(m, p + "flow.time_norm_eps", 0);

    bounded64(m, p + "sampler.temperature", 0);
    m.size(p + "sampler.default_steps");
    require(std::isfinite(m.f32(p + "sampler.eos_threshold")), m, p + "sampler.eos_threshold is not finite");
    m.u32(p + "generation.min_frames_before_eos");
    require(bounded64(m, p + "generation.tokens_per_second", 0) > 0, m, p + "generation.tokens_per_second is 0");
    bounded64(m, p + "generation.padding_seconds", 0);
    for (const char * key : {"short_text_words", "frames_after_eos_short", "frames_after_eos_long", "fade_in_samples"}) m.u32(p + "generation." + key);
    require(bounded64(m, p + "reference.max_seconds", 0) > 0, m, p + "reference.max_seconds is 0");
    require(bounded64(m, p + "reference.frame_seconds", 0) > 0, m, p + "reference.frame_seconds is 0");
    for (const char * key : {"pause_seconds", "fade_seconds", "floor_db"}) bounded64(m, p + "reference." + key, 0);
    read_text(m);
    read_tokenizer(m);

    const std::string c = p + "mimi.";
    const int64_t hop = m.size(c + "hop_length");
    const std::vector<int32_t> ratios = m.i32_array(c + "ratios");
    const int stride = m.size(c + "downsample_stride");
    require(!ratios.empty() && (int64_t) ratios.size() <= m.tensor_count() && std::all_of(ratios.begin(), ratios.end(), [](int32_t r) { return r >= 1; }), m,
            c + "ratios is empty, holds a stride below 1 or names more blocks than the file holds tensors");
    int64_t samples = stride;
    for (int32_t r : ratios) samples = std::min<int64_t>(samples * r, (int64_t) INT_MAX + 1);
    require(samples == hop, m, c + "ratios and " + c + "downsample_stride do not multiply to " + c + "hop_length");
    const int residual_layers = m.count(c + "residual_layers");
    m.size(c + "dilation_base");
    const int mimi_layers = m.count(c + "num_layers"), mimi_heads = m.size(c + "num_heads");
    m.size(c + "context");
    bounded(m, c + "rope_base", 1);
    bounded(m, c + "norm_eps", 0);
    const std::string sha256 = m.str(c + "encoder_sha256");
    require(sha256.size() == 64 && std::all_of(sha256.begin(), sha256.end(), [](char x) { return std::isxdigit((unsigned char) x); }), m,
            c + "encoder_sha256 is not a SHA-256 in hexadecimal");

    // The widths the checkpoint's configuration gives and no key holds: each is the width of one tensor, which the
    // others are checked against.
    const int64_t dim = m.width("lm.text_embd", 0), rows = m.width("lm.text_embd", 1);
    const int64_t latent = m.width("lm.bos", 0), ffn = m.width("lm.blk.0.ffn_up", 1);
    const int64_t flow = m.width("flow.in_proj.weight", 1), half_frequencies = m.width("flow.time.0.freqs", 0);
    require(rows >= (int64_t) m.str_array(p + "tokenizer.tokens").size(), m, "lm.text_embd has fewer rows than the tokenizer has pieces");
    require(dim % lm_heads == 0 && (dim / lm_heads) % 2 == 0, m, p + "lm.num_heads does not divide the width into heads of an even width, which RoPE turns in pairs");
    const int64_t channels = m.width("mimi.quant_out", 1), filters = m.width("mimi.enc.conv_in.weight", 1);
    const int64_t kernel = m.width("mimi.enc.conv_in.weight", 2), last_kernel = m.width("mimi.enc.conv_out.weight", 2);
    const int64_t residual_kernel = m.width("mimi.enc.blk.0.res.0.conv1.weight", 2), mimi_ffn = m.width("mimi.dec_tf.blk.0.ffn_up", 1);
    const int64_t hidden = m.width("mimi.enc.blk.0.res.0.conv1.weight", 1);
    require(filters % hidden == 0, m, "the residual units of Mimi do not narrow its channels by a whole factor");
    const int64_t compress = filters / hidden;
    require(channels % mimi_heads == 0 && (channels / mimi_heads) % 2 == 0, m, c + "num_heads does not divide Mimi's width into heads of an even width");
    const int64_t widest = filters << ratios.size();
    require(widest % compress == 0 && widest <= (int64_t) INT_MAX, m, "Mimi's blocks widen its channels past what the residual units narrow");

    std::vector<TensorSpec> t = {{"lm.text_embd", {dim, rows}, kMatrix}, {"lm.bos", {latent}, kF32},
                                 {"lm.emb_mean", {latent}, kF32},         {"lm.emb_std", {latent}, kF32},
                                 {"lm.in_proj", {latent, dim}, kF32},     {"lm.speaker_proj", {latent, dim}, kF32},
                                 {"lm.out_norm.weight", {dim}, kF32},     {"lm.out_norm.bias", {dim}, kF32},
                                 {"lm.eos.weight", {dim}, kF32},          {"lm.eos.bias", {1}, kF32}};
    if (bos_before_voice) t.push_back({"lm.bos_before_voice", {dim}, kF32});
    add_transformer(t, "lm", lm_layers, dim, ffn, kMatrix, false);

    add_block(t, "flow.", {{"cond.weight", {dim, flow}, kHalf},          {"cond.bias", {flow}, kF32},
                           {"in_proj.weight", {latent, flow}, kHalf},    {"in_proj.bias", {flow}, kF32},
                           {"final.ada.weight", {flow, 2 * flow}, kHalf}, {"final.ada.bias", {2 * flow}, kF32},
                           {"final.out.weight", {flow, latent}, kHalf},  {"final.out.bias", {latent}, kF32}});
    // LSD conditions the head on the start and the target time of a step, each through an embedder of its own.
    add_numbered(t, "flow.time.", 2,
                 {{"freqs", {half_frequencies}, kF32},                {"fc1.weight", {2 * half_frequencies, flow}, kHalf},
                  {"fc1.bias", {flow}, kF32},                        {"fc2.weight", {flow, flow}, kHalf},
                  {"fc2.bias", {flow}, kF32},                        {"norm", {flow}, kF32}});
    add_numbered(t, "flow.blk.", flow_blocks,
                 {{"norm.weight", {flow}, kF32}, {"norm.bias", {flow}, kF32},     {"fc1.weight", {flow, flow}, kHalf},
                  {"fc1.bias", {flow}, kF32},    {"fc2.weight", {flow, flow}, kHalf}, {"fc2.bias", {flow}, kF32},
                  {"ada.weight", {flow, 3 * flow}, kHalf}, {"ada.bias", {3 * flow}, kF32}});

    const auto residuals = [&](const std::string & prefix, int64_t width) {
        add_numbered(t, prefix + ".res.", residual_layers,
                     {{"conv1.weight", {width, width / compress, residual_kernel}, kHalf}, {"conv1.bias", {width / compress}, kF32},
                      {"conv2.weight", {width / compress, width, 1}, kHalf},             {"conv2.bias", {width}, kF32}});
    };
    add_block(t, "mimi.enc.conv_in.", {{"weight", {1, filters, kernel}, kF32}, {"bias", {filters}, kF32}});
    for (size_t i = 0; i < ratios.size(); i++) {
        const int64_t width = filters << i, r = ratios[ratios.size() - 1 - i];
        const std::string b = "mimi.enc.blk." + std::to_string(i);
        residuals(b, width);
        add_block(t, b + ".down.", {{"first", {r * width, 2 * width}, kHalf}, {"second", {r * width, 2 * width}, kHalf}, {"bias", {2 * width}, kF32}});
    }
    add_block(t, "mimi.enc.conv_out.", {{"weight", {widest, channels, last_kernel}, kHalf}, {"bias", {channels}, kF32}});
    add_transformer(t, "mimi.enc_tf", mimi_layers, channels, mimi_ffn, kHalf, true);
    add_block(t, "mimi.down.", {{"first", {stride * channels, latent}, kHalf}, {"second", {stride * channels, latent}, kHalf}});

    t.push_back({"mimi.quant_out", {latent, channels}, kHalf});
    t.push_back({"mimi.up", {channels, 2 * (int64_t) stride}, kF32});
    add_transformer(t, "mimi.dec_tf", mimi_layers, channels, mimi_ffn, kHalf, true);
    add_block(t, "mimi.dec.conv_in.", {{"weight", {channels, widest, kernel}, kHalf}, {"bias", {widest}, kF32}});
    for (size_t i = 0; i < ratios.size(); i++) {
        const int64_t in = widest >> i, out = in / 2, r = ratios[i];
        const std::string b = "mimi.dec.blk." + std::to_string(i);
        add_block(t, b + ".up.", {{"weight", {in, out, 2 * r}, kHalf}, {"bias", {out}, kF32}});
        residuals(b, out);
    }
    add_block(t, "mimi.dec.conv_out.", {{"weight", {filters, 1, last_kernel}, kHalf}, {"bias", {1}, kF32}});
    return t;
}

}  // namespace

const Layout model_layout = {"pocket-tts", 1, "convert it again with reference/pocket-tts/convert.py, or download it again from its Hugging Face repository",
                             tensors};

}  // namespace pocket
