#include "flow-lm.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "error.h"

namespace pocket {

/*
 * Activations are channel-first ([dim, rows], ne0 = dim), so a linear layer is one matrix product. The transformer's
 * fused projection gives each row's queries of every head, then its keys, then its values, and RoPE turns adjacent pairs
 * of a head's channels (GGML_ROPE_TYPE_NORMAL), as the official apply_rope() does.
 */

namespace {

/** The cache grows by this many positions at least. */
constexpr int64_t kCacheStep = 256;

ggml_tensor * layer_norm(ggml_context * ctx, const ModelFile & m, ggml_tensor * x, const std::string & name, float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), m.tensor(name + ".weight")), m.tensor(name + ".bias"));
}

ggml_tensor * linear(ggml_context * ctx, const ModelFile & m, ggml_tensor * x, const std::string & name) {
    return ggml_add(ctx, mul_mat(ctx, m.tensor(name + ".weight"), x), m.tensor(name + ".bias"));
}

/** The channels [from, from + n) of every row of `x`, [width, rows], as a contiguous tensor [n, rows]. */
ggml_tensor * part(ggml_context * ctx, ggml_tensor * x, int64_t from, int64_t n) {
    return ggml_cont(ctx, ggml_view_2d(ctx, x, n, x->ne[1], x->nb[1], (size_t) from * x->nb[0]));
}

}  // namespace

FlowLM::FlowLM(const ModelFile & m, ggml_backend_t backend) : backend_(backend), m_(m) {
    const std::string p = "pocket-tts.";
    dim_ = (int) m.width("lm.text_embd", 0);
    latent_ = (int) m.width("lm.bos", 0);
    flow_dim_ = (int) m.width("flow.in_proj.weight", 1);
    heads_ = m.size(p + "lm.num_heads");
    layers_ = m.count(p + "lm.num_layers");
    flow_blocks_ = m.count(p + "flow.num_blocks");
    rope_base_ = m.f32(p + "lm.rope_base");
    norm_eps_ = m.f32(p + "lm.norm_eps");
    flow_eps_ = m.f32(p + "flow.norm_eps");
    time_eps_ = m.f32(p + "flow.time_norm_eps");
    bos_before_voice_ = m.boolean(p + "lm.bos_before_voice");
    for (int i = 0; i < 2; i++) frequencies_.push_back(Graph::read(m.tensor("flow.time." + std::to_string(i) + ".freqs")));
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

FlowLM::~FlowLM() {
    if (allocr_) ggml_gallocr_free(allocr_);
    if (cache_buffer_) ggml_backend_buffer_free(cache_buffer_);
    if (cache_ctx_) ggml_free(cache_ctx_);
}

void FlowLM::reserve(int64_t n) {
    if (n <= capacity_) return;
    const int64_t capacity = (std::max(n, 2 * capacity_) + kCacheStep - 1) / kCacheStep * kCacheStep;
    ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 2 * (size_t) layers_, nullptr, true});
    std::vector<ggml_tensor *> keys, values;
    for (int l = 0; l < layers_; l++) {
        keys.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim_, capacity));
        values.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim_, capacity));
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend_);
    if (!buffer) {
        ggml_free(ctx);
        throw Error(Fault::OutOfMemory, "cannot allocate a key/value cache of " + std::to_string(capacity) + " positions on " +
                                            ggml_backend_name(backend_));
    }
    if (positions_ > 0) {
        Graph g;
        for (int l = 0; l < layers_; l++) {
            const auto rows = [&](ggml_tensor * t) { return ggml_view_2d(g.ctx(), t, dim_, positions_, t->nb[1], 0); };
            g.copy(rows(keys_[(size_t) l]), rows(keys[(size_t) l]));
            g.copy(rows(values_[(size_t) l]), rows(values[(size_t) l]));
        }
        g.compute(backend_, allocr_);
    }
    if (cache_buffer_) ggml_backend_buffer_free(cache_buffer_);
    if (cache_ctx_) ggml_free(cache_ctx_);
    cache_ctx_ = ctx;
    cache_buffer_ = buffer;
    keys_ = keys;
    values_ = values;
    capacity_ = capacity;
}

void FlowLM::keep(int64_t n) {
    if (n < 0 || n > positions_) throw std::logic_error("the flow language model keeps " + std::to_string(n) + " of " + std::to_string(positions_) + " positions");
    positions_ = n;
}

void FlowLM::read_cache(int layer, std::vector<float> & keys, std::vector<float> & values) const {
    keys.resize((size_t) (positions_ * dim_));
    values.resize(keys.size());
    if (positions_ == 0) return;
    ggml_backend_tensor_get(keys_.at((size_t) layer), keys.data(), 0, keys.size() * sizeof(float));
    ggml_backend_tensor_get(values_.at((size_t) layer), values.data(), 0, values.size() * sizeof(float));
}

void FlowLM::build_run(Graph & g, ggml_tensor * rows, ggml_tensor ** hidden, ggml_tensor ** eos) {
    ggml_context * ctx = g.ctx();
    const int64_t n = rows->ne[1], p0 = positions_, all = p0 + n, hd = dim_ / heads_;
    std::vector<int32_t> positions((size_t) n);
    for (int64_t i = 0; i < n; i++) positions[(size_t) i] = (int32_t) (p0 + i);
    ggml_tensor * pos = g.input(positions, n);
    ggml_tensor * mask = nullptr;
    if (n > 1) {
        std::vector<float> causal((size_t) (all * n));
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < all; j++) causal[(size_t) (i * all + j)] = j <= p0 + i ? 0.0f : -INFINITY;
        mask = g.input(causal, all, n);
    }
    ggml_tensor * x = rows;
    for (int l = 0; l < layers_; l++) {
        const std::string b = "lm.blk." + std::to_string(l) + ".";
        ggml_tensor * h = layer_norm(ctx, m_, x, b + "attn_norm", norm_eps_);
        ggml_tensor * qkv = mul_mat(ctx, m_.tensor(b + "attn_qkv"), h);
        const auto rope = [&](ggml_tensor * t) {
            return ggml_rope_ext(ctx, ggml_reshape_3d(ctx, t, hd, heads_, n), pos, nullptr, (int) hd, GGML_ROPE_TYPE_NORMAL, 0, rope_base_, 1.0f,
                                 0.0f, 1.0f, 0.0f, 0.0f);
        };
        ggml_tensor * q = rope(part(ctx, qkv, 0, dim_));
        ggml_tensor * k = rope(part(ctx, qkv, dim_, dim_));
        ggml_tensor * v = part(ctx, qkv, 2 * (int64_t) dim_, dim_);
        ggml_tensor * kc = keys_[(size_t) l], * vc = values_[(size_t) l];
        g.copy(ggml_reshape_2d(ctx, k, dim_, n), ggml_view_2d(ctx, kc, dim_, n, kc->nb[1], (size_t) p0 * kc->nb[1]));
        g.copy(v, ggml_view_2d(ctx, vc, dim_, n, vc->nb[1], (size_t) p0 * vc->nb[1]));
        // The copies are in the graph before the views that read the cache, which run after them.
        ggml_tensor * keys = ggml_view_3d(ctx, kc, hd, heads_, all, hd * sizeof(float), kc->nb[1], 0);
        ggml_tensor * values = ggml_view_3d(ctx, vc, hd, heads_, all, hd * sizeof(float), vc->nb[1], 0);
        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, keys, 0, 2, 1, 3));
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, values, 1, 2, 0, 3));
        ggml_tensor * kq = ggml_soft_max_ext(ctx, mul_mat(ctx, kh, qh), mask, 1.0f / std::sqrt((float) hd), 0.0f);
        ggml_tensor * o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, mul_mat(ctx, vt, kq), 0, 2, 1, 3)), dim_, n);
        x = ggml_add(ctx, x, mul_mat(ctx, m_.tensor(b + "attn_out"), o));
        h = layer_norm(ctx, m_, x, b + "ffn_norm", norm_eps_);
        x = ggml_add(ctx, x, mul_mat(ctx, m_.tensor(b + "ffn_down"), gelu_tanh(ctx, mul_mat(ctx, m_.tensor(b + "ffn_up"), h))));
    }
    ggml_tensor * last = ggml_view_2d(ctx, x, dim_, 1, x->nb[1], (size_t) (n - 1) * x->nb[1]);
    *hidden = layer_norm(ctx, m_, last, "lm.out_norm", norm_eps_);
    *eos = ggml_add(ctx, mul_mat(ctx, ggml_reshape_2d(ctx, m_.tensor("lm.eos.weight"), dim_, 1), *hidden), m_.tensor("lm.eos.bias"));
}

ggml_tensor * FlowLM::build_flow(Graph & g, ggml_tensor * hidden, ggml_tensor * noise, int steps, std::vector<ggml_tensor *> * velocities) {
    ggml_context * ctx = g.ctx();
    const int64_t f = flow_dim_;
    const auto chunk = [&](ggml_tensor * t, int i) { return ggml_cont(ctx, ggml_view_2d(ctx, t, f, 1, t->nb[1], (size_t) i * f * sizeof(float))); };
    const auto modulate = [&](ggml_tensor * t, ggml_tensor * shift, ggml_tensor * scale) {
        return ggml_add(ctx, ggml_mul(ctx, t, ggml_scale_bias(ctx, scale, 1.0f, 1.0f)), shift);
    };
    // A time embedder: the cosines and the sines of the time by each frequency, two linear layers about SiLU, and an
    // RMSNorm that divides by the unbiased variance of its input, as the official _rms_norm() does, not by its mean
    // square.
    const auto time = [&](int i, float t) {
        const std::vector<float> & freqs = frequencies_[(size_t) i];
        std::vector<float> embedding(2 * freqs.size());
        for (size_t k = 0; k < freqs.size(); k++) {
            const float a = t * freqs[k];
            embedding[k] = std::cos(a);
            embedding[freqs.size() + k] = std::sin(a);
        }
        const std::string b = "flow.time." + std::to_string(i) + ".";
        ggml_tensor * e = linear(ctx, m_, ggml_silu(ctx, linear(ctx, m_, g.input(embedding, (int64_t) embedding.size(), 1), b + "fc1")), b + "fc2");
        ggml_tensor * centred = ggml_sub(ctx, e, ggml_mean(ctx, e));
        ggml_tensor * var = ggml_scale_bias(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, centred)), 1.0f / (float) (f - 1), time_eps_);
        return ggml_mul(ctx, e, ggml_div(ctx, m_.tensor(b + "norm"), ggml_sqrt(ctx, var)));
    };
    ggml_tensor * cond = linear(ctx, m_, hidden, "flow.cond");
    ggml_tensor * current = noise;
    for (int i = 0; i < steps; i++) {
        const float s = (float) ((double) i / steps), t = (float) ((double) (i + 1) / steps);
        ggml_tensor * y = ggml_add(ctx, cond, ggml_scale(ctx, ggml_add(ctx, time(0, s), time(1, t)), 0.5f));
        ggml_tensor * sy = ggml_silu(ctx, y);
        ggml_tensor * x = linear(ctx, m_, current, "flow.in_proj");
        for (int j = 0; j < flow_blocks_; j++) {
            const std::string b = "flow.blk." + std::to_string(j) + ".";
            ggml_tensor * ada = linear(ctx, m_, sy, b + "ada");
            ggml_tensor * h = modulate(layer_norm(ctx, m_, x, b + "norm", flow_eps_), chunk(ada, 0), chunk(ada, 1));
            h = linear(ctx, m_, ggml_silu(ctx, linear(ctx, m_, h, b + "fc1")), b + "fc2");
            x = ggml_add(ctx, x, ggml_mul(ctx, chunk(ada, 2), h));
        }
        ggml_tensor * ada = linear(ctx, m_, sy, "flow.final.ada");
        ggml_tensor * velocity = linear(ctx, m_, modulate(ggml_norm(ctx, x, flow_eps_), chunk(ada, 0), chunk(ada, 1)), "flow.final.out");
        if (velocities) velocities->push_back(velocity);
        current = ggml_add(ctx, current, ggml_scale(ctx, velocity, 1.0f / (float) steps));
    }
    return current;
}

ggml_tensor * FlowLM::build_voice_rows(Graph & g, const std::vector<float> & latent, int64_t frames) {
    if (frames < 1 || (int64_t) latent.size() != frames * latent_) {
        throw Error(Fault::OutOfRange, "a voice's latent is a whole number of frames of " + std::to_string(latent_) + " values");
    }
    ggml_tensor * rows = mul_mat(g.ctx(), m_.tensor("lm.speaker_proj"), g.input(latent, latent_, frames));
    if (bos_before_voice_) rows = ggml_concat(g.ctx(), ggml_reshape_2d(g.ctx(), m_.tensor("lm.bos_before_voice"), dim_, 1), rows, 1);
    return rows;
}

std::vector<float> FlowLM::voice_rows(const std::vector<float> & latent, int64_t frames) {
    Graph g;
    ggml_tensor * rows = build_voice_rows(g, latent, frames);
    g.output(rows);
    g.compute(backend_, allocr_);
    return Graph::read(rows);
}

std::vector<float> FlowLM::text_rows(const std::vector<int32_t> & tokens) {
    Graph g;
    ggml_tensor * rows = ggml_get_rows(g.ctx(), m_.tensor("lm.text_embd"), g.input(tokens, (int64_t) tokens.size()));
    g.output(rows);
    g.compute(backend_, allocr_);
    return Graph::read(rows);
}

LmOutput FlowLM::run(Graph & g, ggml_tensor * rows) {
    reserve(positions_ + rows->ne[1]);
    ggml_tensor * hidden = nullptr, * eos = nullptr;
    build_run(g, rows, &hidden, &eos);
    g.output(hidden);
    g.output(eos);
    g.compute(backend_, allocr_);
    positions_ += rows->ne[1];
    return {Graph::read(hidden), Graph::read(eos)[0]};
}

LmOutput FlowLM::voice(const std::vector<float> & latent, int64_t frames) {
    positions_ = 0;
    Graph g;
    return run(g, build_voice_rows(g, latent, frames));
}

LmOutput FlowLM::text(const std::vector<int32_t> & tokens) {
    if (tokens.empty()) throw Error(Fault::OutOfRange, "the flow language model has no token to read");
    Graph g;
    return run(g, ggml_get_rows(g.ctx(), m_.tensor("lm.text_embd"), g.input(tokens, (int64_t) tokens.size())));
}

StepOutput FlowLM::step(const std::vector<float> & latent, const std::vector<float> & noise, int steps) {
    if ((!latent.empty() && (int) latent.size() != latent_) || (int) noise.size() != latent_ || steps < 1) {
        throw std::logic_error("a step of the flow language model takes a latent and noise of " + std::to_string(latent_) + " values");
    }
    reserve(positions_ + 1);
    Graph g;
    // The start of speech is the learned bos_emb, which the official model puts where its input is NaN.
    ggml_tensor * in = latent.empty() ? ggml_reshape_2d(g.ctx(), m_.tensor("lm.bos"), latent_, 1) : g.input(latent, latent_, 1);
    ggml_tensor * hidden = nullptr, * eos = nullptr;
    build_run(g, mul_mat(g.ctx(), m_.tensor("lm.in_proj"), in), &hidden, &eos);
    ggml_tensor * drawn = build_flow(g, hidden, g.input(noise, latent_, 1), steps, nullptr);
    for (ggml_tensor * t : {hidden, eos, drawn}) g.output(t);
    g.compute(backend_, allocr_);
    positions_ += 1;
    return {{Graph::read(hidden), Graph::read(eos)[0]}, Graph::read(drawn)};
}

std::vector<float> FlowLM::flow(const std::vector<float> & hidden, const std::vector<float> & noise, int steps, std::vector<float> * velocities) {
    if ((int) hidden.size() != dim_ || (int) noise.size() != latent_ || steps < 1) {
        throw std::logic_error("the flow head takes an output of " + std::to_string(dim_) + " values and noise of " + std::to_string(latent_));
    }
    Graph g;
    std::vector<ggml_tensor *> each;
    ggml_tensor * drawn = build_flow(g, g.input(hidden, dim_, 1), g.input(noise, latent_, 1), steps, velocities ? &each : nullptr);
    g.output(drawn);
    for (ggml_tensor * t : each) g.output(t);
    g.compute(backend_, allocr_);
    if (velocities) {
        velocities->clear();
        for (ggml_tensor * t : each) {
            const std::vector<float> v = Graph::read(t);
            velocities->insert(velocities->end(), v.begin(), v.end());
        }
    }
    return Graph::read(drawn);
}

}  // namespace pocket
