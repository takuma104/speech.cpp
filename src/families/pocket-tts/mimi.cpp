#include "mimi.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "error.h"

namespace pocket {

/*
 * Activations are channel-first ([channels, time], ne0 = channels), so a linear layer is one matrix product and a
 * per-channel parameter broadcasts over time. A convolution of width K and dilation d runs as K matrix products, one
 * per tap, over views of its input shifted in time, the input after the (K - 1) d samples before it: zeros in the
 * encoder, as the official module starts its state, and the carried tail in the decoder. A convolution of width 2s and
 * stride s reads its input after s samples before it, cut into frames of s samples, which a reshape of the
 * channel-first layout gives for free: output frame n is the first half of the kernel on frame n and the second half
 * on frame n + 1. A transposed convolution of width 2s and stride s gives every tap of every input frame in one matrix
 * product: output frame n is the first s taps of input frame n plus the last s taps of frame n - 1, which the decoder
 * carries from one call to the next, the official module cutting the s samples past the last frame.
 */

namespace {

struct Layers {
    Graph & g;
    const ModelFile & m;

    ggml_context * ctx() const { return g.ctx(); }

    ggml_tensor * linear(ggml_tensor * x, const std::string & name) {
        return ggml_add(ctx(), mul_mat(ctx(), m.tensor(name + ".weight"), x), m.tensor(name + ".bias"));
    }

    ggml_tensor * layer_norm(ggml_tensor * x, const std::string & name, float eps) {
        return ggml_add(ctx(), ggml_mul(ctx(), ggml_norm(ctx(), x, eps), m.tensor(name + ".weight")), m.tensor(name + ".bias"));
    }

    /**
     * `x` after the `n` samples before it: the decoder's carried `tail`, [channels, n], which then receives the last n
     * samples of what it returns, or zeros where there is no tail.
     */
    ggml_tensor * after_tail(ggml_tensor * x, int64_t n, ggml_tensor * tail) {
        if (n == 0) return x;
        if (!tail) return ggml_concat(ctx(), g.zeros(x->ne[0], n), x, 1);
        ggml_tensor * xp = ggml_concat(ctx(), tail, x, 1);
        // The copy reads xp, which holds the old tail already, so it cannot overtake the concatenation.
        g.copy(ggml_view_2d(ctx(), xp, xp->ne[0], n, xp->nb[1], (size_t) x->ne[1] * xp->nb[1]), tail);
        return xp;
    }

    /** A causal convolution of stride 1; weight ne = [in, out, k]. */
    ggml_tensor * conv(ggml_tensor * x, const std::string & name, int dilation = 1, ggml_tensor * tail = nullptr) {
        ggml_tensor * w = m.tensor(name + ".weight");
        const int64_t in = w->ne[0], out = w->ne[1], width = w->ne[2], t = x->ne[1];
        ggml_tensor * y = nullptr;
        if (width == 1) {
            y = mul_mat(ctx(), ggml_reshape_2d(ctx(), w, in, out), x);
        } else if (in == 1) {
            // The waveform's convolution reads one channel, so each tap is the samples times a column of weights.
            // Vulkan's matrix product rounds its activations to half precision; products of the samples by each tap's
            // weights keep float32 on every backend.
            ggml_tensor * xp = ggml_repeat_4d(ctx(), after_tail(x, (width - 1) * dilation, tail), out, t + (width - 1) * dilation, 1, 1);
            for (int64_t k = 0; k < width; k++) {
                ggml_tensor * wk = ggml_view_1d(ctx(), w, out, k * w->nb[2]);
                ggml_tensor * yk = ggml_mul(ctx(), ggml_view_2d(ctx(), xp, out, t, xp->nb[1], k * dilation * xp->nb[1]), wk);
                y = y ? ggml_add(ctx(), y, yk) : yk;
            }
        } else {
            ggml_tensor * xp = after_tail(x, (width - 1) * dilation, tail);
            for (int64_t k = 0; k < width; k++) {
                ggml_tensor * wk = ggml_view_2d(ctx(), w, in, out, w->nb[1], k * w->nb[2]);
                ggml_tensor * xk = ggml_view_2d(ctx(), xp, in, t, xp->nb[1], k * dilation * xp->nb[1]);
                ggml_tensor * yk = mul_mat(ctx(), wk, xk);
                y = y ? ggml_add(ctx(), y, yk) : yk;
            }
        }
        return ggml_add(ctx(), y, m.tensor(name + ".bias"));
    }

    /**
     * A causal convolution of width 2s and stride s, its kernel stored in halves [s * in, out], on `x` after the s
     * samples `before`, [channels, s].
     */
    ggml_tensor * down(ggml_tensor * x, ggml_tensor * before, const std::string & name, int s, bool bias) {
        const int64_t c = x->ne[0], n = x->ne[1] / s;
        ggml_tensor * frames = ggml_reshape_2d(ctx(), ggml_concat(ctx(), before, x, 1), c * s, n + 1);
        ggml_tensor * own = ggml_view_2d(ctx(), frames, c * s, n, frames->nb[1], 0);
        ggml_tensor * next = ggml_view_2d(ctx(), frames, c * s, n, frames->nb[1], frames->nb[1]);
        ggml_tensor * y = ggml_add(ctx(), mul_mat(ctx(), m.tensor(name + ".first"), own), mul_mat(ctx(), m.tensor(name + ".second"), next));
        return bias ? ggml_add(ctx(), y, m.tensor(name + ".bias")) : y;
    }

    /**
     * The overlap of a transposed convolution: `z`, [out, 2s, frames], every tap of every input frame, gives output
     * frame n as its first s taps of frame n plus its last s taps of frame n - 1, `carry`'s for the first frame, which
     * then receives those of the last frame. [out, s * frames].
     */
    ggml_tensor * overlap(ggml_tensor * z, int64_t s, ggml_tensor * carry) {
        const int64_t out = z->ne[0], frames = z->ne[2];
        auto taps = [&](int64_t first_tap, int64_t first_frame, int64_t n) {
            return ggml_cont(ctx(), ggml_view_3d(ctx(), z, out, s, n, z->nb[1], z->nb[2], first_tap * z->nb[1] + first_frame * z->nb[2]));
        };
        ggml_tensor * edge = ggml_reshape_3d(ctx(), carry, out, s, 1);
        ggml_tensor * before = frames > 1 ? ggml_concat(ctx(), edge, taps(s, 0, frames - 1), 2) : edge;
        ggml_tensor * y = ggml_add(ctx(), taps(0, 0, frames), before);
        // The sum reads the old carry; it goes into the graph before the copy that overwrites it.
        g.expand(y);
        g.copy(taps(s, frames - 1, 1), carry);
        return ggml_reshape_2d(ctx(), y, out, s * frames);
    }

    /** A causal transposed convolution of width 2s and stride s; weight ne = [in, out, 2s]. */
    ggml_tensor * up(ggml_tensor * x, const std::string & name, int s, ggml_tensor * carry) {
        ggml_tensor * w = m.tensor(name + ".weight");
        const int64_t in = w->ne[0], out = w->ne[1], frames = x->ne[1];
        ggml_tensor * z = ggml_reshape_3d(ctx(), mul_mat(ctx(), ggml_reshape_2d(ctx(), w, in, out * 2 * s), x), out, 2 * s, frames);
        return ggml_add(ctx(), overlap(z, s, carry), m.tensor(name + ".bias"));
    }

    /** A SEANet residual unit: ELU, a convolution to fewer channels, ELU, a width-1 convolution back, and the skip. */
    ggml_tensor * residual(ggml_tensor * x, const std::string & name, int dilation, ggml_tensor * tail) {
        ggml_tensor * h = conv(ggml_elu(ctx(), x), name + ".conv1", dilation, tail);
        h = conv(ggml_elu(ctx(), h), name + ".conv2");
        return ggml_add(ctx(), x, h);
    }

    /**
     * Attention of the queries `q`, [head_dim, heads, t], to the keys and values `k` and `v`, [head_dim, heads, c + t],
     * of positions `first` - c onwards, whose last t rows are the queries' own: each query to the keys of the `context`
     * positions up to its own that are not before position 0, in blocks of `context` queries, so that no block holds
     * more than 2 * context - 1 keys however long the input.
     */
    ggml_tensor * attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int64_t first, int64_t context) {
        const int64_t hd = q->ne[0], heads = q->ne[1], t = q->ne[2], c = k->ne[2] - t;
        ggml_tensor * out = nullptr;
        for (int64_t a = 0; a < t; a += context) {
            const int64_t qn = std::min(context, t - a), k0 = std::max<int64_t>(0, c + a - context + 1), kn = c + a + qn - k0;
            const auto rows = [&](ggml_tensor * x, int64_t from, int64_t n) {
                return ggml_view_3d(ctx(), x, hd, heads, n, x->nb[1], x->nb[2], (size_t) from * x->nb[2]);
            };
            ggml_tensor * qb = ggml_cont(ctx(), ggml_permute(ctx(), rows(q, a, qn), 0, 2, 1, 3));
            ggml_tensor * kb = ggml_cont(ctx(), ggml_permute(ctx(), rows(k, k0, kn), 0, 2, 1, 3));
            ggml_tensor * vb = ggml_cont(ctx(), ggml_permute(ctx(), rows(v, k0, kn), 1, 2, 0, 3));
            std::vector<float> mask((size_t) (kn * qn));
            for (int64_t i = 0; i < qn; i++) {
                for (int64_t j = 0; j < kn; j++) {
                    const int64_t qp = first + a + i, kp = first - c + k0 + j;
                    mask[(size_t) (i * kn + j)] = kp >= 0 && kp <= qp && qp - kp < context ? 0.0f : -INFINITY;
                }
            }
            ggml_tensor * kq = ggml_soft_max_ext(ctx(), mul_mat(ctx(), kb, qb), g.input(mask, kn, qn), 1.0f / std::sqrt((float) hd), 0.0f);
            ggml_tensor * kqv = ggml_cont(ctx(), ggml_permute(ctx(), mul_mat(ctx(), vb, kq), 0, 2, 1, 3));
            ggml_tensor * o = ggml_reshape_2d(ctx(), kqv, hd * heads, qn);
            out = out ? ggml_concat(ctx(), out, o, 1) : o;
        }
        return out;
    }

    /**
     * A transformer of `prefix` on `x`, [dim, t], the rows from position `first`: pre-norm layers of causal attention
     * within `context` positions, with RoPE on pairs of channels, and a feed-forward of tanh-approximated GELU, each
     * scaled before it is added. `keys` and `values`, where given, hold each layer's keys and values of the context - 1
     * positions before these, [dim, context - 1], and receive those of the last context - 1 positions.
     */
    ggml_tensor * transformer(ggml_tensor * x, const std::string & prefix, int layers, int heads, int64_t context, float base, float eps,
                              int64_t first, const std::vector<ggml_tensor *> * keys, const std::vector<ggml_tensor *> * values) {
        const int64_t dim = x->ne[0], t = x->ne[1], hd = dim / heads;
        std::vector<int32_t> positions((size_t) t);
        for (int64_t i = 0; i < t; i++) positions[(size_t) i] = (int32_t) (first + i);
        ggml_tensor * pos = g.input(positions, t);
        for (int l = 0; l < layers; l++) {
            const std::string b = prefix + ".blk." + std::to_string(l) + ".";
            ggml_tensor * h = layer_norm(x, b + "attn_norm", eps);
            ggml_tensor * qkv = mul_mat(ctx(), m.tensor(b + "attn_qkv"), h);
            const auto part = [&](int i) {
                return ggml_reshape_3d(ctx(), ggml_cont(ctx(), ggml_view_2d(ctx(), qkv, dim, t, qkv->nb[1], (size_t) i * dim * qkv->nb[0])), hd, heads, t);
            };
            ggml_tensor * q = ggml_rope_ext(ctx(), part(0), pos, nullptr, (int) hd, GGML_ROPE_TYPE_NORMAL, 0, base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * k = ggml_rope_ext(ctx(), part(1), pos, nullptr, (int) hd, GGML_ROPE_TYPE_NORMAL, 0, base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * v = part(2);
            if (keys) {
                ggml_tensor * k_all = ggml_concat(ctx(), (*keys)[(size_t) l], ggml_reshape_2d(ctx(), k, dim, t), 1);
                ggml_tensor * v_all = ggml_concat(ctx(), (*values)[(size_t) l], ggml_reshape_2d(ctx(), v, dim, t), 1);
                g.copy(ggml_view_2d(ctx(), k_all, dim, context - 1, k_all->nb[1], (size_t) t * k_all->nb[1]), (*keys)[(size_t) l]);
                g.copy(ggml_view_2d(ctx(), v_all, dim, context - 1, v_all->nb[1], (size_t) t * v_all->nb[1]), (*values)[(size_t) l]);
                k = ggml_reshape_3d(ctx(), k_all, hd, heads, context - 1 + t);
                v = ggml_reshape_3d(ctx(), v_all, hd, heads, context - 1 + t);
            }
            ggml_tensor * o = mul_mat(ctx(), m.tensor(b + "attn_out"), attention(q, k, v, first, context));
            x = ggml_add(ctx(), x, ggml_mul(ctx(), o, m.tensor(b + "attn_scale")));
            h = layer_norm(x, b + "ffn_norm", eps);
            h = mul_mat(ctx(), m.tensor(b + "ffn_down"), gelu_tanh(ctx(), mul_mat(ctx(), m.tensor(b + "ffn_up"), h)));
            x = ggml_add(ctx(), x, ggml_mul(ctx(), h, m.tensor(b + "ffn_scale")));
        }
        return x;
    }
};

}  // namespace

Mimi::Mimi(const ModelFile & m, ggml_backend_t backend) : backend_(backend), m_(m) {
    const std::string c = "pocket-tts.mimi.";
    sample_rate_ = m.size("speech.sample_rate");
    hop_ = m.size(c + "hop_length");
    latent_ = (int) m.width("lm.bos", 0);
    channels_ = (int) m.width("mimi.quant_out", 1);
    stride_ = m.size(c + "downsample_stride");
    heads_ = m.size(c + "num_heads");
    layers_ = m.count(c + "num_layers");
    context_ = m.size(c + "context");
    residual_layers_ = m.count(c + "residual_layers");
    dilation_base_ = m.size(c + "dilation_base");
    rope_base_ = m.f32(c + "rope_base");
    norm_eps_ = m.f32(c + "norm_eps");
    ratios_ = m.i32_array(c + "ratios");
    encoder_sha256_ = m.str(c + "encoder_sha256");

    const size_t states = 3 + 2 * (size_t) layers_ + ratios_.size() * (1 + (size_t) residual_layers_);
    state_ctx_ = ggml_init({ggml_tensor_overhead() * states, nullptr, true});
    up_state_ = state(channels_, stride_);
    for (int l = 0; l < layers_; l++) {
        keys_.push_back(state(channels_, context_ - 1));
        values_.push_back(state(channels_, context_ - 1));
    }
    conv_in_state_ = state(channels_, m.width("mimi.dec.conv_in.weight", 2) - 1);
    for (size_t i = 0; i < ratios_.size(); i++) {
        const std::string b = "mimi.dec.blk." + std::to_string(i);
        const int64_t out = m.width(b + ".up.weight", 1);
        block_up_states_.push_back(state(out, ratios_[i]));
        residual_states_.emplace_back();
        for (int j = 0, dilation = 1; j < residual_layers_; j++, dilation *= dilation_base_) {
            residual_states_.back().push_back(state(out, (m.width(b + ".res." + std::to_string(j) + ".conv1.weight", 2) - 1) * dilation));
        }
    }
    conv_out_state_ = state(m.width("mimi.dec.conv_out.weight", 0), m.width("mimi.dec.conv_out.weight", 2) - 1);
    state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_, backend_);
    if (!state_buffer_) throw Error(Fault::OutOfMemory, "cannot allocate the state of Mimi's decoder on " + std::string(ggml_backend_name(backend_)));
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    start_decoding();
}

Mimi::~Mimi() {
    if (allocr_) ggml_gallocr_free(allocr_);
    if (state_buffer_) ggml_backend_buffer_free(state_buffer_);
    if (state_ctx_) ggml_free(state_ctx_);
}

ggml_tensor * Mimi::state(int64_t ne0, int64_t ne1) {
    return ggml_new_tensor_2d(state_ctx_, GGML_TYPE_F32, ne0, ne1);
}

std::vector<float> Mimi::encode(const std::vector<float> & audio, MimiStages * stages) {
    if (audio.empty()) throw Error(Fault::OutOfRange, "Mimi has no audio to encode");
    const int64_t frames = ((int64_t) audio.size() + hop_ - 1) / hop_, n = frames * hop_;
    std::vector<float> padded = audio;
    padded.resize((size_t) n, 0.0f);
    Graph g;
    Layers l{g, m_};
    std::vector<std::pair<std::string, ggml_tensor *>> kept;
    const auto stage = [&](const std::string & name, ggml_tensor * t) {
        if (!stages) return;
        g.output(t);
        kept.push_back({name, t});
    };
    ggml_tensor * x = l.conv(g.input(padded, 1, n), "mimi.enc.conv_in");
    stage("conv_in", x);
    for (size_t i = 0; i < ratios_.size(); i++) {
        const std::string b = "mimi.enc.blk." + std::to_string(i);
        for (int j = 0, dilation = 1; j < residual_layers_; j++, dilation *= dilation_base_) x = l.residual(x, b + ".res." + std::to_string(j), dilation, nullptr);
        const int r = ratios_[ratios_.size() - 1 - i];
        x = l.down(ggml_elu(g.ctx(), x), g.zeros(x->ne[0], r), b + ".down", r, true);
        stage("block" + std::to_string(i), x);
    }
    x = l.conv(ggml_elu(g.ctx(), x), "mimi.enc.conv_out");
    stage("seanet", x);
    x = l.transformer(x, "mimi.enc_tf", layers_, heads_, context_, rope_base_, norm_eps_, 0, nullptr, nullptr);
    stage("transformer", x);
    // The official downsampling pads on the left with its input's first frame repeated rather than with zeros.
    ggml_tensor * first = ggml_repeat_4d(g.ctx(), ggml_view_2d(g.ctx(), x, x->ne[0], 1, x->nb[1], 0), x->ne[0], stride_, 1, 1);
    x = l.down(x, first, "mimi.down", stride_, false);
    g.output(x);
    g.compute(backend_, allocr_);
    if (stages) {
        for (const auto & [name, t] : kept) (*stages)[name] = Graph::read(t);
    }
    return Graph::read(x);
}

void Mimi::start_decoding() {
    ggml_backend_buffer_clear(state_buffer_, 0);
    position_ = 0;
}

std::vector<float> Mimi::decode(const std::vector<float> & latent, int64_t frames, MimiStages * stages) {
    if (frames < 1 || (int64_t) latent.size() != frames * latent_) {
        throw Error(Fault::OutOfRange, "Mimi's decoder takes a whole number of frames of " + std::to_string(latent_) + " values");
    }
    Graph g;
    Layers l{g, m_};
    std::vector<std::pair<std::string, ggml_tensor *>> kept;
    const auto stage = [&](const std::string & name, ggml_tensor * t) {
        if (!stages) return;
        g.output(t);
        kept.push_back({name, t});
    };
    ggml_tensor * x = mul_mat(g.ctx(), m_.tensor("mimi.quant_out"), g.input(latent, latent_, frames));
    stage("quant", x);
    // The depthwise transposed convolution: each channel times its own kernel of 2s taps.
    ggml_tensor * w = m_.tensor("mimi.up");
    ggml_tensor * z = ggml_mul(g.ctx(), ggml_repeat_4d(g.ctx(), ggml_reshape_3d(g.ctx(), x, channels_, 1, frames), channels_, 2 * stride_, frames, 1),
                               ggml_reshape_3d(g.ctx(), w, channels_, 2 * stride_, 1));
    x = l.overlap(z, stride_, up_state_);
    stage("up", x);
    x = l.transformer(x, "mimi.dec_tf", layers_, heads_, context_, rope_base_, norm_eps_, position_, &keys_, &values_);
    stage("transformer", x);
    x = l.conv(x, "mimi.dec.conv_in", 1, conv_in_state_);
    stage("conv_in", x);
    for (size_t i = 0; i < ratios_.size(); i++) {
        const std::string b = "mimi.dec.blk." + std::to_string(i);
        x = l.up(ggml_elu(g.ctx(), x), b + ".up", ratios_[i], block_up_states_[i]);
        for (int j = 0, dilation = 1; j < residual_layers_; j++, dilation *= dilation_base_) {
            x = l.residual(x, b + ".res." + std::to_string(j), dilation, residual_states_[i][(size_t) j]);
        }
        stage("block" + std::to_string(i), x);
    }
    x = l.conv(ggml_elu(g.ctx(), x), "mimi.dec.conv_out", 1, conv_out_state_);
    g.output(x);
    g.compute(backend_, allocr_);
    if (stages) {
        for (const auto & [name, t] : kept) (*stages)[name] = Graph::read(t);
    }
    position_ += frames * stride_;
    return Graph::read(x);
}

}  // namespace pocket
