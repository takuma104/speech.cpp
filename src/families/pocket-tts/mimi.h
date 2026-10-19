#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"

namespace pocket {

/** What Mimi computed on the way, by stage name, each in ggml order ([channels, time] with the channel fastest). */
using MimiStages = std::map<std::string, std::vector<float>>;

/**
 * Mimi, the codec of Pocket TTS: 24 kHz mono audio and a latent of 32 values every 1920 samples (12.5 frames a second).
 * Its encoder is SEANet, which turns the samples into 512 channels at 200 frames a second, a transformer, and a
 * strided convolution down to the latent's rate; its decoder is the quantizer's projection back to 512 channels, a
 * depthwise transposed convolution up to 200 frames a second, a transformer and SEANet's decoder. Every convolution is
 * causal, padded on the left alone, and every attention is causal within a window of `context` positions.
 */
class Mimi {
public:
    /** The codec of the model file `m`, which outlives it. */
    Mimi(const ModelFile & m, ggml_backend_t backend);
    ~Mimi();
    Mimi(const Mimi &) = delete;
    Mimi & operator=(const Mimi &) = delete;

    int sample_rate() const { return sample_rate_; }
    /** The samples of a frame of the latent. */
    int hop() const { return hop_; }
    int latent_dim() const { return latent_; }
    /** The SHA-256 of the official encoder's tensors, which binds a voice file to the encoder that made it. */
    const std::string & encoder_sha256() const { return encoder_sha256_; }

    /**
     * The latent, row-major [frames, latent_dim], of mono audio at sample_rate(), padded with silence on the right to
     * whole frames as the official encode_to_latent() pads it. `stages`, when given, receives the first convolution's
     * output (conv_in), each block's after its strided convolution (block0, block1, ...), SEANet's (seanet) and the
     * transformer's (transformer).
     */
    std::vector<float> encode(const std::vector<float> & audio, MimiStages * stages = nullptr);

    /** Forgets the utterance being decoded: the next decode() starts from the state of a fresh decoder. */
    void start_decoding();

    /**
     * The samples of the next `frames` frames of the latent, row-major [frames, latent_dim] as the decoder reads them,
     * frames * hop() of them. Every stage carries what it needs of the frames before (a convolution its last inputs, a
     * transposed convolution the overlap of its last frame, the transformer the keys and values of the last context - 1
     * positions), so that decoding an utterance a few frames at a time gives the samples of decoding it whole but for
     * the rounding of products of other shapes. `stages`, when given, receives the quantizer's projection (quant), the
     * upsampling (up), the transformer (transformer), SEANet's first convolution (conv_in) and each block's output
     * after its residual units (block0, block1, ...).
     */
    std::vector<float> decode(const std::vector<float> & latent, int64_t frames, MimiStages * stages = nullptr);

private:
    /** A persistent tensor of the decoder's state, [ne0, ne1], zeroed by start_decoding(). */
    ggml_tensor * state(int64_t ne0, int64_t ne1);

    ggml_backend_t backend_;
    const ModelFile & m_;
    ggml_gallocr_t allocr_ = nullptr;
    int sample_rate_ = 0, hop_ = 0, latent_ = 0, channels_ = 0, stride_ = 0, heads_ = 0, layers_ = 0, context_ = 0;
    int residual_layers_ = 0, dilation_base_ = 0;
    float rope_base_ = 0, norm_eps_ = 0;
    std::vector<int32_t> ratios_;
    std::string encoder_sha256_;

    ggml_context * state_ctx_ = nullptr;
    ggml_backend_buffer_t state_buffer_ = nullptr;
    /** The positions the decoder has passed since start_decoding(), at the transformer's rate. */
    int64_t position_ = 0;
    ggml_tensor * up_state_ = nullptr;
    std::vector<ggml_tensor *> keys_, values_;
    ggml_tensor * conv_in_state_ = nullptr, * conv_out_state_ = nullptr;
    std::vector<ggml_tensor *> block_up_states_;
    /** Each residual unit's first convolution's tail, block by block. */
    std::vector<std::vector<ggml_tensor *>> residual_states_;
};

}  // namespace pocket
