#pragma once

#include <cstdint>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"

namespace pocket {

/** What a run of the flow language model gives: its last row's output after the final norm, [dim], and its logit of the end of speech. */
struct LmOutput {
    std::vector<float> hidden;
    float eos = 0;
};

/** What a step gives besides: the latent its flow head drew, [latent_dim], in the model's normalized space. */
struct StepOutput {
    LmOutput lm;
    std::vector<float> latent;
};

/**
 * The flow language model of Pocket TTS and its flow head. The model is a causal transformer over one sequence: the
 * voice prompt, the text's tokens, then a latent a step; it keeps the keys and values of every position it has read in
 * a cache on the device, and each step's output gives the logit of the end of speech and conditions the flow head,
 * which turns Gaussian noise into the step's latent in a few steps of Lagrangian self-distillation.
 */
class FlowLM {
public:
    /** The model of the file `m`, which outlives it. */
    FlowLM(const ModelFile & m, ggml_backend_t backend);
    ~FlowLM();
    FlowLM(const FlowLM &) = delete;
    FlowLM & operator=(const FlowLM &) = delete;

    int dim() const { return dim_; }
    int latent_dim() const { return latent_; }
    /** The positions the cache holds. */
    int64_t positions() const { return positions_; }

    /**
     * The rows of a voice's prompt, row-major [rows, dim]: bos_before_voice where the model has it, then each frame of
     * the voice's latent, row-major [frames, latent_dim] as Mimi's encoder gives it, through speaker_proj.
     */
    std::vector<float> voice_rows(const std::vector<float> & latent, int64_t frames);
    /** The rows of the lookup table of `tokens`, row-major [tokens, dim]. */
    std::vector<float> text_rows(const std::vector<int32_t> & tokens);

    /** Runs the voice's prompt (voice_rows()) from position 0, so that the cache holds the voice alone afterwards. */
    LmOutput voice(const std::vector<float> & latent, int64_t frames);
    /** Runs the tokens of a text at the next positions. */
    LmOutput text(const std::vector<int32_t> & tokens);
    /**
     * Runs one step at the next position on the latent before it, [latent_dim] in the model's normalized space, or on the
     * start of speech for an empty one, and draws the step's latent from `noise`, [latent_dim], in `steps` LSD steps.
     */
    StepOutput step(const std::vector<float> & latent, const std::vector<float> & noise, int steps);
    /**
     * The flow head alone: the latent of the output `hidden`, [dim], drawn from `noise` in `steps` LSD steps.
     * `velocities`, when given, receives the head's output of each step, row-major [steps, latent_dim].
     */
    std::vector<float> flow(const std::vector<float> & hidden, const std::vector<float> & noise, int steps, std::vector<float> * velocities = nullptr);

    /** Forgets the positions from `n` on, so that the next row goes to position `n`: a chunk starts again from the voice. */
    void keep(int64_t n);
    /** Makes room in the cache for `n` positions in all, so that no run of them moves the cache. */
    void reserve(int64_t n);
    /** The keys, after RoPE, and the values of layer `layer` at the positions held, row-major [positions, dim]. */
    void read_cache(int layer, std::vector<float> & keys, std::vector<float> & values) const;

private:
    /** The transformer on `rows`, [dim, n], at the next positions, the last row normed and its logit of the end of speech. */
    void build_run(Graph & g, ggml_tensor * rows, ggml_tensor ** hidden, ggml_tensor ** eos);
    /** The flow head on `hidden`, [dim, 1], from `noise`, [latent_dim, 1]; `velocities` receives each step's output. */
    ggml_tensor * build_flow(Graph & g, ggml_tensor * hidden, ggml_tensor * noise, int steps, std::vector<ggml_tensor *> * velocities);
    ggml_tensor * build_voice_rows(Graph & g, const std::vector<float> & latent, int64_t frames);
    LmOutput run(Graph & g, ggml_tensor * rows);

    ggml_backend_t backend_;
    const ModelFile & m_;
    ggml_gallocr_t allocr_ = nullptr;
    int dim_ = 0, latent_ = 0, flow_dim_ = 0, heads_ = 0, layers_ = 0, flow_blocks_ = 0;
    float rope_base_ = 0, norm_eps_ = 0, flow_eps_ = 0, time_eps_ = 0;
    bool bos_before_voice_ = false;
    /** Each time embedder's frequencies, as the checkpoint holds them. */
    std::vector<std::vector<float>> frequencies_;

    ggml_context * cache_ctx_ = nullptr;
    ggml_backend_buffer_t cache_buffer_ = nullptr;
    std::vector<ggml_tensor *> keys_, values_;
    int64_t capacity_ = 0, positions_ = 0;
};

}  // namespace pocket
