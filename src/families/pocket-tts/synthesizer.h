#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "flow-lm.h"
#include "ggml-backend.h"
#include "mimi.h"
#include "model-file.h"
#include "text.h"
#include "voice.h"

namespace pocket {

/** Called with each piece of audio as it is made; returning false stops the synthesis. */
using AudioSink = std::function<bool(const float * samples, size_t n)>;

/** A request: the text, the seed and the variance of the noise, the flow head's steps, and the most frames to speak. */
struct Request {
    std::string text;
    uint64_t seed = 0;
    double temperature = 0;
    int steps = 1;
    /** The most frames the whole speech holds, or 0 for no bound. */
    int64_t max_frames = 0;
    /** The noise of every step in turn, row-major [steps, latent_dim], for a check; drawn from the seed when empty. */
    std::vector<float> noise;
};

/** Why a synthesis ended. */
enum class Ending {
    /** Every chunk came to its end of speech. */
    Complete,
    /** The speech reached the request's max_frames. */
    MaxFrames,
    /** Every chunk was spoken, and some reached their longest, tokens / 3 + 2 s, before the end of speech. */
    ModelLimit,
    /** The sink stopped it. */
    Stopped,
};

/**
 * Pocket TTS's synthesis of a text in a voice, as the official generate_audio() makes it: the text cut into chunks, each
 * spoken from the voice's keys and values a latent at a time until a few frames after the end of speech, and each latent
 * decoded by Mimi as it is made, a frame at a time from a fresh decoder for each chunk whose first 5 ms fade in.
 */
class Synthesizer {
public:
    Synthesizer(const std::string & path, ggml_backend_t backend);

    /** The voice of a WAVE file, encoded now, or of a voice file made for this model. */
    Voice load_voice(const std::string & path);

    Ending synthesize(const Request & r, const Voice & voice, const AudioSink & sink);

    int sample_rate() const { return mimi_.sample_rate(); }
    /** The samples of a frame of speech. */
    int hop() const { return mimi_.hop(); }
    int latent_dim() const { return mimi_.latent_dim(); }
    const Text & text() const { return text_; }

private:
    ggml_backend_t backend_;
    std::unique_ptr<const ModelFile> model_;
    FlowLM lm_;
    Mimi mimi_;
    Text text_;
    std::vector<float> emb_mean_, emb_std_, fade_;
    float eos_threshold_ = 0;
    int min_frames_before_eos_ = 0;
    double tokens_per_second_ = 0, padding_seconds_ = 0;
};

}  // namespace pocket
