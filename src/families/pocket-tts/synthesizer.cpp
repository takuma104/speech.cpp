#include "synthesizer.h"

#include <cmath>
#include <random>
#include <stdexcept>

#include "error.h"
#include "layout.h"
#include "sampling.h"

namespace pocket {

Synthesizer::Synthesizer(const std::string & path, ggml_backend_t backend)
    : backend_(backend),
      model_(naming("model_path", [&] { return std::make_unique<const ModelFile>(path, backend, model_layout); })),
      lm_(*model_, backend),
      mimi_(*model_, backend),
      text_(*model_) {
    const std::string p = "pocket-tts.";
    emb_mean_ = Graph::read(model_->tensor("lm.emb_mean"));
    emb_std_ = Graph::read(model_->tensor("lm.emb_std"));
    fade_ = linspace(0.0f, 1.0f, (int) model_->u32(p + "generation.fade_in_samples"));
    eos_threshold_ = model_->f32(p + "sampler.eos_threshold");
    min_frames_before_eos_ = (int) model_->u32(p + "generation.min_frames_before_eos");
    tokens_per_second_ = model_->f64(p + "generation.tokens_per_second");
    padding_seconds_ = model_->f64(p + "generation.padding_seconds");
}

Voice Synthesizer::load_voice(const std::string & path) {
    return pocket::load_voice(path, *model_, mimi_, backend_);
}

Ending Synthesizer::synthesize(const Request & r, const Voice & voice, const AudioSink & sink) {
    if (r.steps < 1 || !(r.temperature >= 0)) throw std::logic_error("a Pocket TTS request reached the synthesis with steps or a temperature out of range");
    const std::vector<Chunk> chunks = text_.chunks(r.text);
    const int dim = mimi_.latent_dim();
    // The official code draws each latent's noise of a standard deviation of temp ** 0.5.
    const float scale = (float) std::sqrt(r.temperature);
    std::mt19937_64 rng(r.seed);
    lm_.voice(voice.latent, voice.frames);
    const int64_t voice_positions = lm_.positions();
    const double frame_rate = (double) mimi_.sample_rate() / mimi_.hop();
    int64_t spoken = 0;
    size_t drawn = 0;
    bool limited = false;
    for (const Chunk & chunk : chunks) {
        // _estimate_max_gen_len(): the tokens / 3 + 2 s the chunk may speak at most.
        const int64_t longest = (int64_t) std::ceil(((double) chunk.tokens.size() / tokens_per_second_ + padding_seconds_) * frame_rate);
        lm_.keep(voice_positions);
        lm_.reserve(voice_positions + (int64_t) chunk.tokens.size() + longest);
        lm_.text(chunk.tokens);
        mimi_.start_decoding();
        std::vector<float> previous;
        int64_t eos_step = -1, step = 0;
        for (; step < longest; step++) {
            std::vector<float> noise;
            if (r.noise.empty()) {
                noise = gaussian(rng, (size_t) dim);
                for (float & n : noise) n *= scale;
            } else {
                if ((drawn + 1) * (size_t) dim > r.noise.size()) throw std::logic_error("a request's noise ran out before its speech ended");
                noise.assign(r.noise.begin() + (std::ptrdiff_t) (drawn * dim), r.noise.begin() + (std::ptrdiff_t) ((drawn + 1) * dim));
            }
            drawn++;
            const StepOutput out = lm_.step(previous, noise, r.steps);
            if (out.lm.eos > eos_threshold_ && eos_step < 0 && step >= min_frames_before_eos_) eos_step = step;
            if (eos_step >= 0 && step >= eos_step + chunk.frames_after_eos) break;
            std::vector<float> latent(out.latent.size());
            for (size_t i = 0; i < latent.size(); i++) latent[i] = out.latent[i] * emb_std_[i % dim] + emb_mean_[i % dim];
            std::vector<float> audio = mimi_.decode(latent, 1);
            if (step == 0) {
                for (size_t i = 0; i < fade_.size() && i < audio.size(); i++) audio[i] *= fade_[i];
            }
            if (!sink(audio.data(), audio.size())) return Ending::Stopped;
            previous = out.latent;
            if (r.max_frames > 0 && ++spoken >= r.max_frames) return Ending::MaxFrames;
        }
        if (step == longest) limited = true;
    }
    return limited ? Ending::ModelLimit : Ending::Complete;
}

}  // namespace pocket
