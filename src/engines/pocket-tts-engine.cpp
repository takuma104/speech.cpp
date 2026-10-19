#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "engine.h"
#include "error.h"
#include "pocket-tts/synthesizer.h"

namespace {

/** The frames of a warm-up's speech, which run the prompt, a step and a frame of the decoder. */
constexpr int64_t kWarmUpFrames = 4;

class PocketTtsEngine : public Engine {
public:
    PocketTtsEngine(const std::string & path, ggml_backend_t backend) : synth_(path, backend) {}

    speech_stop speak(const std::string & text, const RequestValues & values, Run & run) override {
        pocket::Request r;
        r.text = text;
        r.seed = (uint64_t) values.integer(SPEECH_OPT_SEED);
        r.temperature = values.number(SPEECH_OPT_TEMPERATURE);
        r.steps = (int) values.integer(SPEECH_OPT_STEPS);
        if (values.has(SPEECH_OPT_MAX_SECONDS)) {
            const double seconds = values.number(SPEECH_OPT_MAX_SECONDS);
            r.max_frames = (int64_t) std::floor(seconds * synth_.sample_rate()) / synth_.hop();
            if (r.max_frames < 1) {
                throw Error(Fault::OutOfRange, "max_seconds is shorter than a frame of the model's speech; give at least " +
                                                   std::to_string((double) synth_.hop() / synth_.sample_rate()) + " seconds",
                            "max_seconds");
            }
        }
        const pocket::Ending ending = synth_.synthesize(r, voices_.at(values.string(SPEECH_OPT_VOICE)),
                                                        [&](const float * samples, size_t n) { return run.audio(samples, n); });
        // A synthesis the sink stopped is cancelled, which the request reports whatever this returns.
        return ending == pocket::Ending::MaxFrames    ? SPEECH_STOP_MAX_SECONDS
               : ending == pocket::Ending::ModelLimit ? SPEECH_STOP_MODEL_LIMIT
                                                      : SPEECH_STOP_COMPLETE;
    }

    std::string add_voice(const std::string & name, const std::string & path) override {
        voices_.emplace(name, synth_.load_voice(path));
        return "";
    }

    void warm_up() override {
        // The first synthesis compiles the GPU kernels, which on Vulkan takes seconds; a voice of a silent frame and a
        // few frames of speech run every graph, since a model has no voice when it loads.
        pocket::Request warmup;
        warmup.text = "Hello.";
        warmup.steps = 1;
        warmup.max_frames = kWarmUpFrames;
        const pocket::Voice silence{std::vector<float>((size_t) synth_.latent_dim(), 0.0f), 1};
        synth_.synthesize(warmup, silence, [](const float *, size_t) { return true; });
    }

private:
    pocket::Synthesizer synth_;
    std::map<std::string, pocket::Voice> voices_;
};

/** The shortest decimal that reads back as `v`, so that a default the model file holds shows as the official code writes it. */
double shortest(double v) {
    char text[32];
    for (int digits = 1;; digits++) {
        std::snprintf(text, sizeof text, "%.*g", digits, v);
        if (std::strtod(text, nullptr) == v) return std::strtod(text, nullptr);
    }
}

/** The tokens of a text as the synthesis reads them: those of its chunks, each prepared. */
struct TokenCounter {
    explicit TokenCounter(const ModelFile & m) : text(m) {}
    size_t count(const std::string & t) const {
        size_t n = 0;
        for (const pocket::Chunk & c : text.chunks(t)) n += c.tokens.size();
        return n;
    }
    pocket::Text text;
};

}  // namespace

/**
 * The table of the options Pocket TTS takes. It has no voice of its own: a request speaks in a voice added since
 * loading, from a WAVE file or a voice file. Its one language is checked and not used. It samples with the
 * configuration's temperature, the variance of the noise of each latent, and the official steps of its flow head, and it
 * decides as it speaks when the speech ends, so it takes max_seconds, without a bound of its own, since a request cuts a
 * long text into chunks. A text may have any number of tokens, which max_text_tokens gives as 2^53 - 1.
 */
FamilyInfo describe_pocket_tts(const std::shared_ptr<const ModelFile> & file) {
    const ModelFile & m = *file;
    FamilyInfo info;
    info.incremental = true;
    info.voice_codec = m.str("pocket-tts.mimi.encoder_sha256");
    info.voice_recordings = true;
    info.max_text_tokens = (size_t) kMaxSeed;
    info.options = {
        {SPEECH_OPT_VOICE, true, true},
        {SPEECH_OPT_LANGUAGE, false, false, std::string("auto")},
        {SPEECH_OPT_SEED, false, true, std::nullopt, 0, (double) kMaxSeed},
        {SPEECH_OPT_STEPS, false, true, (int64_t) m.u32("pocket-tts.sampler.default_steps"), 1, (double) INT_MAX},
        {SPEECH_OPT_MAX_SECONDS, false, true, std::nullopt, 0, INFINITY, true},
        {SPEECH_OPT_TEMPERATURE, false, true, shortest(m.f64("pocket-tts.sampler.temperature")), 0, std::numeric_limits<float>::max()},
    };
    const auto counter = std::make_shared<Lazy<TokenCounter>>(file);
    info.count_tokens = [counter](const std::string & text) { return counter->get().count(text); };
    return info;
}

std::unique_ptr<Engine> load_pocket_tts(const std::string & path, ggml_backend_t backend, int) {
    return std::make_unique<PocketTtsEngine>(path, backend);
}

void make_pocket_tts_voice(const std::string & model_path, const VoiceRecipe & recipe, const std::string & voice_path, ggml_backend_t backend) {
    if (!recipe.embedding.empty() || !recipe.normalize || recipe.lufs) {
        throw Error(Fault::InvalidArgument, "Pocket TTS takes its recording as it is, without an embedding or a loudness; leave them out",
                    !recipe.embedding.empty() ? "embedding" : "loudness");
    }
    // The official code speaks in the voice of one recording, read as it is.
    if (recipe.parts.size() != 1 || recipe.parts[0].voice_file || recipe.parts[0].transcript || recipe.parts[0].speaker) {
        throw Error(Fault::InvalidArgument, "Pocket TTS makes a voice of one WAVE file, without a transcript or a speaker; give it one recording",
                    "references");
    }
    pocket::make_voice_file(model_path, recipe.parts[0].path, voice_path, backend);
}
