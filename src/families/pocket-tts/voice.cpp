#include "voice.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "backend.h"
#include "error.h"
#include "gguf.h"
#include "layout.h"
#include "resample.h"
#include "wav.h"

namespace pocket {

namespace {

const Storage kF32 = float32_storage();

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

}  // namespace

Layout voice_layout(const ModelFile & model) {
    return {kVoiceArchitecture, kVoiceLayout, "make it again from its WAVE file with speech voice MODEL REFERENCE.wav VOICE.gguf",
            [&model](const ModelFile & m) {
                const std::string p = "pocket-tts-voice.";
                const std::string encoder = m.str(p + "encoder_sha256"), own = model.str("pocket-tts.mimi.encoder_sha256");
                require(encoder.size() == 64, m, p + "encoder_sha256 is not a SHA-256 in hexadecimal");
                if (encoder != own) {
                    throw Error(Fault::InvalidArgument, m.path() + " was made with the Mimi encoder of SHA-256 " + encoder + ", and " + model.str("general.name") +
                                                            " has the encoder " + own + "; make the voice again from its WAVE file with this model");
                }
                require(m.f64(p + "reference.seconds") > 0 && m.u32(p + "reference.sample_rate") > 0, m, "the voice file gives its recording no length or no rate");
                m.one_of(p + "device_kind", {"cpu", "gpu", "igpu"});
                return std::vector<TensorSpec>{{"latent", {model.width("lm.bos", 0), m.width("latent", 1)}, kF32}};
            }};
}

ReferenceRules::ReferenceRules(const ModelFile & m)
    : max_seconds(m.f64("pocket-tts.reference.max_seconds")),
      frame_seconds(m.f64("pocket-tts.reference.frame_seconds")),
      pause_seconds(m.f64("pocket-tts.reference.pause_seconds")),
      fade_seconds(m.f64("pocket-tts.reference.fade_seconds")),
      floor_db(m.f64("pocket-tts.reference.floor_db")) {}

std::vector<float> linspace(float start, float end, int n) {
    std::vector<float> out((size_t) std::max(n, 0));
    if (n == 1) out[0] = start;
    if (n < 2) return out;
    const float step = (end - start) / (float) (n - 1);
    for (int i = 0; i < n; i++) out[(size_t) i] = i < n / 2 ? start + step * (float) i : end - step * (float) (n - 1 - i);
    return out;
}

std::vector<float> end_on_pause(const std::vector<float> & audio, int sample_rate, const ReferenceRules & r) {
    const int64_t frame = std::max<int64_t>(1, (int64_t) (r.frame_seconds * sample_rate));
    const int64_t frames = (int64_t) audio.size() / frame;
    if (frames == 0) return audio;
    // The official code measures each frame's RMS and its decibels in float32.
    std::vector<float> db((size_t) frames);
    for (int64_t f = 0; f < frames; f++) {
        double sum = 0;
        for (int64_t i = f * frame; i < (f + 1) * frame; i++) sum += (double) (audio[(size_t) i] * audio[(size_t) i]);
        const float rms = std::sqrt((float) (sum / (double) frame));
        db[(size_t) f] = 20.0f * std::log10(rms + 1e-12f);
    }
    const float threshold = *std::max_element(db.begin(), db.end()) - (float) r.floor_db;
    int64_t last = frames - 1;
    while (!(db[(size_t) last] > threshold)) last--;
    const int64_t end = (last + 1) * frame;
    std::vector<float> out(audio.begin(), audio.begin() + end);
    const int64_t fade = std::min<int64_t>((int64_t) (r.fade_seconds * sample_rate), end);
    const std::vector<float> ramp = linspace(1.0f, 0.0f, (int) fade);
    for (int64_t i = 0; i < fade; i++) out[(size_t) (end - fade + i)] *= ramp[(size_t) i];
    out.resize(out.size() + (size_t) (r.pause_seconds * sample_rate), 0.0f);
    return out;
}

Reading read_reference(const std::string & path, int rate, const ReferenceRules & rules, const std::string & input) {
    return naming(input, [&] {
        const Wav wav = read_wav(path);
        std::vector<float> mono = wav.mono();
        const double seconds = (double) mono.size() / wav.sample_rate;
        if (mono.empty()) throw Error(Fault::InvalidArgument, path + " holds no audio; give a recording of the voice");
        if (seconds > rules.max_seconds) {
            throw Error(Fault::OutOfRange, path + " is " + std::to_string(seconds) + " s long, and a Pocket TTS voice is made of at most " +
                                               std::to_string(rules.max_seconds) + " s; give a shorter recording");
        }
        std::vector<float> audio = [&] {
            try {
                return Resampler(wav.sample_rate, rate)(std::move(mono));
            } catch (const std::invalid_argument & e) {
                throw Error(Fault::InvalidArgument, path + ": " + e.what());
            }
        }();
        return Reading{end_on_pause(audio, rate, rules), seconds, wav.sample_rate};
    });
}

Voice load_voice(const std::string & path, const ModelFile & model, Mimi & mimi, ggml_backend_t backend) {
    Voice v;
    if (starts_with_riff(path)) {
        v.latent = mimi.encode(read_reference(path, mimi.sample_rate(), ReferenceRules(model), "path").audio);
    } else {
        const ModelFile file(path, backend, voice_layout(model));
        v.latent = Graph::read(file.tensor("latent"));
    }
    v.frames = (int64_t) v.latent.size() / mimi.latent_dim();
    return v;
}

void make_voice_file(const std::string & model_path, const std::string & reference_path, const std::string & voice_path, ggml_backend_t backend) {
    // Mimi::encode() reads these tensors and no others.
    const auto model = naming("model_path", [&] {
        return std::make_unique<const ModelFile>(model_path, backend, model_layout, [](const std::string & name) {
            return name.compare(0, 8, "mimi.enc") == 0 || name.compare(0, 9, "mimi.down") == 0;
        });
    });
    Mimi mimi(*model, backend);
    const Reading reading = read_reference(reference_path, mimi.sample_rate(), ReferenceRules(*model), "reference_path");
    const std::vector<float> latent = mimi.encode(reading.audio);
    const size_t bytes = latent.size() * sizeof(float);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ggml_tensor_overhead() + bytes + 64, nullptr, false}), ggml_free);
    if (!ctx) throw Error(Fault::OutOfMemory, "cannot hold the latent of a voice file in memory");
    ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, mimi.latent_dim(), (int64_t) latent.size() / mimi.latent_dim());
    ggml_set_name(t, "latent");
    std::memcpy(t->data, latent.data(), bytes);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> g(gguf_init_empty(), gguf_free);
    gguf_set_val_str(g.get(), "general.architecture", kVoiceArchitecture);
    gguf_set_val_u32(g.get(), "speech.layout", kVoiceLayout);
    gguf_set_val_str(g.get(), "speech.requires", kVoiceLayoutRequires);
    gguf_set_val_str(g.get(), "pocket-tts-voice.encoder_sha256", mimi.encoder_sha256().c_str());
    gguf_set_val_f64(g.get(), "pocket-tts-voice.reference.seconds", reading.seconds);
    gguf_set_val_u32(g.get(), "pocket-tts-voice.reference.sample_rate", (uint32_t) reading.sample_rate);
    gguf_set_val_str(g.get(), "pocket-tts-voice.device_kind", device_kind(backend).c_str());
    gguf_add_tensor(g.get(), t);
    naming("voice_path", [&] {
        if (!gguf_write_to_file(g.get(), voice_path.c_str(), false)) {
            throw Error(Fault::Io, "cannot write " + voice_path + "; check that its folder exists and is writable");
        }
    });
}

}  // namespace pocket
