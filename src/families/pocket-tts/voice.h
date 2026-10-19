#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "mimi.h"
#include "model-file.h"

namespace pocket {

/** The general.architecture of a Pocket TTS voice file. */
constexpr const char * kVoiceArchitecture = "pocket-tts-voice";
constexpr uint32_t kVoiceLayout = 1;
/** The release whose reader first takes voice files of kVoiceLayout, which a voice file names as speech.requires. */
constexpr const char * kVoiceLayoutRequires = "0.8.11";

/**
 * The layout of a voice file for the model file `model`: the latent of one recording, [latent_dim, frames], bound by its
 * hash to the Mimi encoder that made it, which a model of another encoder refuses.
 */
Layout voice_layout(const ModelFile & model);

/** A voice: the latent of its recording as Mimi's encoder gives it, row-major [frames, latent_dim]. */
struct Voice {
    std::vector<float> latent;
    int64_t frames = 0;
};

/** What the official code does to a recording before Mimi encodes it, from the model file's pocket-tts.reference.*. */
struct ReferenceRules {
    explicit ReferenceRules(const ModelFile & m);
    double max_seconds, frame_seconds, pause_seconds, fade_seconds, floor_db;
};

/**
 * end_on_pause(): `audio` at `sample_rate` cut after its last frame no more than floor_db below its loudest, frames of
 * frame_seconds, its last fade_seconds faded out, and pause_seconds of zeros appended, so that the voice prompt ends on
 * a short pause as the model's training prompts did. Audio shorter than a frame is left as it is.
 */
std::vector<float> end_on_pause(const std::vector<float> & audio, int sample_rate, const ReferenceRules & rules);

/**
 * torch.linspace(start, end, n) in float32 as torch computes it: the first half from start, the second back from end.
 */
std::vector<float> linspace(float start, float end, int n);

/** A recording read for a voice: its audio ready for Mimi, its length and its rate. */
struct Reading {
    std::vector<float> audio;
    double seconds = 0;
    int sample_rate = 0;
};

/**
 * The recording in the WAVE file at `path` as a voice reads it: its channels averaged, resampled to `rate` and ended
 * on a pause. A recording longer than max_seconds is refused, naming `input`.
 */
Reading read_reference(const std::string & path, int rate, const ReferenceRules & rules, const std::string & input);

/** The voice of a WAVE file, encoded by `mimi`, or of a voice file made for `model`. */
Voice load_voice(const std::string & path, const ModelFile & model, Mimi & mimi, ggml_backend_t backend);

/**
 * Makes a voice file at `voice_path` for the model file at `model_path` from the recording in the WAVE file at
 * `reference_path`, reading from the model file Mimi's encoder alone and running it on `backend`.
 */
void make_voice_file(const std::string & model_path, const std::string & reference_path, const std::string & voice_path, ggml_backend_t backend);

}  // namespace pocket
