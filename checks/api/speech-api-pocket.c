/*
 * The checks of speech-api-check that Pocket TTS alone takes: max_seconds without a bound of its own, a temperature of 0
 * that draws no noise, the steps of the flow head, a long text spoken in chunks in one request, and voice files of one
 * recording, with what a Pocket TTS voice does not take refused.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "speech-api-check.h"

/** Whether `a` and `b` hold the same samples, bit for bit. */
static int same_samples(const Audio * a, const Audio * b) {
    return a->n == b->n && memcmp(a->samples, b->samples, a->n * sizeof(float)) == 0;
}

/** Writes `seconds` of a quiet tone, 16-bit mono at 24 kHz, to `path`; whether it was written. */
static int write_tone(const char * path, double seconds) {
    FILE * f = open_utf8(path, "wb");
    if (!f) return 0;
    const unsigned rate = 24000, n = (unsigned) (seconds * rate), bytes = 2 * n;
    const unsigned char header[44] = {'R', 'I', 'F', 'F', (unsigned char) (36 + bytes), (unsigned char) ((36 + bytes) >> 8),
                                      (unsigned char) ((36 + bytes) >> 16), (unsigned char) ((36 + bytes) >> 24), 'W', 'A', 'V', 'E',
                                      'f', 'm', 't', ' ', 16, 0, 0, 0, 1, 0, 1, 0, 0xC0, 0x5D, 0, 0, 0x80, 0xBB, 0, 0, 2, 0, 16, 0,
                                      'd', 'a', 't', 'a', (unsigned char) bytes, (unsigned char) (bytes >> 8), (unsigned char) (bytes >> 16),
                                      (unsigned char) (bytes >> 24)};
    int ok = fwrite(header, 1, sizeof header, f) == sizeof header;
    for (unsigned i = 0; ok && i < n; i++) {
        const int16_t s = (int16_t) (3000 * sin(2 * 3.141592653589793 * 220 * i / rate));
        const unsigned char le[2] = {(unsigned char) (s & 0xFF), (unsigned char) ((uint16_t) s >> 8)};
        ok = fwrite(le, 1, 2, f) == 2;
    }
    return fclose(f) == 0 && ok;
}

int check_pocket_voices(const char * model_path, const char * reference, const char * made, const speech_load_params * params) {
    char path[4096], long_wav[4096];
    int ok = 1;
    speech_voice_params * v = NULL;
    if (speech_voice_params_new(&v) != SPEECH_OK) return fail("speech_voice_params_new");
    snprintf(path, sizeof path, "%s.from.gguf", made);
    ok &= expect(speech_voice_params_add_reference(v, reference), SPEECH_OK, NULL, "a reference") &&
          expect(speech_voice_make_from(model_path, v, path, params), SPEECH_OK, NULL, "a voice of one reference");
    ok &= expect(speech_voice_params_add_reference(v, reference), SPEECH_OK, NULL, "a second reference") &&
          expect(speech_voice_make_from(model_path, v, path, params), SPEECH_ERROR_INVALID_ARGUMENT, "references", "a voice of two references");
    speech_voice_params_free(v);
    if (speech_voice_params_new(&v) != SPEECH_OK) return fail("speech_voice_params_new");
    speech_voice_params_add_reference(v, reference);
    ok &= expect(speech_voice_params_set_loudness(v, -23), SPEECH_OK, NULL, "a loudness of -23 LUFS") &&
          expect(speech_voice_make_from(model_path, v, path, params), SPEECH_ERROR_INVALID_ARGUMENT, "loudness", "a voice at -23 LUFS");
    ok &= expect(speech_voice_params_keep_loudness(v), SPEECH_OK, NULL, "the loudness kept") &&
          expect(speech_voice_make_from(model_path, v, path, params), SPEECH_ERROR_INVALID_ARGUMENT, "loudness", "a voice of the loudness kept");
    speech_voice_params_free(v);
    float values[8 * 32];
    for (int i = 0; i < 8 * 32; i++) values[i] = 0.5f * sinf(0.01f * (float) i);
    if (speech_voice_params_new(&v) != SPEECH_OK) return fail("speech_voice_params_new");
    ok &= expect(speech_voice_params_set_embedding(v, values, 8, 32), SPEECH_OK, NULL, "an embedding") &&
          expect(speech_voice_make_from(model_path, v, path, params), SPEECH_ERROR_INVALID_ARGUMENT, "embedding", "a voice of an embedding");
    speech_voice_params_free(v);
    // A recording of 31 s, past the 30 s a voice is made of.
    snprintf(long_wav, sizeof long_wav, "%s.long.wav", made);
    ok &= write_tone(long_wav, 31) &&
          expect(speech_voice_make(model_path, long_wav, path, params), SPEECH_ERROR_OUT_OF_RANGE, "reference_path", "a recording of 31 s");
    remove(long_wav);
    remove(path);
    if (ok) printf("a voice of one recording was made, and of two, of a loudness, of an embedding and of 31 s refused\n");
    else fprintf(stderr, "FAIL: a Pocket TTS voice is not made of one recording alone, as it is\n");
    return ok ? 0 : 1;
}

int check_pocket(speech_model * model, const speech_model_info * info, const char * voice) {
    const int rate = speech_model_info_sample_rate(info);
    speech_request * r = new_request(model, "これは最大の長さで止める、少し長めの文です。止まったところで終わります。", voice, 3);
    int ok = expect(speech_request_set_float(r, SPEECH_OPT_MAX_SECONDS, 1e9), SPEECH_OK, NULL, "max_seconds 1e9") &&
             expect(speech_request_set_float(r, SPEECH_OPT_MAX_SECONDS, 0.5), SPEECH_OK, NULL, "max_seconds 0.5");
    Audio audio = {NULL, 0, 0};
    speech_stop stop;
    if (speak(r, &audio, &stop, NULL) != SPEECH_OK) return fail("a request with max_seconds");
    printf("max_seconds 0.5 gave %.3f s of audio, stopped at %s\n", (double) audio.n / rate, speech_stop_name(stop));
    ok &= stop == SPEECH_STOP_MAX_SECONDS && audio.n > 0 && audio.n <= (size_t) (0.5 * rate);
    free(audio.samples);

    // A temperature of 0 draws no noise, so the seed changes nothing.
    Audio a = {NULL, 0, 0}, b = {NULL, 0, 0};
    r = new_request(model, "温度がゼロです。", voice, 1);
    ok &= expect(speech_request_set_float(r, SPEECH_OPT_TEMPERATURE, -0.1), SPEECH_ERROR_OUT_OF_RANGE, "temperature", "a temperature below 0") &&
          expect(speech_request_set_float(r, SPEECH_OPT_TEMPERATURE, 0), SPEECH_OK, NULL, "a temperature of 0");
    if (speak(r, &a, NULL, NULL) != SPEECH_OK) return fail("a request of temperature 0");
    r = new_request(model, "温度がゼロです。", voice, 2);
    speech_request_set_float(r, SPEECH_OPT_TEMPERATURE, 0);
    if (speak(r, &b, NULL, NULL) != SPEECH_OK) return fail("a request of temperature 0 and another seed");
    ok &= a.n > 0 && same_samples(&a, &b);
    printf("a temperature of 0 gave the same %zu samples with two seeds\n", a.n);
    free(a.samples);
    free(b.samples);

    Audio stepped = {NULL, 0, 0};
    r = new_request(model, "二歩で描きます。", voice, 4);
    ok &= expect(speech_request_set_int(r, SPEECH_OPT_STEPS, 0), SPEECH_ERROR_OUT_OF_RANGE, "steps", "steps 0") &&
          expect(speech_request_set_int(r, SPEECH_OPT_STEPS, 2), SPEECH_OK, NULL, "steps 2") &&
          expect(speak(r, &stepped, NULL, NULL), SPEECH_OK, NULL, "a request of 2 steps") && stepped.n > 0;
    free(stepped.samples);

    // A text of several chunks of at most 50 tokens, spoken in one request.
    char text[2048] = "";
    for (int i = 0; i < 6; i++) strcat(text, SENTENCE);
    size_t tokens = 0;
    Audio paragraph = {NULL, 0, 0};
    if (speech_model_info_text_tokens(info, text, &tokens) != SPEECH_OK) return fail("speech_model_info_text_tokens");
    if (speak(new_request(model, text, voice, 5), &paragraph, &stop, NULL) != SPEECH_OK) return fail("a text of several chunks");
    printf("a text of %zu tokens gave %.2f s of audio, stopped at %s\n", tokens, (double) paragraph.n / rate, speech_stop_name(stop));
    ok &= tokens > 50 && paragraph.n > 0 && stop != SPEECH_STOP_MAX_SECONDS;
    free(paragraph.samples);
    if (!ok) fprintf(stderr, "FAIL: Pocket TTS's rules of a request are not followed\n");
    return ok ? 0 : 1;
}
