"""Runs the official Pocket TTS on the CPU in float32 and saves the tensors the C++ port is checked against.

usage: uv run python dump.py <english|japanese> <out dir> <text> <reference.wav> [--seed n] [--steps n]
                             [--temperature t]

The official TTSModel speaks `text` in the voice of `reference.wav` once, get_state_for_audio_prompt() and then
generate_audio(), with its stages wrapped, so that what is saved is what it computed. --steps is the official
sampler_decode_steps (1 by default) and --temperature its temp (the configuration's 0.3 by default); the noise is drawn
by torch after torch.manual_seed(seed). The Mimi decoder runs one frame a call (max_decoder_frames_per_call = 1), so that
its schedule does not depend on how fast the threads run. Writes <out dir>/*.npy and meta.json:

  ref_wav              the reference as read, mono float32 at its own rate
  ref_resampled        the reference at 24 kHz, through scipy's resample_poly where its rate differs
  ref_prompt           after end_on_pause(): the trailing silence cut, faded out and 80 ms of zeros appended
  enc_conv_in          the SEANet encoder's first convolution, [64, samples]
  enc_blocks           the output of each encoder block after its strided convolution, a list as enc_block{i}
  enc_seanet           the SEANet encoder's output, [512, T * 16] at the encoder's rate
  enc_transformer      the encoder transformer's output, [512, T * 16]
  latent               the voice's latents, [T, 32], which a voice file keeps
  voice_prompt         the voice prompt the flow LM reads, bos_before_voice then speaker_proj(latent), [1 + T, 1024]
  voice_kv             the flow LM's keys and values after the voice prompt, [layers, 2, 1 + T, heads, head_dim]

and for each chunk c of the text, as split_into_best_sentences() cuts it:

  chunk{c}_tokens      the chunk's tokens after prepare_text_prompt(), [N]
  chunk{c}_text_embd   their rows of the lookup table, [N, 1024]
  chunk{c}_prefill_out the flow LM's normed output at the last token, [1024], from which the official code draws a
                       latent it does not keep
  chunk{c}_prefill_noise the noise of that draw, [32]
  chunk{c}_lm_in       the latent each step reads, NaN for the first (the start of speech), [S, 32]
  chunk{c}_lm_out      the flow LM's normed output of each step, [S, 1024]
  chunk{c}_eos         each step's end-of-speech logit, [S]
  chunk{c}_noise       each step's noise, [S, 32]
  chunk{c}_flow_t      the start and target times of each step's flow head calls, [S, steps, 2]
  chunk{c}_flow_x      each flow head call's input, [S, steps, 32], and chunk{c}_flow_out its output
  chunk{c}_latent      each step's latent, [S, 32]; the last step, the one at which generation stops, is not decoded
  chunk{c}_dec_latent  the latents the decoder reads, denormalized by emb_std and emb_mean, [F, 32]
  chunk{c}_dec_quant   the quantizer's output projection, [512, F]
  chunk{c}_dec_up      the depthwise upsampling, [512, F * 16]
  chunk{c}_dec_transformer the decoder transformer's output, [512, F * 16]
  chunk{c}_dec_conv_in the SEANet decoder's first convolution, [512, F * 16]
  chunk{c}_dec_block{i} the output of each decoder block after its residual units
  chunk{c}_dec_out     the decoder's samples before the fade-in, [F * 1920]
  chunk{c}_audio       the chunk's samples as generate_audio() passes them, after the fade-in of its first 5 ms
  audio                the samples of the whole text, the chunks joined
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch
from pocket_tts import TTSModel
from pocket_tts.data import audio as audio_module
from pocket_tts.data import audio_utils
from pocket_tts.models import tts_model as tts_module

from pins import CODE, CONFIGS, model

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(CONFIGS))
parser.add_argument("out_dir")
parser.add_argument("text")
parser.add_argument("reference")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--temperature", type=float, default=None)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

torch.set_num_threads(1)
pinned = model(args.model)
tts = TTSModel.load_model(config=pinned["config_path"], temp=args.temperature)
if args.steps is not None:
    tts.sampler_decode_steps = args.steps
tts.max_decoder_frames_per_call = 1
tts.eval()
flow_lm, mimi = tts.flow_lm, tts.mimi

arrays = {}


def save(name, value):
    value = value.detach().float().numpy() if isinstance(value, torch.Tensor) else np.asarray(value, dtype=np.float32)
    arrays[name] = np.ascontiguousarray(value, dtype=np.float32)


# Every call of each wrapped module, in order: its inputs and its output.
calls = {}


def kept(value):
    """A copy of a hooked value but for the streaming states it holds, since the official code changes some outputs in
    place, as the fade-in of a chunk's first samples does to the decoder's."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(kept(v) for v in value)
    if isinstance(value, dict):
        # A module's streaming state, its keys and values among them, which nothing here reads.
        return {k: kept(v) for k, v in value.items() if not isinstance(v, dict)}
    return value


def record(name, module):
    def hook(_module, inputs, kwargs, output):
        calls.setdefault(name, []).append({"inputs": kept(inputs), "kwargs": kept(kwargs), "output": kept(output)})

    module.register_forward_hook(hook, with_kwargs=True)


def take(name):
    return calls.pop(name, [])


# The voice: the reference as audio_read() reads it, resampled and ended on a pause as the official code does, then
# encoded through the wrapped Mimi encoder and prompted through the flow LM.
seen_audio = {}
official_read, official_convert, official_pause = audio_module.audio_read, audio_utils.convert_audio, audio_utils.end_on_pause


def read(path):
    wav, rate = official_read(path)
    seen_audio["wav"], seen_audio["rate"] = wav, rate
    return wav, rate


def convert(wav, from_rate, to_rate, channels):
    out = official_convert(wav, from_rate, to_rate, channels)
    seen_audio["resampled"] = out
    return out


def pause(wav, rate, *rest, **kwargs):
    out = official_pause(wav, rate, *rest, **kwargs)
    seen_audio["prompt"] = out
    return out


tts_module.audio_read, tts_module.convert_audio, tts_module.end_on_pause = read, convert, pause

record("enc_conv_in", mimi.encoder.model[0])
block_ends = [i + 1 for i, layer in enumerate(mimi.encoder.model) if isinstance(layer, torch.nn.ELU)][:-1]
for b, index in enumerate(block_ends):
    record(f"enc_block{b}", mimi.encoder.model[index])
record("enc_seanet", mimi.encoder)
record("enc_transformer", mimi.encoder_transformer)
record("latent", mimi.downsample)
record("lm", flow_lm)
record("lm_out_norm", flow_lm.out_norm)
record("lm_eos", flow_lm.out_eos)
record("flow", flow_lm.flow_net)
record("text_embd", flow_lm.conditioner)

voice_state = tts.get_state_for_audio_prompt(os.path.abspath(args.reference))
save("ref_wav", seen_audio["wav"][0])
save("ref_resampled", seen_audio["resampled"][0])
save("ref_prompt", seen_audio["prompt"][0])
save("enc_conv_in", take("enc_conv_in")[0]["output"][0])
for b in range(len(block_ends)):
    save(f"enc_block{b}", take(f"enc_block{b}")[0]["output"][0])
save("enc_seanet", take("enc_seanet")[0]["output"][0])
save("enc_transformer", take("enc_transformer")[0]["output"][0][0])
latent = take("latent")[0]["output"][0].transpose(0, 1)
save("latent", latent)
voice_call = take("lm")
assert len(voice_call) == 1
prompt = voice_call[0]["kwargs"]["text_embeddings"][0]
save("voice_prompt", prompt)
assert torch.equal(prompt[1:], torch.nn.functional.linear(latent, flow_lm.speaker_proj_weight))
for name in ("lm_out_norm", "lm_eos", "flow", "text_embd"):
    take(name)
kv = [voice_state[f"transformer.layers.{i}.self_attn"]["cache"][:, 0] for i in range(len(flow_lm.transformer.layers))]
save("voice_kv", torch.stack(kv))
voice_positions = int(voice_state["transformer.layers.0.self_attn"]["offset"][0])
assert voice_positions == prompt.shape[0]

# The text: the chunks split_into_best_sentences() cuts and what prepare_text_prompt() gives each, through the wrapped
# functions of the official module.
chunks = []
official_split, official_prepare = tts_module.split_into_best_sentences, tts_module.prepare_text_prompt


def split(*a, **kw):
    out = official_split(*a, **kw)
    chunks.extend({"text": text} for text in out)
    return out


def prepare(text, *a, **kw):
    out = official_prepare(text, *a, **kw)
    chunk = next(c for c in chunks if c["text"] == text and "prepared" not in c)
    chunk["prepared"], chunk["frames_after_eos_guess"] = out
    return out


tts_module.split_into_best_sentences, tts_module.prepare_text_prompt = split, prepare

record("dec_latent", mimi.quantizer)
record("dec_up", mimi.upsample)
record("dec_transformer", mimi.decoder_transformer)
record("dec_conv_in", mimi.decoder.model[0])
dec_ends = [i for i, layer in enumerate(mimi.decoder.model) if isinstance(layer, torch.nn.ELU)][1:]
for b, index in enumerate(dec_ends):
    record(f"dec_block{b}", mimi.decoder.model[index - 1])
record("dec_out", mimi.decoder)
chunk_audio = []
official_stream = tts._generate_audio_stream_short_text


def stream_chunk(**kw):
    out = list(official_stream(**kw))
    chunk_audio.append(torch.cat(out))
    return iter(out)


tts._generate_audio_stream_short_text = stream_chunk
torch.manual_seed(args.seed)
audio = tts.generate_audio(voice_state, args.text)
save("audio", audio)

# The calls of each chunk: the text's prefill, then one per step; the decoder's calls, one per decoded frame.
lm_calls, norm_calls, eos_calls, flow_calls = take("lm"), take("lm_out_norm"), take("lm_eos"), take("flow")
embd_calls = [e for e in take("text_embd") if e["inputs"][0].shape[1] > 0]
dec_calls = {name: take(name) for name in ["dec_latent", "dec_up", "dec_transformer", "dec_conv_in", "dec_out"] +
             [f"dec_block{b}" for b in range(len(dec_ends))]}
steps = tts.sampler_decode_steps
call, frame = 0, 0
for c, chunk in enumerate(chunks):
    assert torch.equal(flow_lm.conditioner.prepare(chunk["prepared"]), embd_calls[c]["inputs"][0])
    tokens = embd_calls[c]["inputs"][0][0]
    save(f"chunk{c}_tokens", tokens)
    save(f"chunk{c}_text_embd", embd_calls[c]["output"][0])
    assert lm_calls[call]["kwargs"]["sequence"].shape[1] == 0
    save(f"chunk{c}_prefill_out", norm_calls[call]["output"][0, -1])
    save(f"chunk{c}_prefill_noise", flow_calls[call * steps]["inputs"][-1][0])
    call += 1
    first = call
    while call < len(lm_calls) and lm_calls[call]["kwargs"]["sequence"].shape[1] == 1:
        call += 1
    rows = range(first, call)
    save(f"chunk{c}_lm_in", torch.stack([lm_calls[i]["kwargs"]["sequence"][0, 0] for i in rows]))
    save(f"chunk{c}_lm_out", torch.stack([norm_calls[i]["output"][0, -1] for i in rows]))
    save(f"chunk{c}_eos", torch.stack([eos_calls[i]["output"][0, 0] for i in rows]))
    flows = [[flow_calls[i * steps + k] for k in range(steps)] for i in rows]
    save(f"chunk{c}_noise", torch.stack([f[0]["inputs"][-1][0] for f in flows]))
    save(f"chunk{c}_flow_t", torch.tensor([[[float(f[k]["inputs"][1][0, 0]), float(f[k]["inputs"][2][0, 0])] for k in range(steps)] for f in flows]))
    save(f"chunk{c}_flow_x", torch.stack([torch.stack([f[k]["inputs"][-1][0] for k in range(steps)]) for f in flows]))
    save(f"chunk{c}_flow_out", torch.stack([torch.stack([f[k]["output"][0] for k in range(steps)]) for f in flows]))
    latents = torch.stack([lm_calls[i]["output"][0][0] for i in rows])
    save(f"chunk{c}_latent", latents)
    frames = chunk_audio[c].shape[0] // mimi.frame_size
    assert frames in (len(rows) - 1, len(rows))
    chunk["steps"], chunk["frames"] = len(rows), frames
    stage = {name: [dec_calls[name][frame + k]["output"] for k in range(frames)] for name in dec_calls}
    dec_in = torch.cat([dec_calls["dec_latent"][frame + k]["inputs"][0] for k in range(frames)], dim=-1)[0]
    save(f"chunk{c}_dec_latent", dec_in.transpose(0, 1))
    assert torch.allclose(dec_in.transpose(0, 1), latents[:frames] * flow_lm.emb_std + flow_lm.emb_mean)
    save(f"chunk{c}_dec_quant", torch.cat(stage["dec_latent"], dim=-1)[0])
    save(f"chunk{c}_dec_up", torch.cat(stage["dec_up"], dim=-1)[0])
    save(f"chunk{c}_dec_transformer", torch.cat([o[0] for o in stage["dec_transformer"]], dim=-1)[0])
    save(f"chunk{c}_dec_conv_in", torch.cat(stage["dec_conv_in"], dim=-1)[0])
    for b in range(len(dec_ends)):
        save(f"chunk{c}_dec_block{b}", torch.cat(stage[f"dec_block{b}"], dim=-1)[0])
    save(f"chunk{c}_dec_out", torch.cat(stage["dec_out"], dim=-1)[0, 0])
    save(f"chunk{c}_audio", chunk_audio[c])
    frame += frames
assert call == len(lm_calls) and frame == len(dec_calls["dec_out"])
assert torch.equal(audio, torch.cat(chunk_audio))

for name, value in arrays.items():
    np.save(os.path.join(args.out_dir, name + ".npy"), value)
reference_bytes = open(args.reference, "rb").read()
weights = pinned["config"].weights_path.removeprefix("hf://")
meta = {
    "torch": torch.__version__,
    "code": CODE,
    "model": {"name": args.model, "repository": "/".join(weights.split("/")[:2]), "revision": weights.rsplit("@", 1)[1]},
    "config": CONFIGS[args.model],
    "weights": pinned["config"].weights_path,
    "text": args.text,
    "reference": {"file": os.path.basename(args.reference), "sha256": hashlib.sha256(reference_bytes).hexdigest(),
                  "sample_rate": int(seen_audio["rate"]), "samples": int(seen_audio["wav"].shape[-1])},
    "seed": args.seed,
    "steps": steps,
    "temperature": tts.temp,
    "voice_positions": voice_positions,
    "frames": int(latent.shape[0]),
    "sample_rate": int(mimi.sample_rate),
    "chunks": [{key: chunk[key] for key in ("text", "prepared", "frames_after_eos_guess", "steps", "frames")} for chunk in chunks],
}
with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print("wrote", len(arrays), "arrays to", args.out_dir, json.dumps(meta["chunks"], ensure_ascii=False))
