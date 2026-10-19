"""Converts a pinned Pocket TTS model, english or japanese, with its Mimi codec to the one GGUF file the C++ port reads.

usage: uv run python convert.py <english|japanese> <out dir>

Writes the model's file, named under GGUF's naming convention from its identity and the parameters of its tensors, in
layout 1: the tokenizer and the rules the official code applies to a text before it, the flow language model (the
lookup table of the text's tokens, the causal transformer and the end-of-speech head) with its LSD flow head, and Mimi,
the codec, whose encoder makes a voice from a recording and whose decoder turns the latents into samples, with every
constant the C++ reads, the hash of the encoder's tensors, which voice files carry, and the model's identity in the
GGUF specification's general keys.

The tensors are those of the official TTSModel as load_model() builds it from the pinned configuration, in float32: the
checkpoint holds bfloat16, which the model loads into float32 exactly. Tensor shapes follow ggml, whose ne[0] is the
last numpy axis: a Linear weight [out, in] is stored as is (ne = [in, out]). `speech quantize` makes the other weight
types of the file, each tensor in the type src/families/pocket-tts/layout.cpp gives it.

A convolution's weight [out, in, k] is stored as numpy [k, out, in] (ne = [in, out, k]), so that tap k is a plain
[in, out] matrix: the C++ side runs a convolution as one matrix product per tap on channel-first activations. A
transposed convolution's weight [in, out, k] is stored the same way as [k, out, in]. A strided convolution of width 2s
is stored as its two halves, each [out, s * in] with the input channel fastest, so that it runs as two matrix products
on the input cut into frames of s samples. The depthwise transposed convolution that upsamples the latents is stored as
[k, channels].
"""

import argparse
import hashlib
import inspect
import json
import os
import struct

import numpy as np
import torch
import yaml
from gguf import GGUFValueType, GGUFWriter, LlamaFileType, naming_convention, size_label
from pocket_tts import TTSModel
from pocket_tts import default_parameters
from pocket_tts.data import audio_utils
from pocket_tts.models import text_chunking, tts_model
from pocket_tts.modules.text_conditioner import JsonTokenizer, SentencePieceTokenizer
from sentencepiece import sentencepiece_model_pb2

from pins import CARDS, model

ARCH = "pocket-tts"
# Each layout this converter has written, with the first release of speech.cpp whose reader takes it; it writes the
# last.
RELEASES = {1: "0.8.11"}
LAYOUT = max(RELEASES)
# The SPDX identifier of each license a model card names.
LICENSES = {"cc-by-4.0": "CC-BY-4.0", "cc-by-nc-4.0": "CC-BY-NC-4.0"}
# Each model's name under GGUF's naming convention (ggml's docs/gguf.md), besides the size label, which the parameters
# of its tensors give, and the languages it speaks. Each official configuration is one language's model, which its name
# says; the cards list the languages of a whole repository.
NAMES = {"english": {"basename": "Pocket-TTS", "finetune": "English", "languages": ["en"]},
         "japanese": {"basename": "Pocket-TTS", "finetune": "Japanese", "languages": ["ja"]}}

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(NAMES))
parser.add_argument("out_dir")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

pinned = model(args.model)
config = pinned["config"]
tts = TTSModel.load_model(config=pinned["config_path"])
tts.eval()
flow_lm, mimi = tts.flow_lm, tts.mimi
card = yaml.safe_load(open(pinned["card"], encoding="utf-8").read().split("---\n")[1])
weights_url = config.weights_path.removeprefix("hf://")
repository_id = "/".join(weights_url.split("/")[:2])
revision = weights_url.rsplit("@", 1)[1]

# What this reader runs: the LSD flow head, the voice prompt after its own start token, the latents downsampled to the
# frame rate by a convolution into 32 channels, the official defaults of a request and no length of its own.
assert config.flow_lm.flow.type == "lsd" and flow_lm.flow_net.num_time_conds == 2
assert config.flow_lm.insert_bos_before_voice and config.mimi.inner_dim == config.mimi.quantizer.dimension
assert config.model_recommended_frames_after_eos is None
assert default_parameters.DEFAULT_NOISE_CLAMP is None
assert config.mimi.seanet.pad_mode == "constant" and config.mimi.seanet.channels == 1 == config.mimi.channels
assert mimi.encoder_frame_rate != mimi.frame_rate and len(config.mimi.transformer.output_dimensions) == 1
assert config.mimi.transformer.d_model == config.mimi.transformer.input_dimension == config.mimi.seanet.dimension
assert config.mimi.transformer.output_dimensions[0] == config.mimi.seanet.dimension == config.mimi.outer_dim

source = {name: inspect.getsource(module) for name, module in
          (("tts_model", tts_model), ("text_chunking", text_chunking), ("audio_utils", audio_utils))}


def require_source(module, text):
    """Fails unless the official module's source still holds `text`, the line a constant below comes from."""
    assert text in source[module], f"{module} no longer holds {text!r}"


w = GGUFWriter(None, ARCH)


def add_array(key, values, element):
    """An array with its element type given, not inferred from its first value."""
    w.add_key_value(key, list(values), GGUFValueType.ARRAY, sub_type=element)


w.add_uint32("speech.layout", LAYOUT)
w.add_string("speech.requires", RELEASES[LAYOUT])
w.add_string("speech.task", "synthesis")
w.add_uint32("speech.sample_rate", int(config.mimi.sample_rate))
# A Pocket TTS model speaks the one language of its configuration and takes none with a request; a request's language
# is only checked against the model's.
w.add_string("speech.language_use", "checked")

p = f"{ARCH}."
w.add_string(p + "flow", config.flow_lm.flow.type)

# The flow language model: StreamingTransformer's layers normalize with nn.LayerNorm(eps=1e-5), its RoPE turns pairs of
# channels with max_period as its base, and a request's voice prompt starts with bos_before_voice.
layer = flow_lm.transformer.layers[0]
w.add_uint32(p + "lm.num_layers", int(config.flow_lm.transformer.num_layers))
w.add_uint32(p + "lm.num_heads", int(config.flow_lm.transformer.num_heads))
w.add_float32(p + "lm.rope_base", float(config.flow_lm.transformer.max_period))
assert layer.norm1.eps == layer.norm2.eps == flow_lm.out_norm.eps
w.add_float32(p + "lm.norm_eps", float(layer.norm1.eps))
w.add_bool(p + "lm.bos_before_voice", bool(config.flow_lm.insert_bos_before_voice))

# The flow head, SimpleMLPAdaLN: its ResBlocks and FinalLayer normalize with a LayerNorm of eps 1e-6, each time
# embedder with an RMSNorm of eps 1e-5 that divides by the variance of its input.
flow_net = flow_lm.flow_net
w.add_uint32(p + "flow.num_blocks", int(flow_net.num_res_blocks))
assert all(block.in_ln.eps == flow_net.final_layer.norm_final.eps for block in flow_net.res_blocks)
w.add_float32(p + "flow.norm_eps", float(flow_net.final_layer.norm_final.eps))
assert flow_net.time_embed[0].mlp[3].eps == flow_net.time_embed[1].mlp[3].eps
w.add_float32(p + "flow.time_norm_eps", float(flow_net.time_embed[0].mlp[3].eps))

# A request's defaults: the configuration's temperature, the variance of the noise a latent is drawn from, the LSD steps
# and the threshold of the end-of-speech logit of default_parameters.py.
w.add_float32(p + "sampler.temperature", float(config.default_temperature))
w.add_uint32(p + "sampler.default_steps", int(default_parameters.DEFAULT_SAMPLER_DECODE_STEPS))
w.add_float32(p + "sampler.eos_threshold", float(default_parameters.DEFAULT_EOS_THRESHOLD))

# How long a chunk of text may speak and how it ends, from TTSModel: no end before frame 6, at most tokens / 3 + 2 s,
# and after the end a few frames more, 3 for a text of 4 words or fewer and 1 otherwise, each plus 2.
require_source("tts_model", "_MIN_FRAMES_BEFORE_EOS = 6")
require_source("tts_model", "frames_after_eos_guess += 2")
require_source("text_chunking", "if number_of_words <= 4:\n        frames_after_eos_guess = 3")
require_source("text_chunking", "    else:\n        frames_after_eos_guess = 1")
w.add_uint32(p + "generation.min_frames_before_eos", int(TTSModel._MIN_FRAMES_BEFORE_EOS))
w.add_float32(p + "generation.tokens_per_second", float(TTSModel._TOKENS_PER_SECOND_ESTIMATE))
w.add_float32(p + "generation.padding_seconds", float(TTSModel._GEN_SECONDS_PADDING))
w.add_uint32(p + "generation.short_text_words", 4)
w.add_uint32(p + "generation.frames_after_eos_short", 3 + 2)
w.add_uint32(p + "generation.frames_after_eos_long", 1 + 2)
# The decoder of each chunk starts from a fresh state and fades in its first 5 ms, torch.linspace(0, 1, rate // 200).
require_source("tts_model", "torch.linspace(0, 1, self.config.mimi.sample_rate // 200)")
w.add_uint32(p + "generation.fade_in_samples", int(config.mimi.sample_rate) // 200)

# A reference voice: the server and export-voice keep its first 30 s, and end_on_pause() cuts its trailing silence, the
# 20 ms frames more than floor_db below the loudest, fades out its last fade_sec and appends pause_sec of zeros.
require_source("tts_model", "max_samples = int(30 * conditioning_sample_rate)")
require_source("audio_utils", "frame = max(1, int(0.02 * sample_rate))")
pause = inspect.signature(audio_utils.end_on_pause).parameters
w.add_float32(p + "reference.max_seconds", 30.0)
w.add_float32(p + "reference.frame_seconds", 0.02)
w.add_float32(p + "reference.pause_seconds", float(pause["pause_sec"].default))
w.add_float32(p + "reference.fade_seconds", float(pause["fade_sec"].default))
w.add_float32(p + "reference.floor_db", float(pause["floor_db"].default))

# The rules prepare_text_prompt() and split_into_best_sentences() apply to a text, from the configuration, and the
# tokens a chunk of it holds at most, MAX_TOKEN_PER_CHUNK, which generate_audio() takes by default.
w.add_uint32(p + "text.max_tokens", int(default_parameters.MAX_TOKEN_PER_CHUNK))
w.add_bool(p + "text.pad_with_spaces", bool(config.pad_with_spaces_for_short_inputs))
w.add_bool(p + "text.remove_semicolons", bool(config.remove_semicolons))
w.add_bool(p + "text.append_terminal_punctuation", bool(config.append_terminal_punctuation))
w.add_bool(p + "text.capitalize_first_letter", bool(config.capitalize_first_letter))
# str.maketrans() maps one character to a string, "" deleting it. A model whose configuration replaces none has neither
# key, since GGUF has no empty array.
replace = dict(config.replace_characters)
assert all(len(key) == 1 for key in replace)
if replace:
    add_array(p + "text.replace_from", list(replace.keys()), GGUFValueType.STRING)
    add_array(p + "text.replace_to", list(replace.values()), GGUFValueType.STRING)

# The tokenizer, a SentencePiece Unigram model of 4000 pieces with byte fallback, which the official code runs through
# Hugging Face tokenizers from a tokenizer.json (english) or through sentencepiece from a .model (japanese). The two
# split a text differently: tokenizers matches its added tokens in the text, prepends "▁", turns every space into "▁"
# and cuts the text into words before each "▁", each word a Viterbi path of its own; sentencepiece prepends "▁" and turns
# spaces into "▁" too, but runs one Viterbi path over the whole text and matches no control piece in it. The scores are
# written in float64, as each runtime compares their sums: tokenizers parses them from JSON, sentencepiece widens its
# float32 ones.
tokenizer = flow_lm.conditioner.tokenizer
n_bins = int(config.flow_lm.lookup_table.n_bins)
if isinstance(tokenizer, JsonTokenizer):
    spec = json.loads(tokenizer.tokenizer.to_str())
    unigram = spec["model"]
    assert unigram["type"] == "Unigram" and unigram["byte_fallback"]
    assert spec["normalizer"] == {"type": "Prepend", "prepend": "▁"}
    assert spec["pre_tokenizer"] == {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True}
    tokens = [piece for piece, _ in unigram["vocab"]]
    scores = [float(score) for _, score in unigram["vocab"]]
    unknown_id = int(unigram["unk_id"])
    for added in spec["added_tokens"]:
        assert tokens[added["id"]] == added["content"] and not added["normalized"] and added["special"]
        assert not (added["lstrip"] or added["rstrip"] or added["single_word"])
    special_ids = [int(added["id"]) for added in spec["added_tokens"]]
    control_ids = []
    runtime = "tokenizers"
else:
    assert isinstance(tokenizer, SentencePieceTokenizer)
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(tokenizer.sp.serialized_model_proto())
    normalizer, trainer = proto.normalizer_spec, proto.trainer_spec
    assert trainer.model_type == sentencepiece_model_pb2.TrainerSpec.UNIGRAM and trainer.byte_fallback
    assert normalizer.name == "identity" and not normalizer.precompiled_charsmap
    assert normalizer.add_dummy_prefix and normalizer.escape_whitespaces and not normalizer.remove_extra_whitespaces
    assert not trainer.treat_whitespace_as_suffix and not proto.HasField("denormalizer_spec") or not proto.denormalizer_spec.name
    kinds = sentencepiece_model_pb2.ModelProto.SentencePiece
    assert all(piece.type in (kinds.NORMAL, kinds.UNKNOWN, kinds.CONTROL, kinds.BYTE) for piece in proto.pieces)
    assert all((piece.type == kinds.BYTE) == (piece.piece.startswith("<0x") and len(piece.piece) == 6) for piece in proto.pieces)
    tokens = [piece.piece for piece in proto.pieces]
    scores = [float(piece.score) for piece in proto.pieces]
    unknown_id = [i for i, piece in enumerate(proto.pieces) if piece.type == kinds.UNKNOWN][0]
    control_ids = [i for i, piece in enumerate(proto.pieces) if piece.type == kinds.CONTROL]
    special_ids = []
    runtime = "sentencepiece"
assert len(tokens) == n_bins and all(f"<0x{b:02X}>" in tokens for b in range(256))
w.add_string(p + "tokenizer.runtime", runtime)
add_array(p + "tokenizer.tokens", tokens, GGUFValueType.STRING)
add_array(p + "tokenizer.scores", scores, GGUFValueType.FLOAT64)
w.add_uint32(p + "tokenizer.unknown_id", unknown_id)
# The added tokens tokenizers splits out of a text before anything else and skips in decoding, and the control pieces
# sentencepiece matches in no text and decodes to nothing; a file has each key only where its tokenizer has such pieces,
# since GGUF has no empty array.
if special_ids:
    add_array(p + "tokenizer.special_ids", special_ids, GGUFValueType.INT32)
if control_ids:
    add_array(p + "tokenizer.control_ids", control_ids, GGUFValueType.INT32)

# Mimi: SEANet's strides from the waveform's side (the decoder's order, reversed by the encoder), the residual units of
# each block, of dilations dilation_base ** j, the downsampling to the frame rate, and the transformers before and after
# it, which normalize with nn.LayerNorm(eps=1e-5) and attend to the context positions up to their own.
seanet, transformer_config = config.mimi.seanet, config.mimi.transformer
w.add_uint32(p + "mimi.hop_length", int(mimi.frame_size))
add_array(p + "mimi.ratios", [int(r) for r in seanet.ratios], GGUFValueType.INT32)
w.add_uint32(p + "mimi.residual_layers", int(seanet.n_residual_layers))
w.add_uint32(p + "mimi.dilation_base", int(seanet.dilation_base))
w.add_uint32(p + "mimi.downsample_stride", int(mimi.encoder_frame_rate / mimi.frame_rate))
w.add_uint32(p + "mimi.num_layers", int(transformer_config.num_layers))
w.add_uint32(p + "mimi.num_heads", int(transformer_config.num_heads))
w.add_uint32(p + "mimi.context", int(transformer_config.context))
w.add_float32(p + "mimi.rope_base", float(transformer_config.max_period))
mimi_layer = mimi.decoder_transformer.transformer.layers[0]
assert mimi_layer.norm1.eps == mimi_layer.norm2.eps
w.add_float32(p + "mimi.norm_eps", float(mimi_layer.norm1.eps))
assert int(np.prod(seanet.ratios)) * int(mimi.encoder_frame_rate / mimi.frame_rate) == mimi.frame_size


def encoder_sha256():
    """SHA-256 over the tensors of Mimi's encoder, its transformer and its downsampling as the official model holds them,
    which turn a recording into the latents a voice file keeps: for each in ascending order of its name, the name in
    UTF-8, a 0 byte, the number of dimensions as a little-endian u32, each dimension as a little-endian u64, and the
    values as little-endian float32 in row-major order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(mimi.state_dict().items()):
        if not name.startswith(("encoder.", "encoder_transformer.", "downsample.")):
            continue
        values = np.ascontiguousarray(tensor.detach().cpu().float().numpy(), dtype="<f4")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(struct.pack("<I", values.ndim) + struct.pack(f"<{values.ndim}Q", *values.shape))
        digest.update(values.tobytes())
    return digest.hexdigest()


sha256 = encoder_sha256()
w.add_string(p + "mimi.encoder_sha256", sha256)

# The official model's tensors the file holds, every one of them by the end.
state = {name: tensor.detach().float().numpy() for name, tensor in tts.state_dict().items()}
converted = set()


def tensor(key):
    converted.add(key)
    return state[key]


def add(out_name, data):
    w.add_tensor(out_name, np.ascontiguousarray(data, dtype=np.float32))


def linear(prefix_in, out_name):
    add(out_name + ".weight", tensor(prefix_in + ".weight"))
    add(out_name + ".bias", tensor(prefix_in + ".bias"))


def layer_norm(prefix_in, out_name):
    add(out_name + ".weight", tensor(prefix_in + ".weight"))
    add(out_name + ".bias", tensor(prefix_in + ".bias"))


def transformer_layers(prefix_in, prefix_out, layers_n, scaled):
    """A StreamingTransformer's layers: the fused query, key and value projection [3 * d, d] as it is, whose rows hold
    the queries of every head, then the keys, then the values, and the layer scales where the layers have them."""
    for i in range(layers_n):
        a, o = f"{prefix_in}.layers.{i}.", f"{prefix_out}.blk.{i}."
        layer_norm(a + "norm1", o + "attn_norm")
        add(o + "attn_qkv", tensor(a + "self_attn.in_proj.weight"))
        add(o + "attn_out", tensor(a + "self_attn.out_proj.weight"))
        layer_norm(a + "norm2", o + "ffn_norm")
        add(o + "ffn_up", tensor(a + "linear1.weight"))
        add(o + "ffn_down", tensor(a + "linear2.weight"))
        if scaled:
            add(o + "attn_scale", tensor(a + "layer_scale_1.scale"))
            add(o + "ffn_scale", tensor(a + "layer_scale_2.scale"))


# The flow language model. The lookup table has a row past the tokenizer's pieces, the official padding row, which no
# token reads.
f = "flow_lm."
add("lm.text_embd", tensor(f + "conditioner.embed.weight"))
add("lm.bos", tensor(f + "bos_emb"))
add("lm.bos_before_voice", tensor(f + "bos_before_voice").reshape(-1))
add("lm.emb_mean", tensor(f + "emb_mean"))
add("lm.emb_std", tensor(f + "emb_std"))
add("lm.in_proj", tensor(f + "input_linear.weight"))
add("lm.speaker_proj", tensor(f + "speaker_proj_weight"))
transformer_layers(f + "transformer", "lm", int(config.flow_lm.transformer.num_layers), scaled=False)
layer_norm(f + "out_norm", "lm.out_norm")
add("lm.eos.weight", tensor(f + "out_eos.weight").reshape(-1))
add("lm.eos.bias", tensor(f + "out_eos.bias"))

# The flow head. Each time embedder holds its frequencies as the checkpoint rounded them to bfloat16.
h = f + "flow_net."
linear(h + "cond_embed", "flow.cond")
linear(h + "input_proj", "flow.in_proj")
for i in range(flow_net.num_time_conds):
    a, o = f"{h}time_embed.{i}.", f"flow.time.{i}."
    add(o + "freqs", tensor(a + "freqs"))
    linear(a + "mlp.0", o + "fc1")
    linear(a + "mlp.2", o + "fc2")
    add(o + "norm", tensor(a + "mlp.3.alpha"))
for i in range(flow_net.num_res_blocks):
    a, o = f"{h}res_blocks.{i}.", f"flow.blk.{i}."
    layer_norm(a + "in_ln", o + "norm")
    linear(a + "mlp.0", o + "fc1")
    linear(a + "mlp.2", o + "fc2")
    linear(a + "adaLN_modulation.1", o + "ada")
linear(h + "final_layer.adaLN_modulation.1", "flow.final.ada")
linear(h + "final_layer.linear", "flow.final.out")


def conv(prefix_in, out_name, bias=True):
    """A Conv1d of stride 1: weight [out, in, k] stored as [k, out, in]."""
    add(out_name + ".weight", np.transpose(tensor(prefix_in + ".weight"), (2, 0, 1)))
    if bias:
        add(out_name + ".bias", tensor(prefix_in + ".bias"))


def halves(prefix_in, out_name, stride, bias=True):
    """A Conv1d of width 2s and stride s as its two halves, each [out, s * in] with the input channel fastest."""
    weight = tensor(prefix_in + ".weight")
    assert weight.shape[2] == 2 * stride
    add(out_name + ".first", np.transpose(weight[:, :, :stride], (0, 2, 1)).reshape(weight.shape[0], -1))
    add(out_name + ".second", np.transpose(weight[:, :, stride:], (0, 2, 1)).reshape(weight.shape[0], -1))
    if bias:
        add(out_name + ".bias", tensor(prefix_in + ".bias"))


def residual(prefix_in, out_name):
    """A SEANetResnetBlock: ELU, a convolution to half the channels, ELU, a width-1 convolution back, and the skip."""
    conv(prefix_in + ".block.1.conv", out_name + ".conv1")
    conv(prefix_in + ".block.3.conv", out_name + ".conv2")


def check_conv(module, stride, dilation=1):
    assert module.stride[0] == stride and module.dilation[0] == dilation and module.groups == 1


# The encoder, from the waveform to the latents: SEANet with its strides reversed, the transformer, and the strided
# convolution to the frame rate, padded on the left by its first frame repeated.
m = "mimi."
enc = mimi.encoder.model
conv(m + "encoder.model.0.conv", "mimi.enc.conv_in")
index = 1
for i, ratio in enumerate(mimi.encoder.ratios):
    for j in range(seanet.n_residual_layers):
        check_conv(enc[index].block[1].conv, 1, seanet.dilation_base**j)
        residual(f"{m}encoder.model.{index}", f"mimi.enc.blk.{i}.res.{j}")
        index += 1
    assert isinstance(enc[index], torch.nn.ELU)
    check_conv(enc[index + 1].conv, ratio)
    halves(f"{m}encoder.model.{index + 1}.conv", f"mimi.enc.blk.{i}.down", ratio)
    index += 2
assert isinstance(enc[index], torch.nn.ELU) and index + 2 == len(enc)
conv(f"{m}encoder.model.{index + 1}.conv", "mimi.enc.conv_out")
transformer_layers(m + "encoder_transformer.transformer", "mimi.enc_tf", int(transformer_config.num_layers), scaled=True)
assert mimi.downsample.conv.pad_mode == "replicate"
halves(m + "downsample.conv.conv", "mimi.down", int(mimi.encoder_frame_rate / mimi.frame_rate), bias=False)

# The decoder, from the latents to the waveform: the quantizer's output projection, the depthwise transposed convolution
# to the encoder's rate, the transformer, and SEANet. Every transposed convolution has a width of twice its stride.
add("mimi.quant_out", tensor(m + "quantizer.output_proj.weight")[:, :, 0])
up = tensor(m + "upsample.convtr.convtr.weight")
assert mimi.upsample.convtr.convtr.groups == up.shape[0] and up.shape[1] == 1 and mimi.upsample.convtr.convtr.bias is None
add("mimi.up", np.transpose(up[:, 0, :], (1, 0)))
transformer_layers(m + "decoder_transformer.transformer", "mimi.dec_tf", int(transformer_config.num_layers), scaled=True)
dec = mimi.decoder.model
conv(m + "decoder.model.0.conv", "mimi.dec.conv_in")
index = 1
for i, ratio in enumerate(mimi.decoder.ratios):
    assert isinstance(dec[index], torch.nn.ELU)
    convtr = dec[index + 1].convtr
    assert convtr.stride[0] == ratio and convtr.kernel_size[0] == 2 * ratio and convtr.groups == 1
    add(f"mimi.dec.blk.{i}.up.weight", np.transpose(tensor(f"{m}decoder.model.{index + 1}.convtr.weight"), (2, 1, 0)))
    add(f"mimi.dec.blk.{i}.up.bias", tensor(f"{m}decoder.model.{index + 1}.convtr.bias"))
    index += 2
    for j in range(seanet.n_residual_layers):
        check_conv(dec[index].block[1].conv, 1, seanet.dilation_base**j)
        residual(f"{m}decoder.model.{index}", f"mimi.dec.blk.{i}.res.{j}")
        index += 1
assert isinstance(dec[index], torch.nn.ELU) and index + 2 == len(dec)
conv(f"{m}decoder.model.{index + 1}.conv", "mimi.dec.conv_out")

assert converted == set(state), sorted(set(state) - converted)

# The model's identity and languages in the GGUF specification's general keys, which name the file. They follow the
# tensors, whose parameters give the size label, and then go first after general.architecture, since GGUFWriter
# writes the keys in the order they were added.
parts = NAMES[args.model]
label = size_label(*w.get_total_parameter_count())
repository = f"https://huggingface.co/{repository_id}"
w.add_name(repository_id.split("/")[1])
w.add_organization(repository_id.split("/")[0])
w.add_basename(parts["basename"])
w.add_size_label(label)
w.add_finetune(parts["finetune"])
w.add_license(LICENSES[card["license"]])
w.add_source_url(f"{repository}/tree/{revision}")
w.add_source_repo_url(repository)
w.add_file_type(LlamaFileType.ALL_F32)
assert set(parts["languages"]) <= set(card["language"])
w.add_languages(parts["languages"])
w.kv_data[0] = dict(sorted(w.kv_data[0].items(), key=lambda item: not item[0].startswith("general.")))

path = os.path.join(args.out_dir, naming_convention(None, parts["basename"], parts["finetune"], None, label, "f32") + ".gguf")
w.write_header_to_file(path)
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", path, "with the encoder", sha256)
