"""The pinned code and models every script in this folder runs: each model's official configuration, which names its
weights and tokenizer by Hugging Face revision, the model card beside it, and the size and SHA-256 of each file they
name, checked after download.

usage: uv run python pins.py <english|japanese>

prints the local path of the model's configuration, weights, tokenizer and card, downloaded and checked on first use.
"""

import hashlib
import importlib.metadata
import json
import os
import sys

from pocket_tts.utils.config import CONFIGS_DIR, load_config
from pocket_tts.utils.utils import download_if_necessary

_installed = json.loads(importlib.metadata.distribution("pocket-tts").read_text("direct_url.json"))
CODE = {"repository": _installed["url"], "commit": _installed["vcs_info"]["commit_id"]}
# The configuration of each model: english is the official package's own, which names the gated weights of
# kyutai/pocket-tts, whose Mimi encoder makes voices from recordings; japanese is the configuration in its repository.
CONFIGS = {
    "english": str(CONFIGS_DIR / "english.yaml"),
    "japanese": "hf://takuma104/pocket-tts-ja/japanese.yaml@119685fd7a51356429c6d25e37d570b423c17663",
}
# The card of each model's repository, at the revision of its weights, which gives its license.
CARDS = {
    "english": "hf://kyutai/pocket-tts/README.md@7f364c05d1be3faf2b234cdbe4cbb68a026d4218",
    "japanese": "hf://takuma104/pocket-tts-ja/README.md@119685fd7a51356429c6d25e37d570b423c17663",
}
FILES = {
    "hf://takuma104/pocket-tts-ja/japanese.yaml@119685fd7a51356429c6d25e37d570b423c17663":
        (2780, "95cdd6503ab8dc62531af8b3cd12f1afc4ef877fab1a3aca84a82d4e73871d92"),
    "hf://kyutai/pocket-tts/languages/english/model.safetensors@7f364c05d1be3faf2b234cdbe4cbb68a026d4218":
        (219029196, "7767ce79b1b8db523d867e70855a00627aacfe44aaa5c1fd7ebe135fba71fff9"),
    "hf://kyutai/pocket-tts-without-voice-cloning/languages/english/tokenizer.json@00eac05ed3d16bdc3f6b5d598874019c34a89214":
        (245020, "f498428e1eafee50492f7be13dc9bfafcfc12e508cd0eb1b01c92ecd5d8c6687"),
    "hf://kyutai/pocket-tts/README.md@7f364c05d1be3faf2b234cdbe4cbb68a026d4218":
        (26699, "c01f4a60dca73aa844473181096618fe9b3bf6a148986c8c368190ba03627054"),
    "hf://takuma104/pocket-tts-ja/japanese/model.safetensors@562a82cba331499b75a050b7e200f4ee3241b9e9":
        (219029196, "b63cd60eb2966ad4d03ef808036ae1a42ad53788e370f8f11f366f33d2fae47f"),
    "hf://takuma104/pocket-tts-ja/tokenizer.model@562a82cba331499b75a050b7e200f4ee3241b9e9":
        (60589, "2a9486f54e886c9459d60de909a0fb5c08dae2ad02c2f2ace6b80c7425af30c0"),
    "hf://takuma104/pocket-tts-ja/README.md@119685fd7a51356429c6d25e37d570b423c17663":
        (9354, "641775459e8090dd0b63e9b4b47723a28bb1f764913adf00e27a2989eec08324"),
}


def fetch(url):
    """The local path of a pinned file, downloaded on first use and checked against its size and SHA-256."""
    if url not in FILES:
        raise SystemExit(f"{url} is not pinned in pins.py")
    path = str(download_if_necessary(url))
    size, sha256 = FILES[url]
    if os.path.getsize(path) != size:
        raise SystemExit(f"{path} has {os.path.getsize(path)} bytes, not the pinned {size}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise SystemExit(f"{path} has the SHA-256 {digest.hexdigest()}, not the pinned {sha256}")
    return path


def model(name):
    """The configuration of a pinned model, its path, and the checked local paths of its weights, tokenizer and card."""
    config_path = fetch(CONFIGS[name]) if CONFIGS[name].startswith("hf://") else CONFIGS[name]
    config = load_config(config_path)
    return {
        "config": config,
        "config_path": config_path,
        "weights": fetch(config.weights_path),
        "tokenizer": fetch(config.flow_lm.lookup_table.tokenizer_path),
        "card": fetch(CARDS[name]),
    }


if __name__ == "__main__":
    pinned = model(sys.argv[1])
    for key in ("config_path", "weights", "tokenizer", "card"):
        print(key, pinned[key])
