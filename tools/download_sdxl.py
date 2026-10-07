"""Explicitly download the pinned 1024x1024 SDXL bundle; never called by nodes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen


REPOSITORY = "Buuta/dreamshaper-xl-lightning-for-Snapdragon-X-Elite"
REVISION = "09fdf512425d746f4105299ecb46a4bcffa28632"
COMPONENTS = ("text_encoder", "text_encoder_2", *(f"unet/part{i}" for i in range(5)), "vae_decoder/1024x1024")
FILES = tuple(f"{part}/model.{extension}" for part in COMPONENTS for extension in ("onnx", "bin")) + tuple(
    f"{tokenizer}/{name}" for tokenizer in ("tokenizer", "tokenizer_2")
    for name in ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json")
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    with urlopen(f"https://huggingface.co/api/models/{REPOSITORY}/tree/{REVISION}?recursive=true&expand=true&limit=100", timeout=60) as response:
        manifest = {entry["path"]: entry for entry in json.load(response)}

    def download(name):
        target = args.destination / name
        entry = manifest[name]
        checksum = entry.get("lfs", {}).get("oid")
        if target.is_file() and target.stat().st_size == entry["size"]:
            with target.open("rb") as source:
                verified = checksum is None or hashlib.file_digest(source, "sha256").hexdigest() == checksum
            if verified:
                print(f"Verified {name}", flush=True)
                return
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        with urlopen(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}?download=true", timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if partial.stat().st_size != entry["size"]:
            raise ValueError(f"Incomplete download: {name}")
        if checksum:
            with partial.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != checksum:
                    raise ValueError(f"Checksum mismatch: {name}")
        partial.replace(target)
        print(f"Downloaded {name}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(download, FILES))
    (args.destination / "source.json").write_text(json.dumps(dict(repository=REPOSITORY, revision=REVISION, resolution="1024x1024"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
