"""Build a reproducible ComfyUI custom-node ZIP without bundling runtimes or models."""

import argparse
import hashlib
from pathlib import Path
import re
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def build(output, tag=None):
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = config["project"]["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Release version must be major.minor.patch.")
    if tag is not None and tag != "v" + version:
        raise ValueError(f"Tag {tag!r} must match v{version} in pyproject.toml.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"comfyui-qnn-{version}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for name in sorted(config["tool"]["release"]["files"]):
            source = ROOT / name
            if not source.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError(f"Release file escapes project: {name}")
            entry = zipfile.ZipInfo("comfyui_qnn/" + name, date_time=(2020, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            package.writestr(entry, source.read_bytes(), compresslevel=9)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--tag")
    args = parser.parse_args()
    print(build(args.output, args.tag))


if __name__ == "__main__":
    main()
