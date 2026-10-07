# Release process

This project ships a ComfyUI custom-node ZIP, rather than a PyPI wheel. The archive contains the node, worker source, dependency lists, workflows, documentation, license, and benchmark findings. Model binaries and installed vendor runtimes are excluded.

## Development checks

Use Python 3.11 or newer. The CPU tests do not require a Snapdragon device, model downloads, or ComfyUI:

```powershell
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python tools/build_release.py --tag v0.1.0
```

GitHub Actions runs these tests on Linux and Windows, builds the ZIP, and uploads it with a SHA-256 checksum. The archive's file list is explicit in `pyproject.toml`, and ZIP timestamps are fixed for repeatable builds on the same toolchain.

## Cut a release

1. Update `project.version` in `pyproject.toml` and add a matching entry to `CHANGELOG.md`.
2. Run the checks above using the new version tag. On an X Elite, install the resulting ZIP into a clean custom-node directory, load the example workflow, generate an image, and check cancellation. Use the ComfyUI host interpreter for `python tools/check_comfyui.py --comfyui C:/path/to/ComfyUI --package C:/path/to/extracted/comfyui_qnn_sd15 --generate --cancel`. Record the tested ComfyUI version, worker versions, and model release in the release notes. CI cannot validate NPU execution.
3. Commit and push to `main`, then tag the tested commit:

```powershell
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

4. The tag workflow checks that the tag matches the package version and creates a **draft prerelease** with the ZIP and checksum. Review its notes and attach the hardware validation results before publishing:

```powershell
gh release edit v0.1.0 --draft=false
```

Do not move published tags. Fixes get a new patch version. Registry publication is a separate future step; it requires a registered ComfyUI publisher and is not configured here.

## Initial 0.1.0 validation

On October 7, 2026, all five CPU unit tests passed locally. A fresh extraction of the release ZIP passed registration with the ComfyUI 0.39.0 x64 host, generated a finite, nonuniform 512×512 image with the example's 20 steps and seed 42, and acknowledged cancellation with its worker process fully exited. The node call took 12.85 seconds including worker startup and loading; this is a single smoke check, not a new performance comparison. The benchmark was not resumed.

Hardware: Snapdragon X Elite X1E80100, 32 GB RAM. Worker: native ARM64 Python 3.11, Torch 2.10.0+cpu, QAI AppBuilder 2.50.40, Transformers 4.57.6, Diffusers 0.35.1. Model: Qualcomm SD 1.5 v0.64.0 X Elite w8a16, QAIRT 2.50. Linux/Windows CI tests cover CPU contracts and packaging, not hardware execution.
