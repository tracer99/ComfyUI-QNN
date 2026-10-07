# Release process

This project ships a ComfyUI custom-node ZIP, rather than a PyPI wheel. The archive contains the node, worker source, dependency lists, workflows, documentation, license, and benchmark findings. Model binaries and installed vendor runtimes are excluded.

## Development checks

Use Python 3.11 or newer. The CPU tests do not require a Snapdragon device, model downloads, or ComfyUI:

```powershell
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python tools/build_release.py --tag v0.2.0
```

GitHub Actions runs these tests on Linux and Windows, builds the ZIP, and uploads it with a SHA-256 checksum. The archive's file list is explicit in `pyproject.toml`, and ZIP timestamps are fixed for repeatable builds on the same toolchain.

## Cut a release

1. Update `project.version` in `pyproject.toml` and add a matching entry to `CHANGELOG.md`.
2. Run the checks above using the new version tag. On an X Elite, extract the ZIP outside `custom_nodes` and run the ComfyUI host interpreter with `python tools/check_comfyui.py --comfyui C:/path/to/ComfyUI --package C:/path/to/extracted/comfyui_qnn --generate --cancel`. Repeat with `--family sdxl`. Both checks must pass registration, bundle validation, generation, and cancellation with the worker exited. Record the host, worker, and model versions in the release notes. CI cannot validate NPU execution.
3. Push the tested commit and tag. A feature branch can produce a draft prerelease for review; merge it before publishing:

```powershell
git tag -a v0.2.0 -m "Release 0.2.0"
git push origin v0.2.0
```

4. The tag workflow checks that the tag matches the package version and creates a **draft prerelease** with the ZIP and checksum. Review its notes and attach the hardware validation results before publishing:

```powershell
gh release edit v0.2.0 --draft=false
```

Do not move published tags. Fixes get a new patch version. Registry publication is a separate future step; it requires a registered ComfyUI publisher and is not configured here.

## 0.2.0 hardware validation

On October 7, 2026, 12 CPU unit tests passed locally. The release ZIP registered both nodes with the ComfyUI 0.39.0 x64 host. The SD 1.5 example generated a finite, nonuniform 512×512 image in 29.41 seconds; the SDXL example generated a finite, nonuniform 1024×1024 image in 41.71 seconds. Both checks acknowledged cancellation and confirmed worker exit. Visual inspection found a coherent red teapot on a wooden table in the SDXL result. These are individual smoke checks, not comparative benchmarks or power measurements.

Hardware: Snapdragon X Elite X1E80100, 32 GB RAM. SDXL worker: native ARM64 Python 3.11, Torch 2.10.0+cpu, ONNX Runtime 1.24.4, ONNX Runtime QNN 2.3.0, ONNX 1.20.1, Transformers 4.57.6, Diffusers 0.35.1. Model: Buuta DreamShaper XL Lightning revision `09fdf512425d746f4105299ecb46a4bcffa28632`, FP16 five-part UNet and 1024×1024 decoder. SD 1.5 uses the same worker/model versions documented below.

Preloading all eight SDXL sessions hit a QNN memory-allocation failure on this machine. Loading sessions by phase (encoders, UNet, decoder) completed generation successfully. Provider selection requires the NPU, and CPU fallback is disabled. Linux/Windows CI covers CPU contracts and packaging; SDXL quality, speed, and energy versus DirectML remain unmeasured. The stopped benchmark was not resumed.

## Initial 0.1.0 validation

On October 7, 2026, all five CPU unit tests passed locally. A fresh extraction of the release ZIP passed registration with the ComfyUI 0.39.0 x64 host, generated a finite, nonuniform 512×512 image with the example's 20 steps and seed 42, and acknowledged cancellation with its worker process fully exited. The node call took 12.85 seconds including worker startup and loading; this is a single smoke check, not a new performance comparison. The benchmark was not resumed.

Hardware: Snapdragon X Elite X1E80100, 32 GB RAM. Worker: native ARM64 Python 3.11, Torch 2.10.0+cpu, QAI AppBuilder 2.50.40, Transformers 4.57.6, Diffusers 0.35.1. Model: Qualcomm SD 1.5 v0.64.0 X Elite w8a16, QAIRT 2.50. Linux/Windows CI tests cover CPU contracts and packaging, not hardware execution.
