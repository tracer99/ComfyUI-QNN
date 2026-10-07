# Snapdragon SD 1.5 for ComfyUI

Experimental, optional custom node for Qualcomm's precompiled Snapdragon X Elite SD 1.5 w8a16 bundle. Generates one 512×512 IMAGE; connect it to Preview Image or Save Image. The worker uses the NPU with no CPU fallback or network requests.

## Install

From your ComfyUI directory, clone the standalone repository into `custom_nodes`:

```powershell
git clone https://github.com/tracer99/ComfyUI-QNN-SD15.git custom_nodes/comfyui_qnn_sd15
```

Create a separate environment using **native ARM64 Python 3.11**, then install its worker dependencies (replace the interpreter path):

```powershell
& 'C:/path/to/arm64/python.exe' -m venv .venv-qnn
.venv-qnn/Scripts/python.exe -m pip install -r custom_nodes/comfyui_qnn_sd15/requirements-worker.txt
```

An existing worker environment can be selected with `COMFYUI_QNN_PYTHON`, set to its full `python.exe` path before starting ComfyUI. Keep these dependencies out of the ComfyUI host environment. The host may use x64 DirectML; the ARM fork is not required by this extension, but the host must support the ComfyUI V3 node API.

For a versioned installation, download the ZIP from [Releases](https://github.com/tracer99/ComfyUI-QNN-SD15/releases), verify its accompanying SHA-256 checksum, and extract its `comfyui_qnn_sd15` folder into `custom_nodes`. Root `requirements.txt` intentionally has no host dependencies; `requirements-worker.txt` is installed manually into the ARM64 worker environment. The ZIP contains no models or runtime binaries.

Manually obtain the [Qualcomm SD 1.5 X Elite bundle](https://huggingface.co/qualcomm/Stable-Diffusion-v1.5). Validated release: v0.64.0, QAIRT 2.50, X Elite, w8a16. Put `text_encoder.bin`, `unet.bin`, and `vae.bin` in `models/qnn/sd15/`. Put `vocab.json`, `merges.txt`, `tokenizer_config.json`, and `special_tokens_map.json` from [the source tokenizer](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/tokenizer) in `models/qnn/sd15/tokenizer/`.

Restart ComfyUI and drag `workflow.json` into the canvas. Select `sd15`, edit the prompt, and queue the workflow. `workflow-api.json` is the equivalent API prompt. If the interpreter is missing, check `COMFYUI_QNN_PYTHON`; if HTP fails, check that the model bundle and runtime match this device.

## Scope

Text-to-image only: fixed resolution, one image, DPM-Solver++ multistep sampling, prompt, negative prompt, seed, steps, and CFG. Arbitrary checkpoints, SDXL, LoRA, ControlNet, img2img, and integration with KSampler are outside this initial version. The model truncates prompts to 77 tokens.

Each execution starts and releases its worker and model contexts, so startup and model loading are paid for every execution.

The [existing community SDXL nodes](https://github.com/buuta-buta-butaata/SDXL-with-Snapdragon-X-Elite-NPU/tree/main/ComfyUI/custom_nodes/onnxruntime-qnn-nodes), inspected at revision `2f65bbea8f546cf26bd7378d1aab5ee39c9fbc5b`, require five ONNX UNet parts, dual CLIP encoders, and SDXL conditioning. They cannot load this three-file SD 1.5 context-binary bundle. Supporting it there would require a separate model path; this small package keeps that path optional and independently reviewable.

## Partial benchmark findings

Measured on October 7, 2026, on a Snapdragon X Elite X1E80100 notebook with Adreno X1-85 graphics and 32 GB RAM, running on battery. QNN completed 36 images; DirectML completed 22 before the run was stopped by the user. The table uses only the **22 matching prompts and seeds**, at 512×512, 20 steps, CFG 7.5, with the same DPM-Solver++ multistep scheduler and CPU initial noise.

| Measurement | Qualcomm NPU | DirectML |
|---|---:|---:|
| Median generation time | 4.91 s | 100.46 s |
| Average generation time | 4.90 s | 107.83 s |
| Estimated whole-machine energy per image | 85.5 J | 1,578 J |
| Average whole-machine power during generation | 17.44 W | 14.63 W |

Warm NPU generation was **20.4× faster by median time**, using approximately **94.6% less estimated energy per image**. It drew more power while generating, but finished much sooner. These timings include text encoding, denoising, and VAE decoding; they exclude loading, worker startup, and image saving. The earlier complete NPU ComfyUI API workflow took approximately **17.5 seconds**, including startup and loading. Measurements preceded packaging as an extension; there is no matched end-to-end DirectML workflow measurement.

Eight representative image pairs showed broadly comparable composition and detail, with differences in lighting, texture, and anatomy. Both paths produced artifacts, including a double-spout teapot. This visual review found no obvious quality collapse but does not establish quality equivalence. [View the comparison gallery](findings/comparison.jpg), with QNN on the left and DirectML on the right, or inspect the [calculated results](findings/summary.json).

Energy was integrated over completed generation intervals using coarse battery discharge readings. It measures the whole notebook, not NPU/GPU power, and is not idle-subtracted. The runs were sequential; average idle power differed (QNN 11.82 W, DirectML 8.57 W), and background activity and thermal conditions were not fully controlled. Qualcomm does not pin the exact source-weight revision in the bundle, so weight equivalence is unverified. DirectML reported a CPU fallback for `aten::frac.out`; no separate CPU benchmark was completed.

These partial findings support optional SD 1.5 support on this machine. They do not establish performance or quality for other models or Snapdragon devices.

## Development and releases

Version metadata lives in `pyproject.toml`. See [RELEASING.md](RELEASING.md) for CPU tests, reproducible ZIP packaging, hardware smoke checks, and the tag-triggered draft prerelease workflow. ComfyUI Registry publication is not configured.

Source code is licensed under [GPL-3.0](LICENSE). Qualcomm models and runtime packages retain their own licenses and must be obtained separately.
