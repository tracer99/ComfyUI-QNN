# Snapdragon Diffusion for ComfyUI

Optional custom nodes for Snapdragon X Elite NPU text-to-image generation. `SnapdragonSD15` generates 512×512; `SnapdragonSDXL` generates 1024×1024 using DreamShaper XL Lightning. Connect either IMAGE output to Preview Image or Save Image. The workers run locally on the NPU with CPU fallback disabled.

## Install

From your ComfyUI directory:

```powershell
git clone https://github.com/tracer99/ComfyUI-QNN.git custom_nodes/comfyui_qnn
```

Upgrading from 0.1.0: close ComfyUI, move `custom_nodes/comfyui_qnn_sd15` outside `custom_nodes`, then install the new folder. Keep only one copy. The `SnapdragonSD15` node ID and its workflow inputs are unchanged. GitHub redirects the old repository URL to the new name.

Use native **ARM64 Python 3.11** for the workers; your ComfyUI host can continue using x64 DirectML. The ARM fork is not required by this extension. ComfyUI must support the V3 node API.

### SD 1.5

```powershell
& 'C:/path/to/arm64/python.exe' -m venv .venv-qnn
.venv-qnn/Scripts/python.exe -m pip install -r custom_nodes/comfyui_qnn/requirements-worker.txt
```

Manually obtain the [Qualcomm SD 1.5 X Elite bundle](https://huggingface.co/qualcomm/Stable-Diffusion-v1.5): validated release v0.64.0, QAIRT 2.50, w8a16. Put `text_encoder.bin`, `unet.bin`, and `vae.bin` in `models/qnn/sd15/`. Put `vocab.json`, `merges.txt`, `tokenizer_config.json`, and `special_tokens_map.json` from [the source tokenizer](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/tokenizer) in `models/qnn/sd15/tokenizer/`.

Import `workflow.json` for SD 1.5; `workflow-api.json` is its API prompt. An existing worker can be selected with `COMFYUI_QNN_PYTHON` pointing to its full `python.exe` path.

### SDXL

```powershell
& 'C:/path/to/arm64/python.exe' -m venv .venv-qnn-sdxl
.venv-qnn-sdxl/Scripts/python.exe -m pip install -r custom_nodes/comfyui_qnn/requirements-sdxl-worker.txt
.venv-qnn-sdxl/Scripts/python.exe custom_nodes/comfyui_qnn/tools/download_sdxl.py --destination models/qnn/dreamshaper-xl-lightning
```

The last command explicitly downloads approximately 7 GB from [the compiled DreamShaper XL Lightning bundle](https://huggingface.co/Buuta/dreamshaper-xl-lightning-for-Snapdragon-X-Elite/tree/09fdf512425d746f4105299ecb46a4bcffa28632), pinned to revision `09fdf512425d746f4105299ecb46a4bcffa28632`. It verifies available artifact SHA-256 checksums and file sizes. Nodes never initiate downloads. For offline installation, copy exactly these files into the bundle folder:

```text
text_encoder/{model.onnx,model.bin}
text_encoder_2/{model.onnx,model.bin}
unet/part0/{model.onnx,model.bin}
unet/part1/{model.onnx,model.bin}
unet/part2/{model.onnx,model.bin}
unet/part3/{model.onnx,model.bin}
unet/part4/{model.onnx,model.bin}
vae_decoder/1024x1024/{model.onnx,model.bin}
tokenizer/{vocab.json,merges.txt,tokenizer_config.json,special_tokens_map.json}
tokenizer_2/{vocab.json,merges.txt,tokenizer_config.json,special_tokens_map.json}
```

Restart ComfyUI and import `workflow-sdxl.json`; `workflow-sdxl-api.json` is its API prompt. Start with 6 steps and CFG 2. Both required text encoders are internal and share the prompt fields. Resolution is fixed by the compiled bundle and shown in the node name. An existing worker can be selected with `COMFYUI_QNN_SDXL_PYTHON` pointing to its ARM64 `python.exe`.

Missing model errors name the required file. Missing interpreter errors identify the environment variable. If QNN cannot find an NPU or load a context, check Snapdragon drivers and the pinned worker dependencies. SDXL releases encoder sessions before loading the UNet and releases the UNet before decoding to reduce NPU memory pressure.

## Scope

One image per execution; prompts, negative prompts, seed, steps, and CFG. SD 1.5 uses DPM-Solver++ multistep sampling; SDXL uses the reference Euler scheduler with leading timesteps. Compiled prompts are limited to 77 tokens. Arbitrary checkpoints/resolutions, LoRA, ControlNet, img2img, and KSampler integration are not supported.

Workers and model sessions are released after each execution, including cancellation. Model loading is paid again on the next execution. The separate SDXL worker avoids provider-registration conflicts with QNN extensions in the host.

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

Version metadata lives in `pyproject.toml`. See [RELEASING.md](RELEASING.md) for CPU tests, reproducible ZIP packaging, hardware smoke checks, and tag-triggered draft prereleases. Release ZIPs contain the `comfyui_qnn` folder, no models or runtimes. Verify the accompanying SHA-256 before extraction. Root `requirements.txt` intentionally installs no host dependencies. ComfyUI Registry publication is not configured.

Source is [GPL-3.0](LICENSE). SDXL graph wiring follows the [community reference](https://github.com/buuta-buta-butaata/SDXL-with-Snapdragon-X-Elite-NPU) at revision `2f65bbea8f546cf26bd7378d1aab5ee39c9fbc5b`; see [NOTICE](NOTICE). Models and runtimes retain their own licenses and must be obtained separately.
