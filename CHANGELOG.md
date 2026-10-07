# Changelog

## 0.2.1

- Stop the complete Windows worker process tree on cancellation, including the virtual-environment launcher's child Python process.
- Verify child-process cleanup in hardware cancellation checks.

## 0.2.0

- Rename the project to ComfyUI-QNN and the installation folder to comfyui_qnn.
- Add isolated ARM64 QNN SDXL generation at 1024×1024 with example workflows.
- Validate model files and context references before loading; select bundles by model family.
- Release encoder and UNet sessions between phases to reduce NPU memory pressure.
- Preserve SnapdragonSD15 node IDs and workflows.
- Add an explicit pinned SDXL download command; keep host dependencies empty.

## 0.1.0

- Add optional SD 1.5 text-to-image generation with an isolated ARM64 QNN worker.
- Include canvas and API workflows, installation instructions, and partial benchmark findings.
- Add reproducible ZIP packaging, checksums, CPU unit tests, and draft releases from version tags.

Experimental: validated only with the documented Snapdragon X Elite model bundle. Model binaries and vendor runtime packages are not included.
