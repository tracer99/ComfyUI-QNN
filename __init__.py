import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

import folder_paths
import comfy.model_management
from comfy_api.latest import ComfyExtension, io


class SnapdragonSD15(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        root = Path(folder_paths.models_dir) / "qnn"
        bundles = sorted(path.name for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
        return io.Schema(
            node_id="SnapdragonSD15",
            display_name="Snapdragon SD 1.5",
            category="snapdragon/qnn",
            is_experimental=True,
            inputs=[
                io.Combo.Input("model", options=bundles),
                io.String.Input("prompt", multiline=True),
                io.String.Input("negative_prompt", multiline=True, default=""),
                io.Int.Input("seed", default=0, min=-0x8000000000000000, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=20, min=1),
                io.Float.Input("cfg", default=7.5),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, model: str, prompt: str, negative_prompt: str, seed: int, steps: int, cfg: float) -> io.NodeOutput:
        root = Path(folder_paths.models_dir) / "qnn"
        bundle = root / model
        if not folder_paths.is_within_directory(str(root), str(bundle)):
            raise ValueError("QNN model must be inside models/qnn.")
        for name in ("text_encoder.bin", "unet.bin", "vae.bin", "tokenizer"):
            path = bundle / name
            if not folder_paths.is_within_directory(str(root), str(path)):
                raise ValueError("QNN model files must be inside models/qnn.")
            if not path.exists():
                raise FileNotFoundError(path)
        checkout = Path(folder_paths.base_path)
        python = os.environ.get("COMFYUI_QNN_PYTHON", str(checkout / ".venv-qnn" / "Scripts" / "python.exe"))
        if not Path(python).is_file():
            raise FileNotFoundError("Install the SD 1.5 worker environment or set COMFYUI_QNN_PYTHON to its ARM64 python.exe.")
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        request = dict(prompt=prompt, negative_prompt=negative_prompt, seed=seed, steps=steps, cfg=cfg)
        with tempfile.TemporaryDirectory(dir=folder_paths.get_temp_directory(), prefix="qnn-sd15-") as temporary:
            directory = Path(temporary)
            with (directory / "worker.log").open("w", encoding="utf-8") as log, (directory / "request.json").open("w+", encoding="utf-8") as inputs:
                json.dump(request, inputs)
                inputs.seek(0)
                process = subprocess.Popen(
                    [python, "-I", str(Path(__file__).with_name("qnn_sd15.py")), "--model", str(bundle.resolve()), "--output", str(directory / "image.npy"), "--report", str(directory / "report.json")],
                    stdin=inputs, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                try:
                    while process.poll() is None:
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        try:
                            process.wait(timeout=0.1)
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
            if process.returncode:
                raise RuntimeError("QNN SD 1.5 failed:\n" + (directory / "worker.log").read_text(encoding="utf-8")[-4000:])
            image = np.load(directory / "image.npy", allow_pickle=False)
        return io.NodeOutput(torch.from_numpy(image))


class SnapdragonSD15Extension(ComfyExtension):
    async def get_node_list(self):
        return [SnapdragonSD15]


async def comfy_entrypoint():
    return SnapdragonSD15Extension()
