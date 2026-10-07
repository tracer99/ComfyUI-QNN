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
    node_id = "SnapdragonSD15"
    display_name = "Snapdragon SD 1.5 (512×512)"
    marker = "text_encoder.bin"
    files = ("text_encoder.bin", "unet.bin", "vae.bin", "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json")
    worker = "qnn_sd15.py"
    environment = ".venv-qnn"
    interpreter_variable = "COMFYUI_QNN_PYTHON"
    default_steps = 20
    default_cfg = 7.5

    @classmethod
    def define_schema(cls):
        root = Path(folder_paths.models_dir) / "qnn"
        bundles = sorted(path.name for path in root.iterdir() if path.is_dir() and (path / cls.marker).exists()) if root.is_dir() else []
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category="snapdragon/qnn",
            is_experimental=True,
            inputs=[
                io.Combo.Input("model", options=bundles),
                io.String.Input("prompt", multiline=True),
                io.String.Input("negative_prompt", multiline=True, default=""),
                io.Int.Input("seed", default=0, min=-0x8000000000000000, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=cls.default_steps, min=1),
                io.Float.Input("cfg", default=cls.default_cfg),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, model: str, prompt: str, negative_prompt: str, seed: int, steps: int, cfg: float) -> io.NodeOutput:
        root = Path(folder_paths.models_dir) / "qnn"
        bundle = root / model
        if not folder_paths.is_within_directory(str(root), str(bundle)):
            raise ValueError("QNN model must be inside models/qnn.")
        for name in cls.files:
            path = bundle / name
            if not folder_paths.is_within_directory(str(root), str(path)):
                raise ValueError("QNN model files must be inside models/qnn.")
            if not path.is_file():
                raise FileNotFoundError(f"Missing QNN model file: {path}. See the extension's model installation guide.")
        checkout = Path(folder_paths.base_path)
        python = os.environ.get(cls.interpreter_variable, str(checkout / cls.environment / "Scripts" / "python.exe"))
        if not Path(python).is_file():
            raise FileNotFoundError(f"Install {cls.environment} or set {cls.interpreter_variable} to its ARM64 python.exe. See the extension README.")
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        request = dict(prompt=prompt, negative_prompt=negative_prompt, seed=seed, steps=steps, cfg=cfg)
        with tempfile.TemporaryDirectory(dir=folder_paths.get_temp_directory(), prefix="qnn-") as temporary:
            directory = Path(temporary)
            with (directory / "worker.log").open("w", encoding="utf-8") as log, (directory / "request.json").open("w+", encoding="utf-8") as inputs:
                json.dump(request, inputs)
                inputs.seek(0)
                process = subprocess.Popen(
                    [python, "-I", str(Path(__file__).with_name(cls.worker)), "--model", str(bundle.resolve()), "--output", str(directory / "image.npy"), "--report", str(directory / "report.json")],
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
                raise RuntimeError(f"{cls.display_name} failed:\n" + (directory / "worker.log").read_text(encoding="utf-8")[-4000:])
            image = np.load(directory / "image.npy", allow_pickle=False)
        return io.NodeOutput(torch.from_numpy(image))


class SnapdragonSDXL(SnapdragonSD15):
    node_id = "SnapdragonSDXL"
    display_name = "Snapdragon SDXL (1024×1024)"
    marker = "text_encoder"
    files = tuple(f"{component}/model.{extension}" for component in ("text_encoder", "text_encoder_2", *(f"unet/part{i}" for i in range(5)), "vae_decoder/1024x1024") for extension in ("onnx", "bin")) + tuple(
        f"{tokenizer}/{name}" for tokenizer in ("tokenizer", "tokenizer_2")
        for name in ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json")
    )
    worker = "qnn_sdxl.py"
    environment = ".venv-qnn-sdxl"
    interpreter_variable = "COMFYUI_QNN_SDXL_PYTHON"
    default_steps = 6
    default_cfg = 2.0


class SnapdragonDiffusionExtension(ComfyExtension):
    async def get_node_list(self):
        return [SnapdragonSD15, SnapdragonSDXL]


async def comfy_entrypoint():
    return SnapdragonDiffusionExtension()
