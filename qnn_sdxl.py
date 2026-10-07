"""Offline DreamShaper XL Lightning generation with ONNX Runtime's QNN NPU provider.

Graph connections follow buuta-buta-butaata/SDXL-with-Snapdragon-X-Elite-NPU;
see NOTICE for the reference revision and license.
"""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import onnx
import onnxruntime as ort
import torch
from diffusers import EulerDiscreteScheduler
from transformers import CLIPTokenizer
try:
    import onnxruntime_qnn as qnn
except ImportError:
    qnn = None


COMPONENTS = ("text_encoder", "text_encoder_2", *(f"unet/part{i}" for i in range(5)), "vae_decoder/1024x1024")
INPUT_DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int32)": np.int32, "tensor(int64)": np.int64}


def session_options():
    if qnn is None:
        raise RuntimeError("Install requirements-sdxl-worker.txt in native ARM64 Python 3.11.")
    devices = ort.get_ep_devices()
    if not any(device.ep_name == "QNNExecutionProvider" for device in devices):
        ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
        devices = ort.get_ep_devices()
    devices = [device for device in devices if device.ep_name == "QNNExecutionProvider" and device.device.type == ort.OrtHardwareDeviceType.NPU]
    if not devices:
        raise RuntimeError("No QNN NPU device found. Check Snapdragon drivers and requirements-sdxl-worker.txt.")
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {"backend_path": qnn.get_qnn_htp_path(), "htp_performance_mode": "burst", "enable_htp_fp16_precision": "1"})
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.log_severity_level = 3
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    return options


def validate_context(path, root):
    root = root.resolve()
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Model path escapes bundle: {path}")
    graph = onnx.load(str(path), load_external_data=False)
    if graph.graph.initializer or graph.graph.sparse_initializer or not graph.graph.node or any(node.op_type != "EPContext" or node.domain != "com.microsoft" for node in graph.graph.node):
        raise ValueError(f"Expected a compiled QNN context wrapper: {path}")
    for node in graph.graph.node:
        attributes = {attribute.name: attribute for attribute in node.attribute}
        if not {"embed_mode", "ep_cache_context", "source"}.issubset(attributes) or attributes["embed_mode"].i != 0 or attributes["source"].s != b"QNN":
            raise ValueError(f"Expected an external QNN context binary: {path}")
        name = attributes["ep_cache_context"].s.decode("utf-8")
        binary = path.parent / name
        if name not in ("model.bin", "./model.bin") or not binary.resolve().is_relative_to(root):
            raise ValueError(f"QNN context binary must be model.bin inside the bundle: {path}")
        if not binary.is_file():
            raise FileNotFoundError(f"Missing QNN context binary: {binary}")


def run(session, values):
    feeds = {}
    for argument in session.get_inputs():
        feeds[argument.name] = np.ascontiguousarray(values[argument.name], dtype=INPUT_DTYPES[argument.type])
    return dict(zip((output.name for output in session.get_outputs()), session.run(None, feeds)))


class SDXL:
    def __init__(self, directory):
        directory = Path(directory)
        for component in COMPONENTS:
            validate_context(directory / component / "model.onnx", directory)
        for tokenizer in ("tokenizer", "tokenizer_2"):
            for name in ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"):
                path = directory / tokenizer / name
                if not path.resolve().is_relative_to(directory.resolve()):
                    raise ValueError(f"Tokenizer path escapes bundle: {path}")
                if not path.is_file():
                    raise FileNotFoundError(f"Missing tokenizer file: {path}")
        self.sessions = []
        self.directory = directory
        self.load_seconds = 0
        try:
            self.tokenizers = [CLIPTokenizer.from_pretrained(directory / name, local_files_only=True) for name in ("tokenizer", "tokenizer_2")]
            self.options = session_options()
            self.load(COMPONENTS[:2])
        except BaseException:
            self.close()
            raise

    def close(self):
        self.sessions.clear()

    def load(self, components):
        self.close()
        start = perf_counter()
        try:
            for component in components:
                self.sessions.append(ort.InferenceSession(str(self.directory / component / "model.onnx"), sess_options=self.options))
        except BaseException:
            self.close()
            raise
        self.load_seconds += perf_counter() - start

    def encode(self, prompt):
        outputs = []
        for tokenizer, session in zip(self.tokenizers, self.sessions[:2]):
            ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np").input_ids
            outputs.append(run(session, {"input_ids": ids}))
        # These are the exported penultimate hidden states and projected CLIP-G pool.
        hidden_l = outputs[0]["output_1"]
        pooled, hidden_g = outputs[1]["output_0"], outputs[1]["output_1"]
        if hidden_l.shape != (1, 77, 768) or hidden_g.shape != (1, 77, 1280) or pooled.shape != (1, 1280):
            raise ValueError("Unsupported SDXL text-encoder outputs; use the pinned DreamShaper bundle.")
        return np.concatenate((hidden_l, hidden_g), axis=-1), pooled

    def denoise(self, latent, timestep, conditioning):
        hidden, pooled = conditioning
        common, down, mid, up1, up2 = self.sessions
        values = dict(timestep=np.array([timestep], dtype=np.float32), text_embeds=pooled, time_ids=np.array([[1024, 1024, 0, 0, 1024, 1024]], dtype=np.float32))
        values.update(run(common, values))
        values.update(sample=latent.numpy(), encoder_hidden_states=hidden)
        values.update(run(down, values))
        values["add_88"] = values["add_88"].transpose(0, 3, 1, 2)
        values.update(run(mid, values))
        values.update(run(up1, values))
        values["add_156"] = values["add_156"].transpose(0, 3, 1, 2)
        noise = run(up2, values)["out_sample"]
        return torch.from_numpy(noise.astype(np.float32))

    def decode(self, latent):
        self.load(COMPONENTS[7:])
        image = run(self.sessions[0], {"latent_sample": (latent / 0.13025).numpy()})["output_0"].transpose(0, 2, 3, 1)
        if image.shape != (1, 1024, 1024, 3):
            raise ValueError(f"Unsupported SDXL decoder output: {image.shape}")
        return (image.astype(np.float32) / 2 + 0.5).clip(0, 1).copy()


def generate(model, prompt, negative_prompt, seed, steps, cfg):
    start = perf_counter()
    negative = model.encode(negative_prompt)
    positive = model.encode(prompt)
    encoded = perf_counter()
    initial_load = model.load_seconds
    model.load(COMPONENTS[2:7])
    unet_load = model.load_seconds - initial_load
    scheduler = EulerDiscreteScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", timestep_spacing="leading", steps_offset=1)
    scheduler.set_timesteps(steps)
    latent = torch.randn((1, 4, 128, 128), generator=torch.Generator(device="cpu").manual_seed(seed)) * scheduler.init_noise_sigma
    for timestep in scheduler.timesteps:
        scaled = scheduler.scale_model_input(latent, timestep)
        conditional = model.denoise(scaled, float(timestep), positive)
        if cfg == 1:
            noise = conditional
        else:
            unconditional = model.denoise(scaled, float(timestep), negative)
            noise = unconditional + cfg * (conditional - unconditional)
        latent = scheduler.step(noise, timestep, latent).prev_sample
    denoised = perf_counter()
    image = model.decode(latent)
    end = perf_counter()
    decoder_load = model.load_seconds - initial_load - unet_load
    return image, dict(text_seconds=encoded - start, denoise_seconds=denoised - encoded - unet_load, decode_seconds=end - denoised - decoder_load, generation_seconds=end - start - unet_load - decoder_load)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    request = json.load(sys.stdin)
    start = perf_counter()
    model = SDXL(args.model)
    try:
        image, timings = generate(model, **request)
        timings.update(load_seconds=model.load_seconds, worker_seconds=perf_counter() - start, backend="onnxruntime-qnn-htp", resolution="1024x1024", steps=request["steps"], seed=request["seed"])
        np.save(args.output, image, allow_pickle=False)
        Path(args.report).write_text(json.dumps(timings, indent=2), encoding="utf-8")
    finally:
        model.close()


if __name__ == "__main__":
    main()
