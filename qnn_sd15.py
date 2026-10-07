"""Offline SD 1.5 generation using Qualcomm's X Elite context binaries."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler
from transformers import CLIPTokenizer
try:
    from qai_appbuilder import QNNConfig, QNNContext, Runtime, LogLevel, ProfilingLevel
except ImportError:
    QNNContext = None


class SD15:
    def __init__(self, directory):
        if QNNContext is None:
            raise RuntimeError("Install requirements-worker.txt in native ARM64 Python 3.11.")
        directory = Path(directory)
        self.tokenizer = CLIPTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
        QNNConfig.Config(runtime=Runtime.HTP, log_level=LogLevel.ERROR, profiling_level=ProfilingLevel.OFF)
        self.contexts = []
        try:
            for name in ("text_encoder", "unet", "vae"):
                self.contexts.append(QNNContext("sd15_" + name, str(directory / (name + ".bin"))))
            self.unet_inputs = self.contexts[1].getInputName()
            if set(self.unet_inputs) != {"latent", "timestep", "text_emb"}:
                raise ValueError("Expected the Qualcomm SD 1.5 UNet inputs latent, timestep, and text_emb.")
        except BaseException:
            self.close()
            raise

    def close(self):
        for context in reversed(self.contexts):
            context.release()
        self.contexts.clear()

    def encode(self, prompt):
        tokens = self.tokenizer(prompt, padding="max_length", max_length=77, truncation=True).input_ids
        return self.contexts[0].Inference([np.asarray(tokens, dtype=np.float32)])[0].reshape(1, 77, 768)

    def denoise(self, latent, timestep, embedding):
        values = dict(latent=latent.permute(0, 2, 3, 1).contiguous().numpy(), timestep=np.array([[timestep]], dtype=np.float32), text_emb=embedding)
        inputs = [values[name] for name in self.unet_inputs]
        output = self.contexts[1].Inference(inputs)[0].reshape(1, 64, 64, 4)
        return torch.from_numpy(output).permute(0, 3, 1, 2).contiguous()

    def decode(self, latent):
        inputs = [latent.permute(0, 2, 3, 1).contiguous().numpy()]
        output = self.contexts[2].Inference(inputs)[0]
        return output.reshape(1, 512, 512, 3).clip(0, 1).copy()


def generate(model, prompt, negative_prompt, seed, steps, cfg):
    start = perf_counter()
    negative = model.encode(negative_prompt)
    positive = model.encode(prompt)
    encoded = perf_counter()
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    scheduler.set_timesteps(steps)
    latent = torch.randn((1, 4, 64, 64), generator=torch.Generator(device="cpu").manual_seed(seed))
    for timestep in scheduler.timesteps:
        t = int(timestep.item())
        unconditional = model.denoise(latent, t, negative)
        conditional = model.denoise(latent, t, positive)
        noise = unconditional + cfg * (conditional - unconditional)
        latent = scheduler.step(noise, t, latent).prev_sample
    denoised = perf_counter()
    image = model.decode(latent)
    end = perf_counter()
    return image, {"text_seconds": encoded - start, "denoise_seconds": denoised - encoded, "decode_seconds": end - denoised, "generation_seconds": end - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    request = json.load(sys.stdin)
    start = perf_counter()
    model = SD15(args.model)
    loaded = perf_counter()
    try:
        image, timings = generate(model, **request)
        timings.update(load_seconds=loaded - start, backend="qnn-htp", steps=request["steps"], seed=request["seed"])
        np.save(args.output, image, allow_pickle=False)
        Path(args.report).write_text(json.dumps(timings), encoding="utf-8")
    finally:
        model.close()


if __name__ == "__main__":
    main()
