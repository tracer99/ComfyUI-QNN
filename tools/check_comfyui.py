"""Check this package with a real ComfyUI host; NPU checks are explicit options."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import threading
from time import perf_counter
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=ROOT)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--cancel", action="store_true")
    parser.add_argument("--family", choices=("sd15", "sdxl"), default="sd15")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(options.comfyui.resolve()))

    # Import the selected host only after selecting its path and device policy.
    from comfy.cli_args import args
    args.cpu = True
    import comfy.model_management as management
    import nodes

    assert asyncio.run(nodes.load_custom_node(str(options.package.resolve())))
    node = nodes.NODE_CLASS_MAPPINGS["SnapdragonSDXL" if options.family == "sdxl" else "SnapdragonSD15"]
    assert list(node.RETURN_TYPES) == ["IMAGE"]
    workflow = json.loads((options.package / ("workflow-sdxl-api.json" if options.family == "sdxl" else "workflow-api.json")).read_text(encoding="utf-8"))
    assert set(workflow["1"]["inputs"]) == set(node.INPUT_TYPES()["required"])
    result = {"registration": "passed"}
    module = sys.modules[node.__module__]
    with tempfile.TemporaryDirectory() as temporary, patch.object(module.folder_paths, "models_dir", temporary):
        root = Path(temporary) / "qnn"
        (root / "sd15").mkdir(parents=True)
        (root / "sd15" / "text_encoder.bin").touch()
        (root / "sdxl" / "text_encoder").mkdir(parents=True)
        for name, bundle in (("SnapdragonSD15", "sd15"), ("SnapdragonSDXL", "sdxl")):
            candidate = nodes.NODE_CLASS_MAPPINGS[name]
            assert candidate.INPUT_TYPES()["required"]["model"][1]["options"] == [bundle]
            for unsafe in ("../outside", "../../checkpoints", str(Path(temporary).resolve() / "outside")):
                try:
                    candidate.execute(unsafe, "test", "", 42, 6, 2)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Node accepted a model path outside models/qnn")
            with patch.object(module.subprocess, "Popen") as popen:
                try:
                    candidate.execute(bundle, "test", "", 42, 6, 2)
                except FileNotFoundError as error:
                    assert "Missing QNN model file" in str(error)
                else:
                    raise AssertionError("Incomplete bundle accepted")
                popen.assert_not_called()
    result["bundle_validation"] = "passed"
    if options.generate:
        inputs = dict(workflow["1"]["inputs"])
        if options.model:
            inputs["model"] = options.model
        start = perf_counter()
        image = node.execute(**inputs).result[0]
        resolution = 1024 if options.family == "sdxl" else 512
        assert tuple(image.shape) == (1, resolution, resolution, 3)
        assert image.isfinite().all().item()
        assert image.min().item() >= 0 and image.max().item() <= 1
        assert image.std().item() > 0.01
        result.update(generation="passed", node_seconds=perf_counter() - start)
        if options.output:
            Image.fromarray((image[0].numpy() * 255).round().astype(np.uint8)).save(options.output)
    if options.cancel:
        module = sys.modules[node.__module__]
        original_popen = module.subprocess.Popen
        processes = []

        def start_worker(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            processes.append(process)
            return process

        timer = threading.Timer(1, management.interrupt_current_processing)
        timer.start()
        try:
            with patch.object(module.subprocess, "Popen", side_effect=start_worker):
                try:
                    inputs = dict(workflow["1"]["inputs"])
                    if options.model:
                        inputs["model"] = options.model
                    node.execute(**inputs)
                except management.InterruptProcessingException:
                    result["cancellation"] = "passed"
                else:
                    raise AssertionError("Execution did not acknowledge cancellation")
        finally:
            timer.cancel()
            timer.join()
            management.interrupt_current_processing(False)
        assert processes and all(process.poll() is not None for process in processes)
        result["worker_cleanup"] = "passed"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
