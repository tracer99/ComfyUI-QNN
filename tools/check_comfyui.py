"""Check this package with a real ComfyUI host; NPU checks are explicit options."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import threading
from time import perf_counter
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=ROOT)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--cancel", action="store_true")
    parser.add_argument("--model", default="sd15")
    options = parser.parse_args()
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(options.comfyui.resolve()))

    # Import the selected host only after selecting its path and device policy.
    from comfy.cli_args import args
    args.cpu = True
    import comfy.model_management as management
    import nodes

    assert asyncio.run(nodes.load_custom_node(str(options.package.resolve())))
    node = nodes.NODE_CLASS_MAPPINGS["SnapdragonSD15"]
    assert list(node.RETURN_TYPES) == ["IMAGE"]
    workflow = json.loads((options.package / "workflow-api.json").read_text(encoding="utf-8"))
    assert set(workflow["1"]["inputs"]) == set(node.INPUT_TYPES()["required"])
    result = {"registration": "passed"}
    if options.generate:
        inputs = dict(workflow["1"]["inputs"], model=options.model)
        start = perf_counter()
        image = node.execute(**inputs).result[0]
        assert tuple(image.shape) == (1, 512, 512, 3)
        assert image.isfinite().all().item()
        assert image.min().item() >= 0 and image.max().item() <= 1
        assert image.std().item() > 0.01
        result.update(generation="passed", node_seconds=perf_counter() - start)
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
                    node.execute(**dict(workflow["1"]["inputs"], model=options.model))
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
