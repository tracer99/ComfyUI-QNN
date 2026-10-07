from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper
import torch

import qnn_sdxl


class Session:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.feeds = None

    def get_inputs(self):
        return [SimpleNamespace(name=name, type=dtype) for name, dtype in self.inputs]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in self.outputs]

    def run(self, names, feeds):
        self.feeds = feeds
        return list(self.outputs.values())


class SDXLTests(unittest.TestCase):
    def test_graph_chain_converts_only_exported_nhwc_boundaries(self):
        f16 = "tensor(float16)"
        f32 = "tensor(float)"
        skip = np.arange(24, dtype=np.float16).reshape(1, 2, 3, 4)
        shared = [("silu_3", f16), ("encoder_hidden_states", f16)]
        common = Session([(name, f32) for name in ("timestep", "text_embeds", "time_ids")], {"silu_3": np.ones((1, 1280), dtype=np.float32)})
        down = Session([("sample", f16)] + shared, {"add_88": skip, "add_55": skip.copy()})
        mid = Session([("add_88", f16)] + shared, {"add_123": skip.copy()})
        up1 = Session([("add_88", f16), ("add_123", f16)] + shared, {"add_156": skip.copy()})
        noise = np.zeros((1, 4, 128, 128), dtype=np.float16)
        up2 = Session([("add_156", f16), ("add_55", f16)] + shared, {"out_sample": noise})
        model = object.__new__(qnn_sdxl.SDXL)
        model.sessions = [common, down, mid, up1, up2]
        latent = torch.ones((1, 4, 128, 128))
        result = model.denoise(latent, 999, (np.ones((1, 77, 2048)), np.ones((1, 1280))))
        np.testing.assert_array_equal(mid.feeds["add_88"], skip.transpose(0, 3, 1, 2))
        np.testing.assert_array_equal(up1.feeds["add_88"], mid.feeds["add_88"])
        np.testing.assert_array_equal(up2.feeds["add_156"], skip.transpose(0, 3, 1, 2))
        np.testing.assert_array_equal(up2.feeds["add_55"], skip)
        np.testing.assert_array_equal(common.feeds["time_ids"], [[1024, 1024, 0, 0, 1024, 1024]])
        self.assertEqual(down.feeds["sample"].dtype, np.float16)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, latent.shape)

    def test_conditioning_uses_named_penultimate_states_and_projected_pool(self):
        model = object.__new__(qnn_sdxl.SDXL)
        tokenizer = lambda *args, **kwargs: SimpleNamespace(input_ids=np.ones((1, 77), dtype=np.int64))
        model.tokenizers = [tokenizer, tokenizer]
        model.sessions = [Session([("input_ids", "tensor(int32)")], {"output_1": np.ones((1, 77, 768)), "output_0": np.zeros((1, 77, 768))}),
                          Session([("input_ids", "tensor(int32)")], {"output_1": np.full((1, 77, 1280), 2), "output_0": np.full((1, 1280), 3)})]
        hidden, pooled = model.encode("teapot")
        self.assertEqual(hidden.shape, (1, 77, 2048))
        self.assertTrue((hidden[..., :768] == 1).all())
        self.assertTrue((hidden[..., 768:] == 2).all())
        self.assertTrue((pooled == 3).all())
        self.assertEqual(model.sessions[0].feeds["input_ids"].dtype, np.int32)

    def test_decoder_scales_latents_and_normalizes_image(self):
        model = object.__new__(qnn_sdxl.SDXL)
        decoder = Session([("latent_sample", "tensor(float16)")], {"output_0": np.full((1, 3, 1024, 1024), -0.5, dtype=np.float16)})
        model.sessions = [decoder]
        with patch.object(model, "load") as load:
            image = model.decode(torch.full((1, 4, 128, 128), 0.13025))
        load.assert_called_once_with(qnn_sdxl.COMPONENTS[7:])
        np.testing.assert_allclose(decoder.feeds["latent_sample"], 1)
        self.assertEqual(image.shape, (1, 1024, 1024, 3))
        self.assertEqual(image.dtype, np.float32)
        self.assertTrue((image == 0.25).all())

    def test_context_rejects_external_paths_and_missing_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            for name, expected in (("../model.bin", ValueError), ("model.bin", FileNotFoundError), ("./model.bin", FileNotFoundError)):
                node = helper.make_node("EPContext", [], [], domain="com.microsoft", embed_mode=0, ep_cache_context=name, source="QNN")
                onnx.save(helper.make_model(helper.make_graph([node], "context", [], [])), model)
                with self.assertRaises(expected):
                    qnn_sdxl.validate_context(model, root)
            (root / "model.bin").touch()
            qnn_sdxl.validate_context(model, root)

    def test_partial_session_load_releases_existing_sessions(self):
        model = object.__new__(qnn_sdxl.SDXL)
        model.directory = Path("unused")
        model.options = None
        model.load_seconds = 0
        model.sessions = [object()]
        with patch.object(qnn_sdxl.ort, "InferenceSession", side_effect=[object(), RuntimeError("invalid context")]):
            with self.assertRaisesRegex(RuntimeError, "invalid context"):
                model.load(("text_encoder", "text_encoder_2"))
        self.assertEqual(model.sessions, [])

    def test_existing_provider_is_reused_and_cpu_fallback_disabled(self):
        options = SimpleNamespace(add_provider_for_devices=lambda *args: None)
        entries = []
        options.add_session_config_entry = lambda *args: entries.append(args)
        device = SimpleNamespace(ep_name="QNNExecutionProvider", device=SimpleNamespace(type=qnn_sdxl.ort.OrtHardwareDeviceType.NPU))
        with patch.object(qnn_sdxl, "qnn", SimpleNamespace(get_qnn_htp_path=lambda: "htp.dll")), \
                patch.object(qnn_sdxl.ort, "get_ep_devices", return_value=[device]), \
                patch.object(qnn_sdxl.ort, "SessionOptions", return_value=options), \
                patch.object(qnn_sdxl.ort, "register_execution_provider_library") as register:
            self.assertIs(qnn_sdxl.session_options(), options)
        register.assert_not_called()
        self.assertIn(("session.disable_cpu_ep_fallback", "1"), entries)

    def test_missing_npu_fails_instead_of_using_cpu(self):
        device = SimpleNamespace(ep_name="QNNExecutionProvider", device=SimpleNamespace(type=qnn_sdxl.ort.OrtHardwareDeviceType.CPU))
        with patch.object(qnn_sdxl, "qnn", object()), patch.object(qnn_sdxl.ort, "get_ep_devices", return_value=[device]):
            with self.assertRaisesRegex(RuntimeError, "No QNN NPU"):
                qnn_sdxl.session_options()
