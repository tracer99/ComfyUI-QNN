from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import qnn_sd15


class WorkerTests(unittest.TestCase):
    def test_unet_inputs_follow_exported_names(self):
        calls = []
        context = SimpleNamespace(Inference=lambda inputs: calls.append(inputs) or [np.zeros((1, 64, 64, 4), dtype=np.float32)])
        model = object.__new__(qnn_sd15.SD15)
        model.contexts = [None, context]
        model.unet_inputs = ["timestep", "text_emb", "latent"]
        latent = torch.arange(4 * 64 * 64, dtype=torch.float32).reshape(1, 4, 64, 64)
        embedding = np.ones((1, 77, 768), dtype=np.float32)
        output = model.denoise(latent, 999, embedding)
        np.testing.assert_array_equal(calls[0][0], [[999]])
        self.assertIs(calls[0][1], embedding)
        np.testing.assert_array_equal(calls[0][2], latent.permute(0, 2, 3, 1).numpy())
        self.assertEqual(output.shape, latent.shape)
        self.assertEqual(output.dtype, torch.float32)

    def test_partial_load_releases_contexts(self):
        released = []

        def load(name, path):
            if name == "sd15_unet":
                raise RuntimeError("Invalid context binary")
            return SimpleNamespace(release=lambda: released.append(name))

        with patch.object(qnn_sd15.CLIPTokenizer, "from_pretrained", return_value=None), \
                patch.object(qnn_sd15, "QNNConfig", SimpleNamespace(Config=lambda **kwargs: None), create=True), \
                patch.object(qnn_sd15, "Runtime", SimpleNamespace(HTP=0), create=True), \
                patch.object(qnn_sd15, "LogLevel", SimpleNamespace(ERROR=0), create=True), \
                patch.object(qnn_sd15, "ProfilingLevel", SimpleNamespace(OFF=0), create=True), \
                patch.object(qnn_sd15, "QNNContext", side_effect=load):
            with self.assertRaisesRegex(RuntimeError, "Invalid context binary"):
                qnn_sd15.SD15(Path("unused-model-directory"))
        self.assertEqual(released, ["sd15_text_encoder"])
