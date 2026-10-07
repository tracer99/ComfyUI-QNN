import hashlib
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
import zipfile

from tools.build_release import ROOT, build


class ReleaseTests(unittest.TestCase):
    def test_archive_contents_checksum_and_repeatability(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            archive = build(Path(directory) / "first")
            second = build(Path(directory) / "second")
            self.assertEqual(archive.read_bytes(), second.read_bytes())
            checksum = archive.with_suffix(".zip.sha256").read_text().split()[0]
            self.assertEqual(checksum, hashlib.sha256(archive.read_bytes()).hexdigest())
            with zipfile.ZipFile(archive) as package:
                expected = {"comfyui_qnn/" + name for name in config["tool"]["release"]["files"]}
                self.assertEqual(set(package.namelist()), expected)
                self.assertIsNone(package.testzip())
                for name in package.namelist():
                    self.assertNotIn("..", Path(name).parts)
                    self.assertFalse(name.endswith((".bin", ".safetensors", ".pyc")))
                api = json.loads(package.read("comfyui_qnn/workflow-api.json"))
                self.assertEqual(api["1"]["class_type"], "SnapdragonSD15")
                self.assertEqual(api["2"]["inputs"]["images"], ["1", 0])

    def test_mismatched_tag_fails_before_packaging(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must match"):
                build(directory, "v999.0.0")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_host_requirements_do_not_install_worker_packages(self):
        requirements = (ROOT / "requirements.txt").read_text().splitlines()
        self.assertFalse([line for line in requirements if line.strip() and not line.startswith("#")])
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(config["project"]["dependencies"], [])
