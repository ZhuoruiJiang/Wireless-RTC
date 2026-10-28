import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("checkpoint_download",Path("scripts/download_checkpoint.py"))
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class CheckpointTests(unittest.TestCase):
    def test_download_provenance_and_skip_existing(self):
        with tempfile.TemporaryDirectory() as temp,patch(
                "urllib.request.urlopen",return_value=io.BytesIO(b"checkpoint")) as fetch,patch("sys.stdout"):
            path = downloader.download("cartpole_thrust",24,temp)
            self.assertEqual(path.read_bytes(),b"checkpoint")
            meta = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(meta["sha256"],hashlib.sha256(b"checkpoint").hexdigest())
            self.assertIn("worlds_l_cartpole_thrust.pkl",meta["url"])
            self.assertEqual(downloader.download("cartpole_thrust",24,temp,True),path)
            self.assertEqual(fetch.call_count,1)
            with self.assertRaises(ValueError):
                downloader.download("cartpole_thrust",24,temp)

    def test_failed_download_does_not_leave_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp,patch("urllib.request.urlopen",side_effect=OSError("network failed")):
            with self.assertRaises(OSError):
                downloader.download("mjc_walker",24,temp)
            self.assertFalse(list(Path(temp).rglob("*.pkl")))
            self.assertFalse(list(Path(temp).rglob("*.part")))
