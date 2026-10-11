"""End-to-end CPU check for the configurable long-context training driver."""

import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from BinaryDiffusionAttention.train_wikitext_1m_pass_compare import one_mode


class TrainingDriverTest(unittest.TestCase):
    def test_both_modes_train_and_write_validation_records_without_cuda(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split = root / "split.txt"
            split.write_text("abcd" * 8 + "a")
            paths = {name: str(split) for name in ("train", "valid", "test")}
            for mode in ("causal", "sbda"):
                with self.subTest(mode=mode):
                    output = root / f"{mode}.json"
                    args = argparse.Namespace(
                        mode=mode, device="cpu", seed=1, data_root=root,
                        seq_len=16, passes=1, dim=8, layers=3, heads=2,
                        ff_hidden=16, lr=0.0003, output=output,
                    )
                    with patch("torch.cuda.is_available", return_value=False), patch(
                        "BinaryDiffusionAttention.train_wikitext_1m_pass_compare.download_wikitext",
                        return_value=paths,
                    ), redirect_stdout(io.StringIO()):
                        result = one_mode(args)
                    self.assertEqual(result["status"], "ok")
                    self.assertEqual(result["dtype"], "fp32")
                    self.assertEqual(result["complete_windows"], 2)
                    self.assertEqual(result["dropped_train_tokens"], 0)
                    self.assertTrue(math.isfinite(result["pass_records"][0]["dev_loss"]))
                    self.assertEqual(json.loads(output.read_text()), result)
