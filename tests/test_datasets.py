"""Dataset boundaries and cache integrity used by both training drivers."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from BinaryDiffusionAttention.datasets import (
    complete_windows, download_file, download_wikitext, encode, make_vocab,
)


class DatasetTest(unittest.TestCase):
    def test_vocab_is_sorted_and_unknown_validation_chars_map_to_zero(self):
        vocab = make_vocab("b a\nb")
        self.assertEqual(vocab, {"<unk>": 0, "\n": 1, " ": 2, "a": 3, "b": 4})
        torch.testing.assert_close(encode("ab?", vocab), torch.tensor([3, 4, 0]))

    def test_windows_supervise_each_complete_next_token_once(self):
        ids = torch.arange(12)
        windows, dropped = complete_windows(ids, 4)
        self.assertEqual(dropped, 3)
        torch.testing.assert_close(torch.cat([w[1:] for w in windows]), torch.arange(1, 9))
        torch.testing.assert_close(windows[0][-1:], windows[1][:1])

    def test_incomplete_and_empty_splits_have_no_windows(self):
        self.assertEqual(complete_windows(torch.arange(3), 4), ((), 2))
        self.assertEqual(complete_windows(torch.tensor([]), 4), ((), 0))
        with self.assertRaises(ValueError):
            complete_windows(torch.arange(3), 0)

    def test_cached_splits_do_not_require_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split_root = root / "wikitext-2-raw"
            split_root.mkdir()
            for split in ("train", "valid", "test"):
                (split_root / f"wiki.{split}.raw").write_bytes(b"x" * 1024)
            with patch("urllib.request.urlopen") as request:
                paths = download_wikitext(root)
            request.assert_not_called()
            self.assertEqual(set(paths), {"train", "valid", "test"})

    def test_failed_transfer_preserves_existing_file_and_cleans_temporary(self):
        class Interrupted(io.BytesIO):
            def read(self, *args):
                if self.tell():
                    raise OSError("interrupted")
                return super().read(*args)

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "data.txt"
            target.write_bytes(b"old")
            with patch("urllib.request.urlopen", return_value=Interrupted(b"new data")):
                with self.assertRaises(RuntimeError):
                    download_file("https://example.test/data", target, minimum_bytes=4)
            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_short_transfer_is_rejected_and_successful_transfer_replaces_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "data.txt"
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"x")):
                with self.assertRaises(RuntimeError):
                    download_file("https://example.test/data", target, minimum_bytes=4)
            self.assertFalse(target.exists())
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"valid data")):
                download_file("https://example.test/data", target, minimum_bytes=4)
            self.assertEqual(target.read_bytes(), b"valid data")
