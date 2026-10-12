"""Count original WikiText-2 articles, separately from concatenated LM windows."""
import argparse
import json
import re
from pathlib import Path


def percentile(values, q):
    pos = (len(values) - 1) * q
    lo = int(pos)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)


def summarize(values):
    values = sorted(values)
    bounds = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 1048576]
    return {
        "count": len(values), "total": sum(values), "min": min(values),
        "mean": sum(values) / len(values),
        **{name: percentile(values, q) for name, q in
           [("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99)]},
        "max": max(values),
        "bins": {f"({low}, {high}]": sum(low < n <= high for n in values)
                 for low, high in zip([0] + bounds[:-1], bounds)},
        "above_1m": sum(n > 1048576 for n in values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="BinaryDiffusionAttention/data/wikitext-2-raw")
    parser.add_argument("--output", default="BinaryDiffusionAttention/results/wikitext2_length_distribution.json")
    args = parser.parse_args()
    result = {"unit": "Unicode characters (the experiments use character tokens)",
              "articles": "Split at single-equals top-level titles; retain headings, whitespace and newlines.",
              "training": "Concatenate the entire split, then cut fixed windows; article boundaries do not reset attention.",
              "splits": {}}
    for split in ["train", "valid", "test"]:
        text = (Path(args.data_root) / f"wiki.{split}.raw").read_text(encoding="utf-8")
        starts = []
        offset = 0
        for line in text.splitlines(keepends=True):
            if re.fullmatch(r"= [^=].* =", line.strip()):
                starts.append(offset)
            offset += len(line)
        assert starts, "No article titles found"
        chunks = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
        result["splits"][split] = {
            "split_characters": len(text), "leading_characters_before_first_title": starts[0],
            "article_characters": summarize([len(c) for c in chunks]),
            "article_whitespace_words": summarize([len(c.split()) for c in chunks]),
            "nonempty_line_characters": summarize([len(line) for line in text.splitlines() if line.strip()]),
            "fixed_windows": {str(n): {"count": (len(text)-1)//n,
                "dropped_characters": (len(text)-1)%n} for n in [65536, 1048576]},
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
