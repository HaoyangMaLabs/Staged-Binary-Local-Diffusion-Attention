"""Measure structural coverage and exact-cell efficiency without training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BiAttn.topology import greedy_offsets, topology_offsets


ROUTES = (
    "H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8",
    "H8A", "H8B", "H8C", "H8D", "H8E", "H8F", "H8G",
)


def route_offsets(route: str, layer: int, total_layers: int) -> tuple[int, ...]:
    """Use the same source schedule as the trained model."""
    if route not in ROUTES or route == "H4":
        raise ValueError(f"route {route!r} needs a valid static schedule")
    return topology_offsets(layer, topology=route.lower(), total_layers=total_layers)


def dense_row(length: int, layers: int) -> dict[str, object]:
    per_layer = length * (length + 1) // 2
    total = per_layer * layers
    return {
        "route": "dense",
        "length": length,
        "layers": layers,
        "coverage": length,
        "coverage_auc": length * layers,
        "source_per_query_layer": (length + 1) / 2,
        "total_cells": total,
        "unique_cells": per_layer,
        "unique_cell_rate": per_layer / total,
        "repeat_cell_rate": 1.0 - per_layer / total,
    }


def sparse_row(route: str, length: int, layers: int) -> dict[str, object]:
    reachable_sets = [{0}]
    used_offsets: set[int] = set()
    offsets_by_layer: list[tuple[int, ...]] = []
    for layer in range(1, layers + 1):
        previous = reachable_sets[-1]
        if route == "H4":
            offsets = greedy_offsets(previous, used_offsets, length)
        else:
            offsets = route_offsets(route, layer, layers)
            used_offsets.update(offset for offset in offsets if offset)
        offsets_by_layer.append(offsets)
        expanded = {
            item + offset
            for item in previous
            for offset in offsets
            if item + offset < length
        }
        reachable_sets.append(expanded)
    reachable = reachable_sets[1:]
    cells: set[tuple[int, int]] = set()
    total_cells = 0
    for offsets in offsets_by_layer:
        layer_cells = {
            (query, query - offset)
            for query in range(length)
            for offset in offsets
            if offset <= query
        }
        total_cells += len(layer_cells)
        cells.update(layer_cells)
    return {
        "route": route,
        "length": length,
        "layers": layers,
        "coverage": len(reachable[-1]),
        "coverage_auc": sum(len(group) for group in reachable),
        "source_per_query_layer": total_cells / length / layers,
        "total_cells": total_cells,
        "unique_cells": len(cells),
        "unique_cell_rate": len(cells) / total_cells,
        "repeat_cell_rate": 1.0 - len(cells) / total_cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="256,512,1024,2048")
    parser.add_argument("--layers", default="4,8,16")
    parser.add_argument("--routes", default="dense," + ",".join(ROUTES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lengths = [int(item) for item in args.lengths.split(",")]
    layers = [int(item) for item in args.layers.split(",")]
    routes = [item.strip() for item in args.routes.split(",") if item.strip()]
    rows: list[dict[str, object]] = []
    for length in lengths:
        for depth in layers:
            for route in routes:
                if route == "dense":
                    row = dense_row(length, depth)
                else:
                    row = sparse_row(route, length, depth)
                rows.append(row)
                print(
                    f"{route:16s} N={length:4d} L={depth:2d} "
                    f"coverage={row['coverage']:4d} "
                    f"auc={row['coverage_auc']:6d} "
                    f"src/q={row['source_per_query_layer']:.2f} "
                    f"repeat={row['repeat_cell_rate']:.3f}"
                )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
