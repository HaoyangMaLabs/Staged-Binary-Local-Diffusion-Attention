"""Keep structural scanning and training on identical source schedules."""

import unittest

from BiAttn.topology import greedy_internal_offsets
from BinaryDiffusionAttention.run_graph_coverage import ROUTES, sparse_row
from BinaryDiffusionAttention.run_mini_experiment import internal_route_stats


class TopologyTest(unittest.TestCase):
    def test_graph_and_training_agree_for_all_internal_routes(self):
        for route in ROUTES:
            for length, depth in ((64, 4), (256, 8)):
                with self.subTest(route=route, length=length, depth=depth):
                    graph = sparse_row(route, length, depth)
                    sources, coverage = internal_route_stats(route.lower(), seq_len=length, layers=depth)
                    self.assertEqual(graph["coverage"], coverage)
                    self.assertEqual(graph["source_per_query_layer"], sources)

    def test_greedy_schedule_handles_exhausted_offsets(self):
        self.assertEqual(greedy_internal_offsets(1, 3), [(0,), (0,), (0,)])
        self.assertEqual(greedy_internal_offsets(2, 3), [(0, 1), (0,), (0,)])
