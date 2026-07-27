import unittest

import numpy as np

from edge_drl.env.requests import Task
from edge_drl.solver.kkt import KKTAllocator, ScheduledTask


class KKTAllocatorTest(unittest.TestCase):
    def test_single_node_compute_delay_matches_sum_square_formula(self):
        allocator = KKTAllocator(
            compute_capacity=np.array([10.0, 10.0]),
            bandwidth_capacity=np.array([[0.0, 100.0], [100.0, 0.0]]),
        )
        tasks = [
            ScheduledTask(
                Task(0, 0, 0, 0, 0, 1, 1.0, np.array([4.0, 0.0, 0.0]), np.zeros(3)),
                (0, -1, -1),
            ),
            ScheduledTask(
                Task(1, 0, 0, 0, 0, 1, 1.0, np.array([9.0, 0.0, 0.0]), np.zeros(3)),
                (0, -1, -1),
            ),
        ]
        result = allocator.allocate(tasks)
        self.assertAlmostEqual(result.compute_delay, (2.0 + 3.0) ** 2 / 10.0)
        self.assertAlmostEqual(result.transmission_delay, 0.0)


if __name__ == "__main__":
    unittest.main()

