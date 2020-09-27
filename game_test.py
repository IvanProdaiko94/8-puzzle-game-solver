import json
import numpy as np
import unittest

from .game import count_inversions, Solver


class MyTestCase(unittest.TestCase):
    def test_count_inversions(self):
        self.assertEqual(count_inversions([1, 2, 3, 4, 6, 8, 5, 7]), 3)
        self.assertEqual(count_inversions([1, 2, 3, 4, 5, 6, 7, 8]), 0)

    def test_Solver_constructor(self):
        init = np.array(json.loads('[[8, 1, 3],[4, 0, 2], [7, 6, 5]]'), dtype=np.int)
        solver = Solver(init)
        self.assertEqual(solver.init.hamming(), 5)
        self.assertEqual(solver.goal.hamming(), 0)

    def test_Solver_is_solvable(self):
        init = np.array(json.loads('[[1, 2, 3],[4, 5, 6], [8, 7, 0]]'), dtype=np.int)
        solver = Solver(init)
        self.assertEqual(solver.init.is_solvable(), False)

        init = np.array(json.loads('[[0, 1, 3],[4, 2, 5], [7, 8, 6]]'), dtype=np.int)
        solver = Solver(init)
        self.assertEqual(solver.init.is_solvable(), True)

        init = np.array(json.loads('[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 15, 14, 0]]'), dtype=np.int)
        solver = Solver(init)
        self.assertEqual(solver.init.is_solvable(), False)

        init = np.array(json.loads('[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]'), dtype=np.int)
        solver = Solver(init)
        self.assertEqual(solver.init.is_solvable(), True)


if __name__ == '__main__':
    unittest.main()
