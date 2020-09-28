import json
import numpy as np
import unittest

from .solver import Solver


class SolverTestCase(unittest.TestCase):

    def test_Solver_unsolvable(self):
        init = np.array(json.loads('[[1, 2, 3], [4, 5, 6], [8, 7, 0]]'), dtype=np.int)
        solver = Solver(init)
        self.assertRaises(RuntimeError, solver.solve)

    # def test_Solver_constructor(self):
    #     init = np.array(json.loads('[[0, 1, 3],[4, 2, 5], [7, 8, 6]]'), dtype=np.int)
    #     solver = Solver(init)
    #     solver.solve()


if __name__ == '__main__':
    unittest.main()
