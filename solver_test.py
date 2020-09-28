import json
import numpy as np
import unittest

from .solver import Solver, AvailableMetrics


class SolverTestCase(unittest.TestCase):

    def test_Solver_unsolvable(self):
        init = np.array(json.loads('[[1, 2, 3], [4, 5, 6], [8, 7, 0]]'), dtype=np.int)
        solver = Solver(init)
        self.assertRaises(RuntimeError, solver.solve)

    def test_Solver_solvable_3by3_manhattan(self):
        init = np.array(json.loads('[[0, 1, 3], [4, 2, 5], [7, 8, 6]]'), dtype=np.int)
        solver = Solver(init)
        solver.solve()
        number_of_moves = solver.moves()
        self.assertEqual(number_of_moves, 4)
        # print("Start:", solver.solution())

    def test_Solver_solvable_3by3_manhattan_2(self):
        init = np.array(json.loads('[[1, 2, 3], [0, 7, 6], [5, 4, 8]]'), dtype=np.int)
        solver = Solver(init)
        solver.solve()
        number_of_moves = solver.moves()
        self.assertEqual(number_of_moves, 7)
        # print("Start:", solver.solution())

    def test_Solver_solvable_3by3_hamming(self):
        init = np.array(json.loads('[[0, 1, 3], [4, 2, 5], [7, 8, 6]]'), dtype=np.int)
        solver = Solver(init, metric=AvailableMetrics.HammingDistance)
        solver.solve()
        number_of_moves = solver.moves()
        self.assertEqual(number_of_moves, 4)
        # print("Start:", solver.solution())

    def test_Solver_solvable_3by3_hamming_2(self):
        init = np.array(json.loads('[[1, 2, 3], [0, 7, 6], [5, 4, 8]]'), dtype=np.int)
        solver = Solver(init, metric=AvailableMetrics.HammingDistance)
        solver.solve()
        number_of_moves = solver.moves()
        self.assertEqual(number_of_moves, 7)
        # print("Start:", solver.solution())


if __name__ == '__main__':
    unittest.main()
