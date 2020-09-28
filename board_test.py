import json
import numpy as np
import unittest

from .board import count_inversions, hamming, manhattan, Board, AvailableMetrics


class BoardTestCase(unittest.TestCase):
    def test_count_inversions(self):
        self.assertEqual(count_inversions([1, 2, 3, 4, 6, 8, 5, 7]), 3)
        self.assertEqual(count_inversions([1, 2, 3, 4, 5, 6, 7, 8]), 0)

    def test_hamming(self):
        self.assertEqual(hamming([1, 2, 3, 4, 5, 6, 7, 8, 0]), 0)
        self.assertEqual(hamming([8, 1, 3, 4, 0, 2, 7, 6, 5]), 5)

    def test_manhattan(self):
        self.assertEqual(manhattan([1, 2, 3, 4, 5, 6, 7, 8, 0]), 0)
        self.assertEqual(manhattan([8, 1, 3, 4, 0, 2, 7, 6, 5]), 10)

    def test_Board_is_solvable(self):
        init = np.array(json.loads('[[1, 2, 3],[4, 5, 6], [8, 7, 0]]'), dtype=np.int)
        goal = np.array(json.loads('[[1, 2, 3],[4, 5, 6], [7, 8, 0]]'), dtype=np.int)
        board = Board(init, goal, prev=None, metric=AvailableMetrics.Manhattan, is_goal=False)
        self.assertEqual(board.is_solvable(), False)

        init = np.array(json.loads('[[0, 1, 3],[4, 2, 5], [7, 8, 6]]'), dtype=np.int)
        board = Board(init, goal, prev=None, metric=AvailableMetrics.Manhattan, is_goal=False)
        self.assertEqual(board.is_solvable(), True)

        init = np.array(json.loads('[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 15, 14, 0]]'), dtype=np.int)
        goal = np.array(json.loads('[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]'), dtype=np.int)
        board = Board(init, goal, prev=None, metric=AvailableMetrics.Manhattan, is_goal=False)
        self.assertEqual(board.is_solvable(), False)

        init = np.array(json.loads('[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]'), dtype=np.int)
        board = Board(init, goal, prev=None, metric=AvailableMetrics.Manhattan, is_goal=False)
        self.assertEqual(board.is_solvable(), True)


if __name__ == '__main__':
    unittest.main()
