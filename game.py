import numpy as np
import heapq
from scipy.spatial.distance import cdist


def count_inversions(x: []) -> int:
    inv_count = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] > x[j]:
                inv_count += 1
    return inv_count


class Board:
    def __init__(self, tiles: np.ndarray, goal: np.ndarray, is_goal=False):
        if tiles.shape[0] != tiles.shape[1]:
            raise AssertionError("input is not squared")
        self.__state = tiles
        self.__goal = goal
        self.__is_goal = is_goal
        # self.__manhattan = cdist(tiles, goal, metric='cityblock')
        self.__hamming = 0
        # calculate hamming distance
        i = 1
        for row in self.__state:
            for item in row:
                if item != 0 and i != item:
                    self.__hamming += 1
                i += 1

    def __repr__(self) -> str:  # string representation of this board
        return "\n" + np.array_str(self.__state) + "\n"

    def __str__(self) -> str:  # string representation of this board
        return "\n" + np.array_str(self.__state) + "\n"

    def tile_at(self, row: int, col: int) -> int:  # tile at (row, col) or 0 if blank
        return self.__state[row][col]

    def size(self) -> int:  # board size n
        shape = self.__state.shape
        return shape[0]

    def hamming(self) -> int:  # number of tiles out of place
        return self.__hamming

    def manhattan(self) -> int:  # sum of Manhattan distances between tiles and goal
        # return self.__manhattan
        pass

    def is_goal(self) -> int:  # is this board the goal board?
        return self.__is_goal

    def __eq__(self, other) -> bool:  # all neighboring boards
        comparison = self.__state == other
        return comparison.all()

    def neighbors(self) -> []:  # all neighboring boards
        result = []
        # find the index of zero element
        zero_at = np.argwhere(self.__state == 0)[0]
        size = self.size() - 1
        # filter out possible positions from those that are available
        available_position_changes = list(filter(lambda pos: 0 <= pos[0] <= size and 0 <= pos[1] <= size, [
            [zero_at[0] - 1, zero_at[1]],
            [zero_at[0] + 1, zero_at[1]],
            [zero_at[0], zero_at[1] - 1],
            [zero_at[0], zero_at[1] + 1],
        ]))
        # create new neighboring boards
        for pos in available_position_changes:
            next_state = np.copy(self.__state)
            next_state[zero_at[0]][zero_at[1]], next_state[pos[0]][pos[1]] = next_state[pos[0]][pos[1]], next_state[zero_at[0]][zero_at[1]]
            result.append(Board(tiles=next_state, goal=self.__goal, is_goal=next_state == self.__goal))

        return result

    def is_solvable(self) -> bool:  # is this board solvable?
        row_order_arr = self.__state.flatten()
        n_inv = count_inversions(row_order_arr)
        if self.size() % 2 != 0:  # size is odd
            # we’ll consider the case when the board size n is an odd integer.
            # In this case, each move changes the number of inversions by an even number.
            # Thus, if a board has an odd number of inversions,
            # it is unsolvable because the goal board has an even number of inversions (zero).
            return n_inv % 2 == 0
        zero_at = np.argwhere(self.__state == 0)[0]
        # we’ll consider the case when the board size n is an even integer.
        # In this case, the parity of the number of inversions is not invariant.
        # However, the parity of the number of inversions plus the row of the blank square(indexed starting at 0)
        # is invariant: each move changes this sum by an even number.
        return (n_inv + zero_at[0]) % 2 == 0


class Solver:
    def __init__(self, initial: np.array):  # find a solution to the initial board (using the A* algorithm)
        if initial.shape[0] != initial.shape[1]:
            raise AssertionError("input is not squared")

        # create goal board
        side_size = initial.shape[0]
        number_of_elements = side_size * side_size
        goal = np.arange(1, number_of_elements+1, dtype=np.int).reshape((side_size, side_size))
        # last element of goal board is always 0
        goal[side_size - 1][side_size - 1] = 0

        self.init = Board(tiles=initial, goal=goal, is_goal=goal == initial)
        self.goal = Board(tiles=goal, goal=goal, is_goal=True)
        self.__solution_steps: [Board] = []

    def moves(self) -> int:  # min number of moves to solve initial board
        return len(self.__solution_steps)

    def solution(self) -> [Board]:  # sequence of boards in a shortest solution
        if not self.init.is_solvable():
            raise RuntimeError("this board is not solvable")
        return self.__solution_steps
