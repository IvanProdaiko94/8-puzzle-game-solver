import numpy as np
from enum import Enum


def count_inversions(x: []) -> int:
    inv_count = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] > x[j]:
                inv_count += 1
    return inv_count


def hamming(x: []) -> int:
    result = 0
    i = 1
    for item in x:
        if item != 0 and i != item:
            result += 1
        i += 1
    return result


def manhattan(x: []) -> int:
    return sum(abs((val - 1) % 3 - i % 3) + abs((val - 1) // 3 - i // 3) for i, val in enumerate(x) if val)


class AvailableMetrics(Enum):
    HammingDistance = 'hamming'
    Manhattan = 'manhattan'


class Board:
    def __init__(self, tiles: np.ndarray, goal: np.ndarray, prev=None, metric=AvailableMetrics.Manhattan, is_goal=False):
        if tiles.shape[0] != tiles.shape[1]:
            raise AssertionError("input is not squared")
        self.__state = tiles
        self.__goal = goal
        self.__is_goal = is_goal
        self.__prev = prev
        self.__metric = metric
        self.cost = 0
        if prev is not None:
            self.cost = prev.cost
        # calculate manhattan distance
        if metric == AvailableMetrics.HammingDistance:
            self.cost += hamming(self.__state.flatten())
        elif metric == AvailableMetrics.Manhattan:
            self.cost += manhattan(self.__state.flatten())
        else:
            raise BaseException("distance metric is not valid")

    def __repr__(self) -> str:  # string representation of this board
        return "\n" + np.array_str(self.__state) + "\n"

    def __str__(self) -> str:  # string representation of this board
        return "\n" + np.array_str(self.__state) + "\n"

    def __lt__(self, other) -> bool:
        return self.cost < other.cost

    def __eq__(self, other) -> bool:  # all neighboring boards
        comparison = self.__state == other.state()
        return comparison.all()

    def prev(self):
        return self.__prev

    def state(self) -> np.ndarray:
        return self.__state

    def tile_at(self, row: int, col: int) -> int:  # tile at (row, col) or 0 if blank
        return self.__state[row][col]

    def size(self) -> int:  # board size n
        shape = self.__state.shape
        return shape[0]

    def is_goal(self) -> int:  # is this board the goal board?
        return self.__is_goal

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
            result.append(
                Board(
                    tiles=next_state,
                    goal=self.__goal,
                    prev=self,
                    metric=self.__metric,
                    is_goal=next_state == self.__goal,
                ),
            )

        return result

    def is_solvable(self) -> bool:  # is this board solvable?
        row_order_arr = []
        for row in self.__state:
            for item in row:
                if item != 0:
                    row_order_arr.append(item)

        n_inv = count_inversions(row_order_arr)
        if self.size() % 2 != 0:  # size is odd
            # we’ll consider the case when the board size n is an odd integer.
            # In this case, each move changes the number of inversions by an even number.
            # Thus, if a board has an odd number of inversions,
            # it is unsolvable because the goal board has an even number of inversions (zero).
            is_solvable = n_inv % 2 == 0
            return is_solvable
        zero_at = np.argwhere(self.__state == 0)[0]
        # we’ll consider the case when the board size n is an even integer.
        # In this case, the parity of the number of inversions is not invariant.
        # However, the parity of the number of inversions plus the row of the blank square(indexed starting at 0)
        # is invariant: each move changes this sum by an even number.
        # That is, when n is even, an n - by - n board is solvable
        # if and only if the number of inversions plus the row of the blank square is odd.
        is_solvable = (n_inv + zero_at[0]) % 2 != 0
        return is_solvable
