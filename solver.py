import numpy as np
import heapq
from board import Board, AvailableMetrics


class Solver:

    # find a solution to the board (using the A* algorithm)
    def __init__(self, initial: np.array, metric=AvailableMetrics.HammingDistance):
        if initial.shape[0] != initial.shape[1]:
            raise BaseException("input is not squared")

        # create goal board
        side_size = initial.shape[0]
        number_of_elements = side_size * side_size
        goal = np.arange(1, number_of_elements+1, dtype=np.int).reshape((side_size, side_size))
        # last element of goal board is always 0
        goal[side_size - 1][side_size - 1] = 0

        self.init = Board(tiles=initial, goal=goal, prev=None, metric=metric, is_goal=goal == initial)
        self.goal = Board(tiles=goal, goal=goal, prev=None, metric=metric, is_goal=True)
        self.result: Board

    def solve(self) -> Board:
        if not self.init.is_solvable():
            raise RuntimeError("this board is not solvable")

        visited = {}
        q = []

        heapq.heappush(q, self.init)
        curr: Board = None
        while True:
            curr = heapq.heappop(q)
            visited[str(curr)] = True
            print(curr)
            if curr == self.goal:
                break
            neighbors = curr.neighbors()
            for board in neighbors:
                if board != curr.prev() and str(board) not in visited:
                    heapq.heappush(q, board)
        self.result = curr

    def moves(self) -> int:  # min number of moves to solve initial board
        moves = 1
        prev = self.result.prev()
        while prev is not None:
            moves += 1
            prev = prev.prev()
        return moves

    def solution(self) -> [Board]:  # sequence of boards in a shortest solution
        result = []
        prev = self.result
        while prev is not None:
            result.append(prev)
            prev = prev.prev()
        return result
