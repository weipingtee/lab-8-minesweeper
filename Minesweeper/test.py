import numpy as np
import random
import heapq
from collections import deque

class Minesweeper:
    def __init__(self, size=10, mines=10):
        self.size = size
        self.mines = mines
        self.board = np.zeros((size, size), dtype=int)  # 0 represents unrevealed cell
        self.visited = np.zeros((size, size), dtype=bool)
        self.generate_board()

    def generate_board(self):
        # Randomly place mines on the board
        mine_indices = random.sample(range(self.size * self.size), self.mines)
        for idx in mine_indices:
            row = idx // self.size
            col = idx % self.size
            self.board[row][col] = -1  # -1 represents a mine
        # Calculate the numbers in cells adjacent to mines
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == -1:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if (dr != 0 or dc != 0) and 0 <= r + dr < self.size and 0 <= c + dc < self.size and self.board[r + dr][c + dc] == -1:
                            count += 1
                self.board[r][c] = count

    def print_board(self, reveal=False):
        for r in range(self.size):
            for c in range(self.size):
                if not reveal and not self.visited[r][c]:
                    print("#", end=" ")
                else:
                    if self.board[r][c] == -1:
                        print("*", end=" ")  # Mine
                    elif self.board[r][c] == 0:
                        print(".", end=" ")  # Empty cell
                    else:
                        print(self.board[r][c], end=" ")
            print()

    def dfs(self, row, col):
        if row < 0 or row >= self.size or col < 0 or col >= self.size or self.visited[row][col]:
            return
        self.visited[row][col] = True
        if self.board[row][col] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    self.dfs(row + dr, col + dc)

    def bfs(self, row, col):
        queue = deque([(row, col)])
        while queue:
            r, c = queue.popleft()
            if not (0 <= r < self.size and 0 <= c < self.size) or self.visited[r][c]:
                continue
            self.visited[r][c] = True
            if self.board[r][c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        queue.append((r + dr, c + dc))

    def a_star(self, start):
        h = lambda node: abs(start[0] - node[0]) + abs(start[1] - node[1])
        g = 1  # Cost of each step
        queue = [(h(start), g, start)]
        while queue:
            _, g, (r, c) = heapq.heappop(queue)
            if not (0 <= r < self.size and 0 <= c < self.size) or self.visited[r][c]:
                continue
            self.visited[r][c] = True
            if self.board[r][c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        heapq.heappush(queue, (h((r + dr, c + dc)) + g, g + 1, (r + dr, c + dc)))

    def greedy(self, start):
        h = lambda node: abs(start[0] - node[0]) + abs(start[1] - node[1])
        queue = [(h(start), start)]
        while queue:
            _, (r, c) = heapq.heappop(queue)
            if not (0 <= r < self.size and 0 <= c < self.size) or self.visited[r][c]:
                continue
            self.visited[r][c] = True
            if self.board[r][c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if 0 <= r + dr < self.size and 0 <= c + dc < self.size:
                            heapq.heappush(queue, (h((r + dr, c + dc)), (r + dr, c + dc)))

    def count_moves(self, algorithm):
        moves = 0
        for r in range(self.size):
            for c in range(self.size):
                if not self.visited[r][c] and self.board[r][c] != -1:
                    moves += 1
                    if algorithm == "dfs":
                        self.dfs(r, c)
                    elif algorithm == "bfs":
                        self.bfs(r, c)
                    elif algorithm == "a_star":
                        self.a_star((r, c))
                    elif algorithm == "greedy":
                        self.greedy((r, c))
        return moves

# Example usage
game = Minesweeper(size=5, mines=5)
print("Minesweeper Board:")
game.print_board(reveal=False)
print("\nNumber of moves using DFS:", game.count_moves("dfs"))
print("Number of moves using BFS:", game.count_moves("bfs"))
print("Number of moves using A*:", game.count_moves("a_star"))
print("Number of moves using Greedy:", game.count_moves("greedy"))
