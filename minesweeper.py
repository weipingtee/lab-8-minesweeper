# from collections import deque
# import time

# def is_valid(board, row, col, num):
#     # Check if the number is already in the row
#     for x in range(9):
#         if board[row][x] == num:
#             return False
    
#     # Check if the number is already in the column
#     for x in range(9):
#         if board[x][col] == num:
#             return False
    
#     # Check if the number is already in the 3x3 grid
#     start_row, start_col = 3 * (row // 3), 3 * (col // 3)
#     for i in range(3):
#         for j in range(3):
#             if board[i + start_row][j + start_col] == num:
#                 return False
    
#     return True

# def find_empty_cell(board):
#     for i in range(9):
#         for j in range(9):
#             if board[i][j] == 0:
#                 return (i, j)
#     return None

# def print_board(board):
#     for i in range(9):
#         if i % 3 == 0 and i != 0:
#             print("- - - - - - - - - - - - ")
#         for j in range(9):
#             if j % 3 == 0 and j != 0:
#                 print(" | ", end="")
#             if j == 8:
#                 print(board[i][j])
#             else:
#                 print(str(board[i][j]) + " ", end="")

# def solve_sudoku_dfs(board):
#     start_time = time.time()
#     steps_taken = [0]  # Steps counter
#     path_length = [0]  # Path length counter

#     def dfs(board, steps_taken, path_length):
#         empty_cell = find_empty_cell(board)
#         if not empty_cell:
#             return True  # No empty cell left, puzzle solved
        
#         row, col = empty_cell
#         path_length[0] += 1  # Increment path length
        
#         for num in range(1, 10):
#             if is_valid(board, row, col, num):
#                 board[row][col] = num
#                 steps_taken[0] += 1  # Increment steps
#                 if dfs(board, steps_taken, path_length):
#                     return True
#                 board[row][col] = 0  # Backtrack
#         return False
    
#     if dfs(board, steps_taken, path_length):
#         end_time = time.time()
#         print("\nSolution using DFS:")
#         print_board(board)
#         print("Time taken: {:.6f} seconds".format(end_time - start_time))
#         print("Total steps taken:", steps_taken[0])
#         print("Path length:", path_length[0])
#     else:
#         print("No solution exists")

# def solve_sudoku_bfs(board):
#     start_time = time.time()
#     steps_taken = [0]  # Steps counter
#     path_length = [0]  # Path length counter
    
#     queue = deque([(board, 0, 0)])
#     while queue:
#         current_board, row, col = queue.popleft()
#         steps_taken[0] += 1  # Increment steps
#         if row == 9:  # Completed all rows
#             end_time = time.time()
#             print("\nSolution using BFS:")
#             print_board(current_board)
#             print("Time taken: {:.6f} seconds".format(end_time - start_time))
#             print("Total steps taken:", steps_taken[0])
#             print("Path length:", path_length[0])
#             return current_board
#         if current_board[row][col] == 0:
#             for num in range(1, 10):
#                 if is_valid(current_board, row, col, num):
#                     current_board[row][col] = num
#                     queue.append((list(map(list, current_board)), row + (col + 1) // 9, (col + 1) % 9))
#                     current_board[row][col] = 0
#                     path_length[0] += 1  # Increment path length
#     print("No solution exists")

# # Example puzzle
# board = [
#     [5, 3, 0, 0, 7, 0, 0, 0, 0],
#     [6, 0, 0, 1, 9, 5, 0, 0, 0],
#     [0, 9, 8, 0, 0, 0, 0, 6, 0],
#     [8, 0, 0, 0, 6, 0, 0, 0, 3],
#     [4, 0, 0, 8, 0, 3, 0, 0, 1],
#     [7, 0, 0, 0, 2, 0, 0, 0, 6],
#     [0, 6, 0, 0, 0, 0, 2, 8, 0],
#     [0, 0, 0, 4, 1, 9, 0, 0, 5],
#     [0, 0, 0, 0, 8, 0, 0, 7, 9]
# ]

# print("Original Sudoku Puzzle:")
# print_board(board)

# solve_sudoku_dfs(board.copy())
# solve_sudoku_bfs(board.copy())

import numpy as np
from collections import deque
import time
import heapq

def is_valid(board, row, col, num):
    # Check if the number is already in the row
    if num in board[row]:
        return False
    
    # Check if the number is already in the column
    if num in board[:, col]:
        return False
    
    # Check if the number is already in the 3x3 grid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[start_row:start_row+3, start_col:start_col+3]:
        return False
    
    return True

def find_empty_cell(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

def h(board):
    # A* heuristic: number of empty cells
    return np.count_nonzero(board == 0)

def solve_sudoku_astar(board):
    start_time = time.time()
    steps_taken = [0]  # Steps counter
    path_length = [0]  # Path length counter

    priority_queue = [(h(board), board.copy(), 0)]
    while priority_queue:
        _, current_board, depth = priority_queue.pop(0)
        steps_taken[0] += 1  # Increment steps
        path_length[0] += 1  # Increment path length
        
        empty_cell = find_empty_cell(current_board)
        if not empty_cell:
            end_time = time.time()
            print("\nSolution using A*:")
            print_board(current_board)
            print("Time taken: {:.6f} seconds".format(end_time - start_time))
            print("Total steps taken:", steps_taken[0])
            print("Path length:", path_length[0])
            return current_board
        
        row, col = empty_cell
        for num in range(1, 10):
            if is_valid(current_board, row, col, num):
                new_board = current_board.copy()
                new_board[row][col] = num
                heapq.heappush(priority_queue, (h(new_board), new_board, depth + 1))
    
    print("No solution exists")

def print_board(board):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - ")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")

# Example puzzle
board = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
])

print("Original Sudoku Puzzle:")
print_board(board)

solve_sudoku_astar(board.copy())
