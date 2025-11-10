def isSafe(row,col,board,n):
   for r in range(row):
      if board[r][col] == "Q":
         return False
   r,c = row-1,col-1
   while r >= 0 and c >= 0:
      if board[r][c] == "Q":
         return False
      r -= 1
      c -= 1
   r,c = row-1,col+1
   while r >= 0 and c < n:
      if board[r][c] == "Q":
         return False
      r -= 1
      c += 1
   return True

def solveNQueens(board,row,n,ans):
   if row == n:
      copy_board = []
      for r in board:
         copy_board.append(r[:])
      ans.append(copy_board)
      return
   for col in range (n):
      if isSafe(row,col,board,n):
         board[row][col] = "Q"
         solveNQueens(board,row+1,n,ans)
         board[row][col] = "."

def solve(n):
   board = [["."]*n for _ in range(n)]
   ans = []
   solveNQueens(board,0,n,ans)
   return ans

n = int(input("Enter Number of Queens you want to place... "))
solution = solve(n)
for s in solution:
   for row in s:
      print(row)
   print()

