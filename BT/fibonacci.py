def recursive(n):
   if n == 0:
      return [0]
   if n == 1:
      return [0,1]
   seq = recursive(n-1)
   seq.append(seq[-1] + seq[-2])
   return seq

def nonrec(n):
   n1,n2 = 0,1
   print(n1,n2,end=" " )
   for i in range(2,n):
      n3 = n1 + n2
      print(n3,end=" ")
      n1,n2 = n2,n3

n = int(input("Enter a number you want..."))

print(recurrsive(n))

def recurrsive(n):
   if n == 0:
      return [0]
   if n == 1:
      return [0,1]
   seq = recurrsive(n-1)
   seq.append(seq[-1] + seq[-2])
   return seq
