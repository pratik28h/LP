def fractional(values,weights,W):
   n =len(values)

   items = []
   for i in range(n):
      ratio = values[i]/weights[i]
      items.append((values[i],weights[i],ratio))
   
   items.sort(key = lambda x:x[2],reverse=True)
   total = 0.0
   remain = W

   for value,weight,ratio in (items):
      if remain == 0:
         break
      if weight <= remain:
         total += value
         remain -= weight
      else:
         fraction = remain / weight
         total += value * fraction
         remain = 0
   return total

values = [50,120,120]
weight = [10,20,30]
5,6,4
170
10
W = 40
print(fractional(values,weight,W))