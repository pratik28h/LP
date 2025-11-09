def knapsack(values, weight, W):
    n = len(values)

    dp = []
    for _ in range(n + 1):
        row = []
        for j in range(W + 1):
            row.append(0)
        dp.append(row)

    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if weight[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weight[i - 1]],
                               dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][W]  


values  = [60, 100, 120]
weights = [10, 20, 30]
W = 50

print("Maximum Profit:", knapsack(values, weights, W))

