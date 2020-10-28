import numpy as np

def mean(arr):
  if len(arr.shape) == 1:
    return sum(arr)/len(arr)
  return sum(mean(xi) for xi in arr)/len(arr)

def cov(dataset):
  row, col = dataset.shape
  mean_ar = list()
  matrix_result = list()
  for r in range(row):
    mean_ar.append(mean(dataset[r]))
  for r1 in range(row):
    for r2 in range(row):
      result = 0
      for c in range(col):
        result += (dataset[r1][c] - mean_ar[r1])*(dataset[r2][c] - mean_ar[r2])
      matrix_result.append(result/(col-1))
  return np.array([matrix_result]).reshape(row, row)
  

X = np.array([2, 3, 6, 3, 7, 8])
Y = np.array([5, 7, 9, 6, 7, 8])

out = cov(np.array([X, Y]))

print("Correct output:\n", np.cov(X, Y))
print("Implementation output:\n", cov(np.array([X, Y])))