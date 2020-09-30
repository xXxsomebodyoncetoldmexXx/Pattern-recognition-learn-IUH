import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from collections import Counter

def find_TH(arr1, arr2):
  arr1 = np.sort(arr1)
  arr2 = np.sort(arr2)
  glo_correct = -1
  threadHold = -1
  min_val = min(np.amin(arr1), np.amin(arr2))
  max_val = max(np.amax(arr1), np.amax(arr2))
  i_1, i_2 = (0, 0)
  for val in range(min_val-1, max_val+2):
    while i_1 < len(arr1) and val > arr1[i_1]:
      i_1 += 1
    while i_2 < len(arr2) and val > arr2[i_2]:
      i_2 += 1
    correct = len(arr1[:i_1]) + len(arr2[i_2:])
    if correct > glo_correct:
      glo_correct = correct
      threadHold = val
  return threadHold

def classification(arr1, arr2, threadHold):
  arr1 = np.sort(arr1)
  arr2 = np.sort(arr2)
  i_1, i_2 = (0, 0)
  while arr1[i_1] < threadHold:
    i_1 += 1
  while arr2[i_2] <= threadHold:
    i_2 += 1
  return (arr1[:i_1], arr2[i_2:])

def draw_hist(arr1, arr2):
  nw_arr1 = Counter(arr1).items()
  nw_arr2 = Counter(arr2).items()
  X,Y = zip(*nw_arr1)
  X2,Y2 = zip(*nw_arr2)
  bar_width = 0.9
  plt.bar(X,Y,bar_width,color="blue",alpha=0.75,label="a")
  bar_width = 0.8
  plt.bar(X2,Y2,bar_width,color="red",alpha=0.75,label="b")
  plt.legend(loc='upper right')
  plt.show()

# 1a
a = np.array([1,2,3,2,3,4,5,6,7])
b = np.array([5,5,6,6,7,8,9,9,8])
print("1a")
print("First array :", np.sort(a))
print("Second array:", np.sort(b))
print("Thread Hold :", find_TH(a, b))
nw_a, nw_b = classification(a, b, find_TH(a, b))
print("New first array :", nw_a)
print("New second array:", nw_b)
draw_hist(a, b)
print()


# 1b
a = np.random.normal(25, 15, size=300000).round(0).astype(np.int)
b = np.random.normal(75, 15, size=300000).round(0).astype(np.int)
print("1b")
print("First array :", np.sort(a))
print("Second array:", np.sort(b))
print("Thread Hold :", find_TH(a, b))
nw_a, nw_b = classification(a, b, find_TH(a, b))
print("New first array :", np.sort(nw_a))
print("New second array:", np.sort(nw_b))
draw_hist(a, b)
print()

# 1c
a = np.random.normal(15, 15, size=30000).round(0).astype(np.int)
b = np.random.normal(65, 15, size=30000).round(0).astype(np.int)
file = "inp.csv"

# Write to file
print("Writing to file:", file)
content = np.vstack((a, b))
df = pd.DataFrame(content)
df.to_csv(file,  na_rep="NAN!")

# Read from file
print("Reading from file:", file)
df = pd.read_csv(file)
a = np.array(df.iloc[0])
b = np.array(df.iloc[1])

print("1c")
print("First array :", np.sort(a))
print("Second array:", np.sort(b))
print("Thread Hold :", find_TH(a, b))
nw_a, nw_b = classification(a, b, find_TH(a, b))
print("New first array :", np.sort(nw_a))
print("New second array:", np.sort(nw_b))
draw_hist(a, b)
print()