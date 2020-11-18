import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

def find_TH(arr1, arr2):
  class1 = np.sort(arr1)
  class2 = np.sort(arr2)
  data_set = set(class1).union(class2)
  if class1[0] > class2[0]:
    class1, class2 = class2, class1
  class1, class2 = Counter(class1), Counter(class2)
  # v <= tp: class1
  # v >  tp: class2
  num_of_errors = arr1.size + arr2.size
  thread_hold = None
  for tp in data_set:
    noe = 0
    for v in class1:
      if (v > tp):
        noe += class1[v]
    for v in class2:
      if (v <= tp):
        noe += class2[v]
    if noe < num_of_errors:
      num_of_errors = noe
      thread_hold = tp
  return (thread_hold, num_of_errors)

def classification(arr1, arr2, threadHold):
  arr1 = np.sort(arr1)
  arr2 = np.sort(arr2)
  if arr1[0] > arr2[0]:
    arr1, arr2 = arr2, arr1
  class1, class2 = [], []
  for v in arr1:
    if v <= threadHold:
      class1.append(v)
  for v in arr2:
    if v > threadHold:
      class2.append(v)
  return (class1, class2)

def draw_hist(arr1, arr2, name):
  nw_arr1 = Counter(arr1).items()
  nw_arr2 = Counter(arr2).items()
  X,Y = zip(*nw_arr1)
  X2,Y2 = zip(*nw_arr2)
  bar_width = 0.9
  plt.bar(X,Y,bar_width,color="blue",alpha=0.75,label="a")
  bar_width = 0.8
  plt.bar(X2,Y2,bar_width,color="red",alpha=0.75,label="b")
  plt.legend(loc='upper right')
  plt.savefig(name)
  plt.show()

# 1a
a = np.array([1,2,3,2,3,4,5,6,7])
b = np.array([5,5,6,6,7,8,9,9,8])
thread_hold, error = find_TH(a, b)
print("1a")
print("First array :", np.sort(a))
print("Second array:", np.sort(b))
print("Thread Hold :", thread_hold)
print("Number of errors:", error)
nw_a, nw_b = classification(a, b, thread_hold)
print("New first array :", nw_a)
print("New second array:", nw_b)
draw_hist(a, b, "1a.png")
print()

# 1b
a = np.random.normal(25, 15, size=300000).round(0).astype(np.int)
b = np.random.normal(75, 15, size=300000).round(0).astype(np.int)
thread_hold, error = find_TH(a, b)
print("1b")
print("First array :", np.sort(a))
print("Second array:", np.sort(b))
print("Thread Hold :", thread_hold)
print("Number of errors:", error)
nw_a, nw_b = classification(a, b, thread_hold)
print("New first array :", np.sort(nw_a))
print("New second array:", np.sort(nw_b))
draw_hist(a, b, "1b.png")
print()

# 1c
a = np.random.normal(15, 15, size=30000).round(0).astype(np.int)
b = np.random.normal(65, 15, size=30000).round(0).astype(np.int)
file = "inp.csv"

# Write to file
print("Writing to file:", file)
content = np.vstack((a, b))
df = pd.DataFrame(content)
df.to_csv(file, na_rep="NAN!")

# Read from file
print("Reading from file:", file)
df = pd.read_csv(file)
a = np.array(df.iloc[0])
b = np.array(df.iloc[1])
thread_hold, error = find_TH(a, b)

print("1c")
print("First array :", np.sort(a))
print("Second array:", np.sort(b))
print("Thread Hold :", thread_hold)
print("Number of error:", error)
nw_a, nw_b = classification(a, b, thread_hold)
print("New first array :", np.sort(nw_a))
print("New second array:", np.sort(nw_b))
draw_hist(a, b, "1c.png")
print()
