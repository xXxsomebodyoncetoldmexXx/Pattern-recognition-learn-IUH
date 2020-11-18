import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

def load_data(path):
  return pd.read_csv(path, sep="\s+")

def find_TH(arr1, arr2):
  class1 = np.sort(arr1)
  class2 = np.sort(arr2)
  data_set = set(class1).union(class2)
  if class1[0] > class2[0]:
    class1, class2 = class2, class1
  class1 = Counter(class1)
  class2 = Counter(class2)
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

def misclassification(arr1, arr2, threadHold):
  arr1 = np.sort(arr1)
  arr2 = np.sort(arr2)
  if arr1[0] > arr2[0]:
    arr1, arr2 = arr2, arr1
  class1_error, class2_error = arr1.size, arr2.size
  for v in arr1:
    if v <= threadHold:
      class1_error -= 1
  for v in arr2:
    if v > threadHold:
      class2_error -= 1
  return (class1_error, class2_error)

df = load_data("Datasets/twoclass.dat")
df.columns = ["Feature1", "Feature2", "Feature3", "Feature4", "Class1", "Class2"]

print("Cả 2 class đều có số lượng đặc trưng là:", len(df.columns)-2)
print("Số lượng mẫu của Class 1 là:", len(df[df["Class1"] == 1]))
print("Số lượng mẫu của Class 2 là:", len(df[df["Class2"] == 1]))

feature_c1 = list()
feature_c2 = list()
class1 = df[df["Class1"] == 1]
class2 = df[df["Class2"] == 1]

print("*CLASS1: ")
for i in range(4):
  feature_c1.append(class1[f"Feature{i+1}"])
  print(f"Feature{i+1}:")
  print("\t+Mean:", feature_c1[i].mean())
  print("\t+Var :", feature_c1[i].var())
  print()

for i in range(4):
  for j in range(i+1, 4):
    print(f"Covariance giữa Feature{i+1} với Feature{j+1} là:")
    print(np.cov(feature_c1[i], feature_c1[j]))
    print()

print("*CLASS2: ")
for i in range(4):
  feature_c2.append(class2[f"Feature{i+1}"])
  print(f"Feature{i+1}:")
  print("\t+Mean:", feature_c2[i].mean())
  print("\t+Var :", feature_c2[i].var())
  print()

for i in range(4):
  for j in range(i+1, 4):
    print(f"Covariance giữa Feature{i+1} với Feature{j+1} là:")
    print(np.cov(feature_c2[i], feature_c2[j]))
    print()

# Select feature 1
selected_feature = 0

# Select train size
train_pecent = 0.8

# split dataset into two part to train and test
train, test = train_test_split(list(zip(feature_c1[selected_feature], feature_c2[selected_feature])), train_size=train_pecent)
train_c1, train_c2 = zip(*train)
test_c1, test_c2 = zip(*test)

# Convert to array
train_c1 = np.array(train_c1)
train_c2 = np.array(train_c2)
test_c1 = np.array(test_c1)
test_c2 = np.array(test_c2)

plt.title("Biểu đồ histogram của đặc trưng 1 thuộc class1")
plt.hist(train_c1)
# plt.show()
plt.title("Biểu đồ plot của đặc trưng 1 thuộc class1")
plt.plot(train_c1)
# plt.show()

thread_hold, error = find_TH(train_c1, train_c2)
print("Biệt số tìm được là:", thread_hold)
print("Tổng số lỗi trong tập train là:", error)

error_c1, error_c2 = misclassification(test_c1, test_c2, thread_hold)
print("Tổng số lỗi trong tập test là:", error_c1 + error_c2)
print("misclassification class1:", error_c1)
print("misclassification class2:", error_c2)

