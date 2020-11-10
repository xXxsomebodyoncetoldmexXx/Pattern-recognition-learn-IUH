import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

def load_data(path):
  return pd.read_csv(path, sep="\s+")

# def split_data_train(dataset, per):
#   l = round(len(dataset)*per)
#   return dataset[:l], dataset[l:]


df = load_data("Datasets/twoclass.dat")
df.columns = ["Feature1", "Feature2", "Feature3", "Feature4", "Class1", "Class2"]

print("Cả 2 class đều có số lượng đặc trưng là:", len(df.columns)-2)
print("Số lượng mẫu của Class 1 là:", len(df[df["Class1"] == 1]))
print("Số lượng mẫu của Class 2 là:", len(df[df["Class2"] == 1]))

feature1 = list()
feature2 = list()
class1 = df[df["Class1"] == 1]
class2 = df[df["Class2"] == 1]

print("*CLASS1: ")
for i in range(4):
  feature1.append(class1[f"Feature{i+1}"])
  print(f"Feature{i+1}:")
  print("\t+Mean:", feature1[i].mean())
  print("\t+Var :", feature1[i].var())
  print()

for i in range(4):
  for j in range(i+1, 4):
    print(f"Covariance giữa Feature{i+1} với Feature{j+1} là:")
    print(np.cov(feature1[i], feature1[j]))
    print()

print("*CLASS2: ")
for i in range(4):
  feature2.append(class2[f"Feature{i+1}"])
  print(f"Feature{i+1}:")
  print("\t+Mean:", feature2[i].mean())
  print("\t+Var :", feature2[i].var())
  print()

for i in range(4):
  for j in range(i+1, 4):
    print(f"Covariance giữa Feature{i+1} với Feature{j+1} là:")
    print(np.cov(feature2[i], feature2[j]))
    print()

# Select feature 1
selected_feature = 0

# split dataset into two part to train and test
train, test = train_test_split(list(zip(feature1[selected_feature], feature2[selected_feature])), train_size=0.5)
train_c1, train_c2 = zip(*train)

hist_c1, bin_edge = np.histogram(train_c1)

# print(train_c1)
# print(np.histogram(train_c1))
plt.hist(train_c1, density=True, bins=15, color="red", alpha=0.75, label="Class 1")
plt.hist(train_c2, density=True, bins=15, color="blue", alpha=0.75, label="Class 2")
# plt.ylabel = "Probability"
# plt.xlabel = "Data"
# plt.legend(loc="best")
plt.show()
# print(train)
# print(np.histogram(train))
# print(Counter(train))
