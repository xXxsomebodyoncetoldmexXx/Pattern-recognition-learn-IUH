import pandas as pd
import numpy as np
from collections import Counter

def load_data(path):
  return pd.read_csv(path, sep="\s+")

def split_data_train(dataset, per):
  l = round(len(dataset)*per)
  return dataset[:l], dataset[l:]


df = load_data("Datasets/twoclass.dat")
df.columns = ["Feature1", "Feature2", "Feature3", "Feature4", "Class1", "Class2"]

print("Cả 2 class đều có số lượng đặc trưng là:", len(df.columns)-2)
print("Số lượng mẫu của Class 1 là:", len(df[df["Class1"] == 1]))
print("Số lượng mẫu của Class 2 là:", len(df[df["Class2"] == 1]))

feature = list()
for i in range(4):
  feature.append(df[f"Feature{i+1}"])
  print(f"Feature{i+1}:")
  print("\t+Mean:", feature[i].mean())
  print("\t+Var :", feature[i].var())
  print()

for i in range(4):
  for j in range(i+1, 4):
    print(f"Covariance giữa Feature{i+1} với Feature{j+1} là:")
    print(np.cov(feature[i], feature[j]))
    print()

# split dataset into two part to train and test
train, test = split_data_train(feature[0], 0.5)
print(train)
print(np.histogram(train))
print(Counter(train))
