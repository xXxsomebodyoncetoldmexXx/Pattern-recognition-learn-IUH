import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from collections import Counter

class Feature:
  def __init__(self, data, name=None, bin_w=None):
    self.name = name
    self.bin_w = bin_w
    if bin_w:
      self.min, self.max = min(data), max(data)
      bins = np.arange((self.min // bin_w) * bin_w,
                       (self.max // bin_w) * bin_w,
                       bin_w)
      self.freq_dict = dict(zip(*np.histogram(data, bins)[::-1]))
    else:
      self.freq_dict = Counter(data)
    self.freq_sum = sum(self.freq_dict.values())

  def get_freq(self, value):
    if self.bin_w:
      value = (value // self.bin_w) * self.bin_w
    return self.freq_dict.get(value, 0)

class NaiveBayes:
  def __init__(self, name, *features):
    self.features = features
    self.name = name

  def prob_value_giving_feature(self, *feature_value):
    '''
    can be know as giving a feature how probability that value is in the giving feature
    P(x | wi)
    '''
    result = 1
    for f, fv in zip(self.features, feature_value):
      if f.freq_sum == 0:
        return 0
      else:
        result *= f.get_freq(fv) / f.freq_sum
    return result

class Classifier:
  def __init__(self, *nbClass):
    self.nbClass = nbClass

  def prob_feature_giving_value(self, data, best_only=True):
    # P(wi | x)
    prob_list = list()
    for nbc in self.nbClass:
      prob_list.append( (nbc.prob_value_giving_feature(*data), nbc.name) )
    prob_sum = sum( [v[0] for v in prob_list] )
    if prob_sum == 0:
      # In case of not found
      # Distribute evenly between every class
      # This make sure the final result <= 1
      number_class = len(self.nbClass)
      prob_list = [ (1/number_class, name) for _, name in prob_list ]
    else:
      prob_list = [ (value/prob_sum, name) for value, name in prob_list ]

    if best_only:
      return max(prob_list)
    else:
      return prob_list

data = pd.read_csv("Datasets/person_data.txt", sep=" ", header=None)
data.columns = ["last Name", "first Name", "height", "weight", "gender"]

genders = ["male", "female"]
height = dict()
name = dict()
fname = dict()
weight = dict()
cls = dict()

for gender in genders:
  # Extract data
  height[gender] = data.loc[data["gender"] == gender, "height"]
  name[gender] = data.loc[data["gender"] == gender, "last Name"]
  fname[gender] = data.loc[data["gender"] == gender, "first Name"]
  weight[gender] = data.loc[data["gender"] == gender, "weight"]
  # Plot the data
#   color = "blue" if gender == "male" else "red"
#   bar_width = 4 if gender == "male" else 3
#   plt.bar(*zip(*fts[gender].freq_dict.items()), bar_width, color=color, alpha=0.75, label=gender)
# plt.legend(loc="best")
# plt.show()

print("Sử dụng đặc trưng chiều cao:")
for gender in genders:
  height_fts = Feature(height[gender], name=gender, bin_w=5)
  cls[gender] = NaiveBayes(gender, height_fts)
c = Classifier(cls["male"], cls["female"])
test = [(140,), (200,), (153,), (188,), (159,), (160,), (180,), (150,), (170,)]
print("data test:", test)
for h in test:
  print(h, c.prob_feature_giving_value(h))
print()

print("Sử dụng đặc trưng tên:")
for gender in genders:
  name_fts = Feature(name[gender], name=gender)
  cls[gender] = NaiveBayes(gender, name_fts)
c = Classifier(cls["male"], cls["female"])
test = [("Edgar",), ("Benjamin",), ("Fred",), ("Albert",), ("Laura",), ("Maria",), ("Paula",), ("Sharon",), ("Jessie",)]
print("data test:", test)
for n in test:
  print(n, c.prob_feature_giving_value(n))
print()

print("Sử dụng đặc trưng tên + chiều cao:")
for gender in genders:
  name_fts =  Feature(name[gender], name=gender)
  height_fts = Feature(height[gender], name=gender, bin_w=5)
  cls[gender] = NaiveBayes(gender, name_fts, height_fts)
c = Classifier(cls["male"], cls["female"])
test = [("Maria", 140), ("Anthony", 200), ("Anthony", 153), ("Jessie", 188), ("Jessie", 159), ("Jessie", 160)]
print("data test:", test)
for name_height in test:
  print(name_height, c.prob_feature_giving_value(name_height))
print()

