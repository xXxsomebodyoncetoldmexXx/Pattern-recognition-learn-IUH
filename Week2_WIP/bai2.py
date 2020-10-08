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
      value = (value // self.bin_w) * bin_w
    return self.freq_dict.get(value, 0)

data = pd.read_csv("person_data.txt", sep=" ", header=None)
data.columns = ["last Name", "first Name", "height", "weight", "gender"]

genders = ["male", "female"]
height = dict()
fts = dict()
for g in genders:
  height[g] = data.loc[data["gender"] == g, "height"]
  fts[g] = Feature(height[g], name=g, bin_w=5)
  print(g, fts[g].freq_dict)
  color = "blue" if g == "male" else "red"
  bar_width = 4 if g == "male" else 3
  plt.bar(*zip(*fts[g].freq_dict.items()), bar_width, color=color, alpha=0.75, label=g)

plt.legend(loc="best")
plt.show()

