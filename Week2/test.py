import numpy as np
from collections import Counter

genders = ["male", "female"]
persons = list()

with open("person_data.txt") as f:
  for line in f:
    persons.append(line.strip().split())

firstnames = dict()
heights = dict()

for gender in genders:
  firstnames[gender] = [ x[0] for x in persons if x[4]==gender]
  heights[gender] = np.array([ x[2] for x in persons if x[4]==gender], np.int)

class Feature:
  def __init__(self, data, name=None, bin_width=None):
    self.name = name
    self.bin_width = bin_width
    if bin_width:
      self.min, self.max = min(data), max(data)
      bins = np.arange((self.min // bin_width)*bin_width,
                       (self.max // bin_width)*bin_width,
                       bin_width)
      freq, bins = np.histogram(data, bins)
      self.freq_dict = dict(zip(bins, freq))
      self.freq_sum = sum(freq)
    else:
      self.freq_dict = dict(Counter(data))
      self.freq_sum = sum(self.freq_dict.values())

  def frequency(self, value):
    if self.bin_width:
      value = (value // bin_width) * bin_width
    return self.freq_dict.get(value, 0)

fts = dict()
for gender in genders:
  fts[gender] = Feature(heights[gender], name=gender, bin_width=5)

class NBclass:
  def __init__(self, name, *features):
    self.name = name
    self.features = features

  def prob_value_given_feature(self, feature_value, feature):
    if feature.freq_sum == 0:
      return 0
    return feature.frequency(feature_value) / feature.freq_sum

cls = dict()
for gender in genders:
  cls[gender] = NBclass(gender, fts[gender])

class Classifier:
  def __init__(self, *NBclasses):
    self.NBclasses = NBclasses

  def prob(self, *d, best_only=True):
    NBclasses = self.NBclasses
    prob_list = list()
    for NBclass in NBclasses:
      ftrs = NBclass.features
      prob = 1
      for i in range(len(ftrs)):
        prob *= NBclass.prob_value_given_feature(d[i], ftrs[i])
      prob_list.append((prob, NBclass.name))
    prob_value = [f[0] for f in prob_list]
    prob_sum = sum(prob_value)
    if prob_sum == 0:
      number_classes = len(self.NBclasses)
      pl = list()
      for prob_ele in prob_list:
        pl.append( ((1/number_classes), prob_ele[1]) )
      prob_list = pl
    else:
      prob_list = [ (p[0] / prob_sum, p[1]) for p in prob_list ]
    if best_only:
      return max(prob_list)
    else:
      return prob_list

fts = dict()
cls = dict()

for gender in genders:
  fts_name = Feature(firstnames[gender], name=gender)
  cls[gender] = NBclass(gender, fts_name)

c = Classifier(cls["male"], cls["female"])
test = ["Edgar", "Benjamin", "Fred", "Albert", "Laura", "Maria", "Paula", "Sharon", "Jessie"]
for name in test:
  print(name, c.prob(name))
  print()
