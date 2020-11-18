import matplotlib.pyplot as plt 
import numpy as np 

in_time = [(0,27), (1,25), (2,16), (3,19), (4,26), (5,20), (6,19), (7,17), (8,10), (9,5), (10,4), (11,4), (12,2)]
cls_late = [(5,3), (6,5), (7,8), (8,15), (9,17), (10,18), (11,19), (12,16), (13,9), (14,8), (15,8)]

X,Y = zip(*in_time)
X2,Y2 = zip(*cls_late)
bar_width = 0.9
plt.bar(X,Y,bar_width,color="blue",alpha=0.75,label="Dung gio")
bar_width = 0.8
plt.bar(X2,Y2,bar_width,color="red",alpha=0.75,label="Tre")
plt.legend(loc='upper right')
plt.savefig("1.png")

in_time_dict = dict(in_time)
too_late_dict = dict(cls_late)

def check_prob_late(min):
    s = in_time_dict.get(min,0)
    if s == 0:
        return 0
    else:
        m = too_late_dict.get(min,0)
        return s/(s+m)

# Copy from Week1/bai1.py
def find_TH(d1, d2):
  class1 = dict(d1)
  class2 = dict(d2)
  data_set = set(class1)
  data_set.union(class2)
  if min(class1) > min(class2):
    class1, class2 = class2, class1
  # v <= tp: class1
  # v >  tp: class2
  num_of_errors = sum(class1.values()) + sum(class2.values())
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

# find threadHold
thread_hold, error = find_TH(in_time, cls_late)


print("Found threadhold at:", thread_hold)
print("Number of error is:", error)
target = ["6:34", "6:35", "6:36", "6:37", "6:38"]
for t in target:
    print("Probability of going late to school when start at", t, "is", 100 - round(check_prob_late(int(t[-1]))*100), "%")
plt.show()
